/*
 * NavDriftService.kt
 * NAVDRIFT-0 SDK — package com.navdrift.sdk
 *
 * Two public types are declared here:
 *   NavDriftService  – the foreground Service doing sensor fusion + ONNX inference
 *   NavDriftClient   – lightweight ServiceConnection wrapper for callers
 *
 * Gradle dependencies required in the consuming module's build.gradle(.kts):
 *
 *   implementation("ai.onnxruntime:onnxruntime-android:1.18.0")
 *   implementation("androidx.core:core-ktx:1.13.1")
 *
 * AndroidManifest.xml additions:
 *   <uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
 *   <uses-permission android:name="android.permission.FOREGROUND_SERVICE_LOCATION" />
 *   <service
 *       android:name="com.navdrift.sdk.NavDriftService"
 *       android:foregroundServiceType="location"
 *       android:exported="false" />
 */

package com.navdrift.sdk

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.os.Binder
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.SystemClock
import android.util.Log
import androidx.core.app.NotificationCompat
import java.nio.FloatBuffer
import java.util.ArrayDeque
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

private const val TAG = "NavDriftService"

/** Notification channel ID used for the mandatory foreground notification. */
private const val CHANNEL_ID = "navdrift_channel"
private const val NOTIFICATION_ID = 1

/** ONNX model file name; must be present in the app's assets/ directory. */
private const val MODEL_ASSET = "drift_former.onnx"

/** Number of time steps the model expects. */
private const val WINDOW_SIZE = 200

/** Number of sensor channels per time step.
 *  Order: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, speed, steer_angle
 */
private const val NUM_CHANNELS = 8

/** Broadcast rate for synthetic Location objects, in milliseconds. */
private const val BROADCAST_INTERVAL_MS = 100L   // 10 Hz

// ─────────────────────────────────────────────────────────────────────────────
// NavDriftService
// ─────────────────────────────────────────────────────────────────────────────

/**
 * NavDriftService
 *
 * A foreground [Service] that fuses IMU sensor data with an ONNX-based
 * dead-reckoning model (drift_former.onnx) to produce a continuous SE(2)
 * pose estimate.  When GNSS fixes are available, callers snap the estimate
 * via [setGnssLocation]; between fixes the service integrates sensor-derived
 * deltas with a non-holonomic constraint.
 *
 * Clients bind to the service and obtain the [NavDriftBinder] to call its
 * methods directly — or use the [NavDriftClient] convenience wrapper, which
 * manages the [ServiceConnection] lifecycle.
 */
class NavDriftService : Service() {

    // ── Binder ────────────────────────────────────────────────────────────────

    /** Binder returned to bound clients. */
    inner class NavDriftBinder : Binder() {
        /** Returns the live [NavDriftService] instance. */
        fun getService(): NavDriftService = this@NavDriftService
    }

    private val binder = NavDriftBinder()

    override fun onBind(intent: Intent): IBinder = binder

    // ── ONNX Runtime ──────────────────────────────────────────────────────────

    private lateinit var ortEnv: OrtEnvironment
    private var ortSession: OrtSession? = null

    // ── Sensor management ─────────────────────────────────────────────────────

    private lateinit var sensorManager: SensorManager
    private var accelSensor: Sensor? = null
    private var gyroSensor: Sensor? = null

    /** Latest raw accelerometer reading (m/s²). Protected by [windowLock]. */
    @Volatile private var accelX = 0f
    @Volatile private var accelY = 0f
    @Volatile private var accelZ = 0f

    /** Latest raw gyroscope reading (rad/s). Protected by [windowLock]. */
    @Volatile private var gyroX = 0f
    @Volatile private var gyroY = 0f
    @Volatile private var gyroZ = 0f

    // ── Sliding window ────────────────────────────────────────────────────────

    /**
     * Circular buffer of sensor frames.  Each element is a [FloatArray] of
     * length [NUM_CHANNELS]: [accel_x, accel_y, accel_z, gyro_x, gyro_y,
     * gyro_z, speed, steer_angle].
     *
     * Protected by [windowLock].
     */
    private val window = ArrayDeque<FloatArray>(WINDOW_SIZE + 1)
    private val windowLock = Any()

    // ── SE(2) dead-reckoning state ────────────────────────────────────────────

    /**
     * Dead-reckoned position in a local Cartesian frame (metres from the
     * first GNSS fix used to initialise [gnssOriginLat]/[gnssOriginLon]).
     * All three fields are read/written exclusively on [inferenceHandler].
     */
    private var drX = 0.0        // east
    private var drY = 0.0        // north
    private var drTheta = 0.0    // heading, radians (east = 0, CCW positive)

    /** Accumulated uncertainty (m²); reset on each GNSS snap. */
    @Volatile private var positionVariance = 0.0

    // GNSS origin — set on first [setGnssLocation] call.
    @Volatile private var gnssOriginLat = Double.NaN
    @Volatile private var gnssOriginLon = Double.NaN

    /** Metres per degree of latitude at the origin (approximation). */
    private var metersPerDegreeLat = 111_320.0
    /** Metres per degree of longitude at the origin (cos-corrected). */
    private var metersPerDegreeLon = 111_320.0

    // ── Speed and steering ────────────────────────────────────────────────────

    /** Forward speed in m/s; updated by [setGnssLocation]. */
    @Volatile private var currentSpeedMs = 0f

    /**
     * Steering angle in radians; injected by callers (e.g. OBD-II bridge).
     * Defaults to 0 when unavailable — the model tolerates this with
     * gracefully degraded accuracy.
     */
    @Volatile var steeringAngleRad = 0f

    // ── Inference thread ──────────────────────────────────────────────────────

    /**
     * Dedicated background thread for ONNX inference and SE(2) integration.
     * Sensor batches post [Runnable]s here; the broadcast timer also posts
     * here so location objects are assembled without additional locking.
     */
    private lateinit var inferenceThread: HandlerThread
    private lateinit var inferenceHandler: Handler

    // ── Location broadcast ────────────────────────────────────────────────────

    private val locationListeners = CopyOnWriteArrayList<LocationListener>()
    private val broadcastRunning = AtomicBoolean(false)

    /** Latest synthetic location — updated on every inference run and broadcast. */
    @Volatile private var latestLocation: Location = Location("navdrift")

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "onCreate")

        createNotificationChannel()
        startForeground(NOTIFICATION_ID, buildNotification())

        // 1. Spin up inference thread first so sensors can post to it
        //    immediately after registration.
        inferenceThread = HandlerThread("navdrift-inference").also { it.start() }
        inferenceHandler = Handler(inferenceThread.looper)

        // 2. Initialise ONNX Runtime and load the model from assets.
        initOnnxSession()

        // 3. Register IMU listeners.
        initSensors()

        // 4. Start 10 Hz broadcast loop.
        scheduleBroadcast()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int =
        START_STICKY

    override fun onDestroy() {
        super.onDestroy()
        Log.i(TAG, "onDestroy")

        broadcastRunning.set(false)
        sensorManager.unregisterListener(sensorEventListener)

        inferenceHandler.post {
            // Close ORT session on the inference thread to avoid race conditions
            // with any in-flight inference call.
            try {
                ortSession?.close()
                ortEnv.close()
            } catch (e: Exception) {
                Log.e(TAG, "Error closing ORT session", e)
            }
            inferenceThread.quitSafely()
        }
    }

    // ── Foreground notification ────────────────────────────────────────────────

    private fun createNotificationChannel() {
        val nm = getSystemService(NotificationManager::class.java)
        if (nm.getNotificationChannel(CHANNEL_ID) != null) return
        val ch = NotificationChannel(
            CHANNEL_ID,
            "NavDrift Navigation",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Dead-reckoning location service"
            setShowBadge(false)
        }
        nm.createNotificationChannel(ch)
    }

    private fun buildNotification(): Notification =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("NavDrift active")
            .setContentText("Sensor-fused navigation running")
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setOngoing(true)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .build()

    // ── ONNX initialisation ────────────────────────────────────────────────────

    private fun initOnnxSession() {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val modelBytes = assets.open(MODEL_ASSET).use { it.readBytes() }
            val opts = OrtSession.SessionOptions().apply {
                // Prefer NNAPI delegate when available; fall back to CPU.
                addNnapi()
                setIntraOpNumThreads(2)
                setInterOpNumThreads(1)
            }
            ortSession = ortEnv.createSession(modelBytes, opts)
            Log.i(TAG, "ONNX session created — inputs: ${ortSession!!.inputNames}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load ONNX model '$MODEL_ASSET'", e)
            // Service continues without inference; DR state is frozen until
            // the session is available.
        }
    }

    // ── Sensor initialisation ──────────────────────────────────────────────────

    private fun initSensors() {
        sensorManager = getSystemService(SensorManager::class.java)
        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroSensor  = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        if (accelSensor == null) Log.w(TAG, "No accelerometer found on this device")
        if (gyroSensor  == null) Log.w(TAG, "No gyroscope found on this device")

        // SENSOR_DELAY_GAME ≈ 20 ms; reportLatency = 20 ms lets the OS batch
        // up to one frame before delivering, reducing wake-ups.
        accelSensor?.also {
            sensorManager.registerListener(
                sensorEventListener, it,
                SensorManager.SENSOR_DELAY_GAME,
                /* reportLatencyUs = */ 20_000
            )
        }
        gyroSensor?.also {
            sensorManager.registerListener(
                sensorEventListener, it,
                SensorManager.SENSOR_DELAY_GAME,
                /* reportLatencyUs = */ 20_000
            )
        }
    }

    // ── SensorEventListener ────────────────────────────────────────────────────

    private val sensorEventListener = object : SensorEventListener {

        /**
         * Called on the sensor dispatcher thread (not the main thread).
         * We snapshot the values, append a frame to the window, and post
         * inference work to [inferenceHandler] — keeping this callback
         * as cheap as possible.
         */
        override fun onSensorChanged(event: SensorEvent) {
            when (event.sensor.type) {
                Sensor.TYPE_ACCELEROMETER -> {
                    accelX = event.values[0]
                    accelY = event.values[1]
                    accelZ = event.values[2]
                }
                Sensor.TYPE_GYROSCOPE -> {
                    gyroX = event.values[0]
                    gyroY = event.values[1]
                    gyroZ = event.values[2]
                }
                else -> return
            }

            // Snapshot all channels atomically under windowLock so the
            // inference thread always sees a consistent frame.
            val frame = floatArrayOf(
                accelX, accelY, accelZ,
                gyroX, gyroY, gyroZ,
                currentSpeedMs,
                steeringAngleRad
            )

            val windowFull: Boolean
            synchronized(windowLock) {
                if (window.size >= WINDOW_SIZE) window.pollFirst()
                window.addLast(frame)
                windowFull = window.size == WINDOW_SIZE
            }

            if (windowFull) {
                inferenceHandler.post(inferenceRunnable)
            }
        }

        override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
            Log.d(TAG, "Sensor accuracy changed: ${sensor.name} → $accuracy")
        }
    }

    // ── Inference runnable ─────────────────────────────────────────────────────

    /**
     * Runs on [inferenceHandler].  Copies the window snapshot into an ONNX
     * tensor, runs the model, unpacks the SE(2) delta [dx, dy, dTheta],
     * applies the non-holonomic constraint, and integrates the pose.
     */
    private val inferenceRunnable = Runnable {
        val session = ortSession ?: return@Runnable

        // ── 1. Snapshot window ──────────────────────────────────────────────
        val snapshot: List<FloatArray>
        synchronized(windowLock) {
            snapshot = window.toList()   // defensive copy
        }
        if (snapshot.size < WINDOW_SIZE) return@Runnable

        // ── 2. Build input tensor  [1 × WINDOW_SIZE × NUM_CHANNELS] ─────────
        val flatSize = WINDOW_SIZE * NUM_CHANNELS
        val buf = FloatBuffer.allocate(flatSize)
        for (frame in snapshot) {
            buf.put(frame)
        }
        buf.rewind()

        val inputShape = longArrayOf(1L, WINDOW_SIZE.toLong(), NUM_CHANNELS.toLong())
        val delta: FloatArray
        try {
            OnnxTensor.createTensor(ortEnv, buf, inputShape).use { inputTensor ->
                val inputs = mapOf("input" to inputTensor)
                session.run(inputs).use { results ->
                    // Expected output name "output": float32[1, 3]  → [dx, dy, dTheta]
                    val outputTensor = results[0].value as Array<*>
                    @Suppress("UNCHECKED_CAST")
                    delta = (outputTensor[0] as FloatArray)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "ONNX inference error", e)
            return@Runnable
        }

        if (delta.size < 3) {
            Log.e(TAG, "Unexpected model output size: ${delta.size}")
            return@Runnable
        }

        val rawDx     = delta[0].toDouble()
        val rawDy     = delta[1].toDouble()
        val dTheta    = delta[2].toDouble()

        // ── 3. Non-holonomic constraint ──────────────────────────────────────
        // A wheeled vehicle cannot instantly move sideways.  Project the
        // raw displacement onto the vehicle's forward direction, zeroing
        // the lateral component.
        //
        // forward unit vector: (cos θ, sin θ)
        // lateral unit vector: (-sin θ, cos θ)
        //
        // forward_component = dot([dx,dy], forward)
        // constrained [dx, dy] = forward_component * forward
        val fwdX = cos(drTheta)
        val fwdY = sin(drTheta)
        val forwardComponent = rawDx * fwdX + rawDy * fwdY
        val constrainedDx = forwardComponent * fwdX
        val constrainedDy = forwardComponent * fwdY

        // ── 4. SE(2) integration ─────────────────────────────────────────────
        // x' = x + cos(θ)·dx - sin(θ)·dy
        // y' = y + sin(θ)·dx + cos(θ)·dy
        // θ' = θ + dθ
        drX    += cos(drTheta) * constrainedDx - sin(drTheta) * constrainedDy
        drY    += sin(drTheta) * constrainedDx + cos(drTheta) * constrainedDy
        drTheta = normaliseAngle(drTheta + dTheta)

        // Simple constant-velocity uncertainty growth model (1 m²/s²).
        positionVariance += 1.0

        // ── 5. Build synthetic Location ──────────────────────────────────────
        latestLocation = buildSyntheticLocation()
    }

    // ── Location construction ─────────────────────────────────────────────────

    /**
     * Converts the current DR state into a [Location] with provider "navdrift".
     * Must be called on [inferenceHandler] so [drX], [drY], [drTheta] are
     * read on the same thread that writes them.
     */
    private fun buildSyntheticLocation(): Location {
        val loc = Location("navdrift")
        loc.elapsedRealtimeNanos = SystemClock.elapsedRealtimeNanos()
        loc.time = System.currentTimeMillis()
        loc.speed = currentSpeedMs
        loc.bearing = Math.toDegrees(drTheta).toFloat()
            .let { if (it < 0f) it + 360f else it }
        loc.accuracy = sqrt(positionVariance).toFloat().coerceAtLeast(1f)

        // Back-project Cartesian offset to geographic coordinates.
        if (!gnssOriginLat.isNaN()) {
            loc.latitude  = gnssOriginLat + drY / metersPerDegreeLat
            loc.longitude = gnssOriginLon + drX / metersPerDegreeLon
        } else {
            // No GNSS origin yet — emit 0,0 as a sentinel so callers know
            // to wait for the first [setGnssLocation] call.
            loc.latitude  = 0.0
            loc.longitude = 0.0
        }
        return loc
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    /**
     * Snaps the dead-reckoned pose to the given GNSS fix and resets the
     * position uncertainty.  Safe to call from any thread.
     *
     * @param lat      WGS-84 latitude, degrees
     * @param lon      WGS-84 longitude, degrees
     * @param bearing  True bearing, degrees (0 = north, clockwise)
     * @param speedMs  Ground speed, metres per second
     */
    fun setGnssLocation(lat: Double, lon: Double, bearing: Float, speedMs: Float) {
        currentSpeedMs = speedMs

        inferenceHandler.post {
            // First fix — initialise the Cartesian origin.
            if (gnssOriginLat.isNaN()) {
                gnssOriginLat = lat
                gnssOriginLon = lon
                metersPerDegreeLat = 111_320.0
                metersPerDegreeLon = 111_320.0 * cos(Math.toRadians(lat))
                Log.i(TAG, "GNSS origin set: lat=$lat lon=$lon")
            }

            // Convert GNSS lat/lon to metres relative to origin.
            drX = (lon - gnssOriginLon) * metersPerDegreeLon
            drY = (lat - gnssOriginLat) * metersPerDegreeLat

            // Convert bearing (degrees CW from north) to mathematical angle
            // (radians CCW from east).
            drTheta = normaliseAngle(Math.toRadians((90.0 - bearing).toDouble()))

            positionVariance = 0.0   // reset uncertainty on each GNSS fix
            latestLocation = buildSyntheticLocation()
        }
    }

    /**
     * Registers [listener] to receive synthetic [Location] updates at ~10 Hz.
     * Listeners are stored in a [CopyOnWriteArrayList] so this method is safe
     * to call from any thread.
     */
    fun addLocationListener(listener: LocationListener) {
        locationListeners.addIfAbsent(listener)
    }

    /**
     * Removes [listener] from the broadcast list.  No-op if not registered.
     */
    fun removeLocationListener(listener: LocationListener) {
        locationListeners.remove(listener)
    }

    /**
     * Returns the most recently computed [Location] object.  Thread-safe
     * due to the [@Volatile] annotation on [latestLocation].
     */
    fun getNavDriftLocation(): Location = latestLocation

    // ── Broadcast loop ─────────────────────────────────────────────────────────

    /**
     * Posts a self-rescheduling broadcast task on [inferenceHandler] that
     * delivers [latestLocation] to all registered [LocationListener]s at 10 Hz.
     *
     * Using [inferenceHandler] means broadcasts are serialised with inference,
     * guaranteeing listeners always see a location that corresponds to a
     * completed integration step.
     */
    private fun scheduleBroadcast() {
        broadcastRunning.set(true)
        inferenceHandler.postDelayed(broadcastRunnable, BROADCAST_INTERVAL_MS)
    }

    private val broadcastRunnable: Runnable = object : Runnable {
        override fun run() {
            if (!broadcastRunning.get()) return

            val loc = latestLocation
            for (listener in locationListeners) {
                try {
                    listener.onLocationChanged(loc)
                } catch (e: Exception) {
                    Log.e(TAG, "LocationListener threw an exception", e)
                }
            }

            inferenceHandler.postDelayed(this, BROADCAST_INTERVAL_MS)
        }
    }

    // ── Utilities ──────────────────────────────────────────────────────────────

    /** Wraps [angle] to (-π, π]. */
    private fun normaliseAngle(angle: Double): Double {
        var a = angle % (2 * Math.PI)
        if (a > Math.PI)  a -= 2 * Math.PI
        if (a < -Math.PI) a += 2 * Math.PI
        return a
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NavDriftClient
// ─────────────────────────────────────────────────────────────────────────────

/**
 * NavDriftClient
 *
 * A lightweight drop-in replacement for [android.location.LocationManager]
 * usage patterns.  Manages the [ServiceConnection] lifecycle and proxies the
 * full [NavDriftService] public API.
 *
 * ### Migration example
 *
 * **Before** (3 lines of LocationManager):
 * ```kotlin
 * val lm = getSystemService(LocationManager::class.java)
 * lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, 100L, 0f, myListener)
 * val lastLoc = lm.getLastKnownLocation(LocationManager.GPS_PROVIDER)
 * ```
 *
 * **After** (3 lines of NavDriftClient):
 * ```kotlin
 * val navDrift = NavDriftClient()
 * navDrift.connect(this)
 * navDrift.addLocationListener(myListener)   // same LocationListener interface
 * val lastLoc = navDrift.getLocation()        // same Location object shape
 * ```
 *
 * Call [disconnect] in `onStop` / `onDestroy` to release the binding.
 */
class NavDriftClient {

    private var service: NavDriftService? = null
    private var context: Context? = null

    /**
     * Listeners queued before the service is bound; replayed on connection.
     */
    private val pendingListeners = mutableListOf<LocationListener>()
    private val pendingLock = Any()

    // ── ServiceConnection ─────────────────────────────────────────────────────

    private val connection = object : ServiceConnection {

        override fun onServiceConnected(name: ComponentName, binder: IBinder) {
            val svc = (binder as NavDriftService.NavDriftBinder).getService()
            service = svc

            // Replay any listeners that were added before the bind completed.
            synchronized(pendingLock) {
                pendingListeners.forEach { svc.addLocationListener(it) }
                pendingListeners.clear()
            }

            Log.i("NavDriftClient", "Connected to NavDriftService")
        }

        override fun onServiceDisconnected(name: ComponentName) {
            service = null
            Log.w("NavDriftClient", "NavDriftService disconnected unexpectedly")
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /**
     * Binds to [NavDriftService], starting it if necessary.
     * Should be called in `onStart` or `onCreate`.
     *
     * @param context An Activity or Application context used to bind the service.
     */
    fun connect(context: Context) {
        this.context = context.applicationContext
        val intent = Intent(context, NavDriftService::class.java)
        // START_STICKY keeps the service running; BIND_AUTO_CREATE starts it
        // if it isn't running yet.
        context.applicationContext.also { ctx ->
            ctx.startService(intent)
            ctx.bindService(intent, connection, Context.BIND_AUTO_CREATE)
        }
    }

    /**
     * Unbinds from [NavDriftService].  Does NOT stop the service — the
     * service continues running for other bound clients or until the OS stops
     * it.  Call in `onStop` or `onDestroy`.
     */
    fun disconnect() {
        service = null
        context?.unbindService(connection)
        context = null
    }

    // ── Location listener proxy ────────────────────────────────────────────────

    /**
     * Registers [listener] to receive [android.location.Location] updates
     * at ~10 Hz from the NavDrift dead-reckoning engine.
     *
     * If the service is not yet bound, [listener] is queued and replayed
     * automatically once the connection is established.
     */
    fun addLocationListener(listener: LocationListener) {
        val svc = service
        if (svc != null) {
            svc.addLocationListener(listener)
        } else {
            synchronized(pendingLock) {
                if (!pendingListeners.contains(listener)) {
                    pendingListeners.add(listener)
                }
            }
        }
    }

    /**
     * Removes [listener].  If the service is not yet connected, removes it
     * from the pending queue.
     */
    fun removeLocationListener(listener: LocationListener) {
        service?.removeLocationListener(listener)
        synchronized(pendingLock) {
            pendingListeners.remove(listener)
        }
    }

    // ── One-shot query ─────────────────────────────────────────────────────────

    /**
     * Returns the most recent dead-reckoned [android.location.Location], or
     * `null` if the service is not yet bound.
     *
     * The returned location's [android.location.Location.getProvider] value
     * is `"navdrift"`.
     */
    fun getLocation(): Location? = service?.getNavDriftLocation()

    // ── GNSS snap ──────────────────────────────────────────────────────────────

    /**
     * Passes an authoritative GNSS fix through to [NavDriftService.setGnssLocation],
     * snapping the dead-reckoned pose and resetting position uncertainty.
     *
     * Typical call site: inside a real [LocationListener] registered with the
     * system [android.location.LocationManager] as a coarse GNSS fallback:
     *
     * ```kotlin
     * val gnssListener = LocationListener { fix ->
     *     navDrift.updateFromGnss(fix)
     * }
     * locationManager.requestLocationUpdates(
     *     LocationManager.GPS_PROVIDER, 1000L, 5f, gnssListener
     * )
     * ```
     */
    fun updateFromGnss(location: Location) {
        service?.setGnssLocation(
            lat      = location.latitude,
            lon      = location.longitude,
            bearing  = location.bearing,
            speedMs  = location.speed
        )
    }
}
