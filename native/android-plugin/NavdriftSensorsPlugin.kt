package com.navdrift.zero

import android.Manifest
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import androidx.core.app.ActivityCompat
import com.getcapacitor.JSObject
import com.getcapacitor.Plugin
import com.getcapacitor.PluginCall
import com.getcapacitor.PluginMethod
import com.getcapacitor.annotation.CapacitorPlugin
import com.getcapacitor.annotation.Permission
import com.getcapacitor.annotation.PermissionCallback

// Real native sensor bridge for NAVDRIFT-0's Android build.
//
// This plugin does not run its own navigation logic — it only reads whatever the phone's real
// hardware actually reports and hands it to the SAME SensorManager JS module the web app already
// uses (see frontend/native-bridge.js). The DriftFormer/EKF/road-matching engine in mobile.html
// is completely unaware of whether a reading came from a browser sensor API or from here — one
// engine, one UI, two ways of feeding it real data.
//
// Every sensor reports its OWN honest state. A missing sensor (getDefaultSensor() returning null)
// emits an "unavailable" event — it never gets treated as "zero" or silently skipped, and nothing
// here ever invents a reading. A denied permission emits "permission-denied", not "no data".
@CapacitorPlugin(
    name = "NavdriftSensors",
    permissions = [
        Permission(strings = [Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION], alias = "location")
    ]
)
class NavdriftSensorsPlugin : Plugin(), SensorEventListener, LocationListener {

    private lateinit var sensorManager: SensorManager
    private var locationManager: LocationManager? = null

    private var accelSensor: Sensor? = null   // prefers TYPE_LINEAR_ACCELERATION (gravity already removed)
    private var usingLinearAccel = true
    private var gyroSensor: Sensor? = null
    private var rotationSensor: Sensor? = null      // TYPE_ROTATION_VECTOR — real device orientation
    private var magSensor: Sensor? = null           // TYPE_MAGNETIC_FIELD — real compass hardware
    private var pressureSensor: Sensor? = null      // TYPE_PRESSURE — real barometer, when present

    private var listening = false

    override fun load() {
        sensorManager = activity.getSystemService(android.content.Context.SENSOR_SERVICE) as SensorManager
        locationManager = activity.getSystemService(android.content.Context.LOCATION_SERVICE) as LocationManager

        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        if (accelSensor == null) {
            // Fall back to the raw accelerometer (gravity included) — still real hardware, just a
            // different reference frame. The JS side is told which one it's getting via the
            // "gravityRemoved" flag on each sample, never silently pretending it's the other kind.
            accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
            usingLinearAccel = false
        }
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        magSensor = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
        pressureSensor = sensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE)
    }

    @PluginMethod
    fun startImu(call: PluginCall) {
        if (!listening) {
            accelSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
            gyroSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
            rotationSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
            magSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_UI) }
            pressureSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL) }
            listening = true
        }
        val result = JSObject()
        result.put("accelerometer", if (accelSensor != null) "available" else "unavailable")
        result.put("gyroscope", if (gyroSensor != null) "available" else "unavailable")
        result.put("orientation", if (rotationSensor != null) "available" else "unavailable")
        result.put("magnetometer", if (magSensor != null) "available" else "unavailable")
        result.put("barometer", if (pressureSensor != null) "available" else "unavailable")
        call.resolve(result)
    }

    @PluginMethod
    fun stopImu(call: PluginCall) {
        if (listening) {
            sensorManager.unregisterListener(this)
            listening = false
        }
        call.resolve()
    }

    @PluginMethod
    fun startGnss(call: PluginCall) {
        if (ActivityCompat.checkSelfPermission(activity, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            requestPermissionForAlias("location", call, "gnssPermsCallback")
            return
        }
        beginLocationUpdates(call)
    }

    @PermissionCallback
    private fun gnssPermsCallback(call: PluginCall) {
        if (ActivityCompat.checkSelfPermission(activity, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
            beginLocationUpdates(call)
        } else {
            val evt = JSObject()
            evt.put("status", "permission-denied")
            notifyListeners("gnssStatus", evt)
            call.reject("GNSS permission denied")
        }
    }

    private fun beginLocationUpdates(call: PluginCall) {
        try {
            // Real GPS/GNSS chip via Android's own LocationManager — no Google Play Services
            // dependency required, so this works on any Android device, not just ones with GMS.
            locationManager?.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1000L, 0f, this)
            call.resolve()
        } catch (e: SecurityException) {
            call.reject("Location permission not granted: ${e.message}")
        }
    }

    @PluginMethod
    fun stopGnss(call: PluginCall) {
        locationManager?.removeUpdates(this)
        call.resolve()
    }

    // ── LocationListener — real GNSS fixes only, never synthesized ──
    override fun onLocationChanged(location: Location) {
        val evt = JSObject()
        evt.put("latitude", location.latitude)
        evt.put("longitude", location.longitude)
        evt.put("altitude", if (location.hasAltitude()) location.altitude else null)
        evt.put("accuracy", if (location.hasAccuracy()) location.accuracy else null)
        evt.put("speed", if (location.hasSpeed()) location.speed else null)
        evt.put("course", if (location.hasBearing()) location.bearing else null)
        evt.put("timestamp", location.time)
        notifyListeners("gnssFix", evt)
    }
    override fun onLocationChanged(locations: MutableList<Location>) {
        if (locations.isNotEmpty()) onLocationChanged(locations.last())
    }
    override fun onProviderDisabled(provider: String) {
        val evt = JSObject(); evt.put("status", "no-data"); notifyListeners("gnssStatus", evt)
    }
    override fun onProviderEnabled(provider: String) {}
    @Deprecated("Deprecated in Java")
    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}

    // ── SensorEventListener — real IMU/magnetometer/barometer samples only ──
    private val rotMatrix = FloatArray(9)
    private val orientVals = FloatArray(3)

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_LINEAR_ACCELERATION, Sensor.TYPE_ACCELEROMETER -> {
                val evt = JSObject()
                evt.put("x", event.values[0]); evt.put("y", event.values[1]); evt.put("z", event.values[2])
                evt.put("gravityRemoved", usingLinearAccel)
                notifyListeners("imuAccel", evt)
            }
            Sensor.TYPE_GYROSCOPE -> {
                val evt = JSObject()
                evt.put("x", event.values[0]); evt.put("y", event.values[1]); evt.put("z", event.values[2])
                notifyListeners("imuGyro", evt)
            }
            Sensor.TYPE_ROTATION_VECTOR -> {
                SensorManager.getRotationMatrixFromVector(rotMatrix, event.values)
                SensorManager.getOrientation(rotMatrix, orientVals)
                val evt = JSObject()
                // radians -> degrees, matching the web DeviceOrientationEvent convention (alpha/beta/gamma)
                evt.put("alpha", Math.toDegrees(orientVals[0].toDouble()))
                evt.put("beta", Math.toDegrees(orientVals[1].toDouble()))
                evt.put("gamma", Math.toDegrees(orientVals[2].toDouble()))
                notifyListeners("orientationSample", evt)
                // Real compass heading derived from the SAME rotation vector the OS computes from
                // its own sensor fusion (magnetometer + accelerometer + gyroscope) — this is a real
                // heading, not a placeholder, and only ever sent while the rotation sensor itself
                // is genuinely reporting.
                val headingDeg = (Math.toDegrees(orientVals[0].toDouble()) + 360.0) % 360.0
                val magEvt = JSObject()
                magEvt.put("headingDeg", headingDeg)
                notifyListeners("magSample", magEvt)
            }
            Sensor.TYPE_PRESSURE -> {
                val hpa = event.values[0]
                val altitudeM = SensorManager.getAltitude(SensorManager.PRESSURE_STANDARD_ATMOSPHERE, hpa)
                val evt = JSObject()
                evt.put("pressureHPa", hpa)
                evt.put("altitudeM", altitudeM)
                notifyListeners("baroSample", evt)
            }
        }
    }
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun handleOnDestroy() {
        if (listening) sensorManager.unregisterListener(this)
        locationManager?.removeUpdates(this)
    }
}
