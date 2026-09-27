import Foundation
import Capacitor
import CoreLocation
import CoreMotion

// Real native sensor bridge for NAVDRIFT-0's iOS build — the exact counterpart of
// native/android-plugin/NavdriftSensorsPlugin.kt, implementing the SAME plugin method/event
// names so frontend/native-bridge.js (and the SensorManager it feeds) doesn't know or care which
// platform it's running on.
//
// THIS FILE CANNOT BE COMPILED, SIGNED, OR RUN WITHOUT A MAC RUNNING XCODE. That is a real,
// unavoidable Apple platform requirement — not a workaround being avoided. Everything up to that
// point (the actual native sensor access code) is written and ready here; the remaining step is
// opening ios/App/App.xcworkspace in Xcode on a Mac, adding this file to the App target, adding
// the Info.plist usage-description keys listed at the bottom of this file, and building/signing
// with an Apple Developer account. Nothing else in NAVDRIFT-0 (web app, Android app) depends on
// this ever happening.
//
// This is exactly the answer to "the iPhone has a barometer but Safari can't access it": on the
// WEB build, WebKit's Barometer API restriction really does mean no access (see BaroSensor in
// mobile.html). Here, in a real native app, CMAltimeter reads the SAME physical pressure sensor
// directly — no WebKit in the way at all. Same hardware, different access path.
@objc(NavdriftSensorsPlugin)
public class NavdriftSensorsPlugin: CAPPlugin, CLLocationManagerDelegate {

    private let locationManager = CLLocationManager()
    private let motionManager = CMMotionManager()
    private let altimeter = CMAltimeter()
    private var imuStarted = false

    public override func load() {
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
    }

    @objc func startGnss(_ call: CAPPluginCall) {
        let status = CLLocationManager.authorizationStatus()
        switch status {
        case .notDetermined:
            locationManager.requestWhenInUseAuthorization()
            // onAuthChange (below) starts updates once the user actually answers the prompt —
            // never assumed granted just because it was requested.
        case .denied, .restricted:
            notifyListeners("gnssStatus", data: ["status": "permission-denied"])
            call.reject("Location permission denied")
            return
        default:
            locationManager.startUpdatingLocation()
        }
        call.resolve()
    }

    @objc func stopGnss(_ call: CAPPluginCall) {
        locationManager.stopUpdatingLocation()
        call.resolve()
    }

    public func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        if manager.authorizationStatus == .authorizedWhenInUse || manager.authorizationStatus == .authorizedAlways {
            manager.startUpdatingLocation()
        } else if manager.authorizationStatus == .denied || manager.authorizationStatus == .restricted {
            notifyListeners("gnssStatus", data: ["status": "permission-denied"])
        }
    }

    public func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let loc = locations.last else { return }
        // hasAccuracy/hasSpeed style honesty: CoreLocation reports -1 for "not available" on
        // several of these fields, so we translate that into a real null rather than passing -1
        // through as if it were a genuine reading.
        var data: [String: Any] = [
            "latitude": loc.coordinate.latitude,
            "longitude": loc.coordinate.longitude,
            "accuracy": loc.horizontalAccuracy >= 0 ? loc.horizontalAccuracy : NSNull(),
            "altitude": loc.verticalAccuracy >= 0 ? loc.altitude : NSNull(),
            "speed": loc.speed >= 0 ? loc.speed : NSNull(),
            "course": loc.course >= 0 ? loc.course : NSNull(),
            "timestamp": loc.timestamp.timeIntervalSince1970 * 1000
        ]
        notifyListeners("gnssFix", data: data)
    }

    @objc func startImu(_ call: CAPPluginCall) {
        var result: [String: Any] = [:]

        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 1.0 / 30.0
            // .xMagneticNorthZVertical gives us a real device-relative attitude AND folds the
            // magnetometer into Apple's own real sensor-fusion heading — same principle as the
            // Android plugin's rotation-vector heading, just Apple's implementation of it.
            motionManager.startDeviceMotionUpdates(using: .xMagneticNorthZVertical, to: .main) { [weak self] motion, error in
                guard let self = self, let motion = motion else { return }
                self.notifyListeners("imuAccel", data: [
                    "x": motion.userAcceleration.x * 9.81,
                    "y": motion.userAcceleration.y * 9.81,
                    "z": motion.userAcceleration.z * 9.81,
                    "gravityRemoved": true // CMDeviceMotion's userAcceleration is already gravity-free
                ])
                self.notifyListeners("imuGyro", data: [
                    "x": motion.rotationRate.x, "y": motion.rotationRate.y, "z": motion.rotationRate.z
                ])
                self.notifyListeners("orientationSample", data: [
                    "alpha": motion.attitude.yaw * 180 / .pi,
                    "beta": motion.attitude.pitch * 180 / .pi,
                    "gamma": motion.attitude.roll * 180 / .pi
                ])
                if let heading = motion.heading as Double?, heading >= 0 {
                    self.notifyListeners("magSample", data: ["headingDeg": heading])
                }
            }
            result["accelerometer"] = "available"
            result["gyroscope"] = "available"
            result["orientation"] = "available"
            result["magnetometer"] = motionManager.isMagnetometerAvailable ? "available" : "unavailable"
        } else {
            result["accelerometer"] = "unavailable"
            result["gyroscope"] = "unavailable"
            result["orientation"] = "unavailable"
            result["magnetometer"] = "unavailable"
        }

        // CMAltimeter reads the real hardware barometer directly — this is the whole point of
        // building a native app: no WebKit Barometer-API restriction exists on this path at all.
        if CMAltimeter.isRelativeAltitudeAvailable() {
            altimeter.startRelativeAltitudeUpdates(to: .main) { [weak self] data, error in
                guard let self = self, let data = data else { return }
                self.notifyListeners("baroSample", data: [
                    "pressureHPa": data.pressure.doubleValue * 10.0, // kPa -> hPa
                    "relativeAltitudeM": data.relativeAltitude.doubleValue
                ])
            }
            result["barometer"] = "available"
        } else {
            result["barometer"] = "unavailable"
        }

        imuStarted = true
        call.resolve(result)
    }

    @objc func stopImu(_ call: CAPPluginCall) {
        if imuStarted {
            motionManager.stopDeviceMotionUpdates()
            if CMAltimeter.isRelativeAltitudeAvailable() { altimeter.stopRelativeAltitudeUpdates() }
            imuStarted = false
        }
        call.resolve()
    }
}

/*
 REQUIRED Info.plist keys (add these in Xcode, Info tab, before building — the app crashes on
 launch without them once location/motion APIs are actually called):

   NSLocationWhenInUseUsageDescription
     "NAVDRIFT-0 uses your location to fuse GNSS with motion sensors for dead reckoning."
   NSMotionUsageDescription
     "NAVDRIFT-0 uses motion sensors (accelerometer, gyroscope, barometer) for dead reckoning
      during GPS signal loss."

 Registration: this plugin needs a matching entry in ios/App/App/capacitor.config.json's
 generated plugin registry (Capacitor's `npx cap sync ios` does this automatically once this
 file is part of the Xcode project) and a bridging reference in
 ios/App/App/AppDelegate.swift is NOT required for Capacitor 6 — plugins are auto-registered by
 the @objc name and CAPPlugin subclass, same mechanism as the Android @CapacitorPlugin annotation.
*/
