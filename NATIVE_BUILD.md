# Building NAVDRIFT-0 as a real native app (Android + iOS)

## What this is, in plain terms

The web app in `frontend/` is the one and only NAVDRIFT-0. Nothing here is a rewrite or a second
copy of it. Capacitor takes that exact same HTML/CSS/JS and wraps it in a real native app shell,
so the app you already approved (UI, map styling, light/dark mode, everything) looks and behaves
identically — the only difference is that on a native build, `frontend/native-bridge.js` detects
it's running natively and switches from browser sensor APIs to a real native sensor plugin
(`native/android-plugin/NavdriftSensorsPlugin.kt`, `native/ios-plugin/NavdriftSensorsPlugin.swift`)
for GPS, accelerometer, gyroscope, orientation, magnetometer and barometer. If that plugin isn't
present for any reason, the web sensor code keeps working exactly as it does today — the web
version never stops being a complete, working fallback.

## Android — what you need to run yourself

I can't run Gradle or Android Studio from here, so these are the exact commands to run on your
own machine (Windows is fine — you already have the repo at `D:\navdrift0-main`). You'll need
Node.js and Android Studio installed; Android Studio's installer also installs the Android SDK
and a working Gradle, so that's the only thing to install if you don't have either already.

```
cd D:\navdrift0-main
npm install
npx cap add android
```

`npx cap add android` generates the actual Gradle project (`android/` — note: this will replace
the stale `android/NavDriftService.kt` file that's sitting in the repo right now; that file is
leftover HTML from an old build, not real Kotlin, and isn't used by anything — safe to lose).

Then copy the real native sensor plugin into the generated project:

```
copy native\android-plugin\NavdriftSensorsPlugin.kt android\app\src\main\java\com\navdrift\zero\NavdriftSensorsPlugin.kt
```

(Create the `com\navdrift\zero` folders if `cap add android` didn't already lay out that package
path — it should, since `capacitor.config.json`'s `appId` is `com.navdrift.zero`.)

Register the plugin in your Android app's main activity — open
`android\app\src\main\java\com\navdrift\zero\MainActivity.java` (or `.kt`) and add, inside
`onCreate` before `super.onCreate` returns control (Capacitor's generated template has a spot for
exactly this — `registerPlugin(...)` calls):

```java
registerPlugin(NavdriftSensorsPlugin.class);
```

Add location permissions to `android\app\src\main\AndroidManifest.xml` (inside the `<manifest>`
tag, alongside whatever Capacitor already put there):

```xml
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
<uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />
```

Then sync and open in Android Studio:

```
npx cap sync android
npx cap open android
```

From Android Studio: connect your Android phone over USB (enable Developer Options + USB
debugging on the phone first), select it as the run target, and press Run. That's the real native
build — no simulated sensors, no scripted data, running on your actual device.

## iOS — what's ready, and the one thing that genuinely needs a Mac

`native/ios-plugin/NavdriftSensorsPlugin.swift` is fully written — real CoreLocation for GPS,
real CoreMotion for accelerometer/gyroscope/orientation/heading, real CMAltimeter for the
barometer (this is the part that doesn't exist on the web version at all, because WebKit blocks
the Barometer API on iOS entirely — a native build reads the same chip directly, no browser in
the way). None of that required a Mac to write.

Compiling, signing, and running an iOS app is the one place Apple's own tooling is unavoidable:
Xcode only runs on macOS, and there's no way around that from any cloud environment or from
Windows. When you do have access to a Mac (your own, a friend's, a college lab machine, or a
cloud Mac rental service):

```
npx cap add ios
```

then copy `native/ios-plugin/NavdriftSensorsPlugin.swift` into the generated `ios/App/App/`
folder via Xcode's "Add Files to App" (not a plain file copy — Xcode needs to register it in the
project), add the two Info.plist keys listed in a comment at the bottom of that Swift file, and
build to a real iPhone with your Apple ID (a free Apple ID is enough to run on your own device for
7 days at a time; a paid Developer account removes that limit).

Until that happens, the web version on an iPhone (Safari, added to the home screen as a PWA)
remains fully functional and is what a reviewer without your exact native build can test.

## What still uses the shared engine either way

Both native builds and the web version run through the exact same `frontend/mobile.html` —
`RoadGraph`, `ekfStep`, `BaroSensor`'s UI, the whole DriftFormer/EKF/dead-reckoning pipeline. The
plugin only supplies raw sensor readings through `SensorManager`; it does not duplicate or
replace any navigation logic. Testing on a real Android phone and testing in a browser on the
same phone should agree once you're feeding both the same real-world drive.
