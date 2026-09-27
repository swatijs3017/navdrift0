# Android build — follow exactly, in order

This is written so you don't have to make any decisions. Just do each step in order. If a step's
output doesn't match what's described, stop and paste me what you actually see.

Everything below runs on your own Windows machine, in the `D:\navdrift0-main` folder, using a
regular Command Prompt or PowerShell window. Anywhere you see a box like this:

```
some command
```

that is something you type and press Enter.

## Step 0 — one-time installs (skip anything you already have)

1. Install Node.js: go to https://nodejs.org, download the "LTS" version, run the installer,
   accept all defaults.
2. Install Android Studio: go to https://developer.android.com/studio, download it, run the
   installer, accept all defaults. The first time you open Android Studio it will download the
   Android SDK — let that finish (it can take a while, it's several GB).

## Step 1 — open a terminal in the project folder

Open File Explorer, go to `D:\navdrift0-main`, click on the address bar at the top, type `cmd`,
press Enter. A black terminal window opens already inside that folder.

## Step 2 — install project dependencies

```
npm install
```

Wait for it to finish. You'll see some yellow warning text scroll by — that's normal, ignore it,
as long as it doesn't end with the word "error".

## Step 3 — generate the Android project

```
npx cap add android
```

This creates a new `android` folder with a real Android Studio project inside it. It will ask
you to confirm overwriting the existing `android` folder — type `y` and press Enter (the file
that was there, `NavDriftService.kt`, is old leftover content that isn't used by anything).

## Step 4 — copy the real sensor code into place

```
mkdir android\app\src\main\java\com\navdrift\zero
copy native\android-plugin\NavdriftSensorsPlugin.kt android\app\src\main\java\com\navdrift\zero\NavdriftSensorsPlugin.kt
```

If it says the folder already exists, that's fine, ignore that and just run the `copy` line.

## Step 5 — tell the app about the sensor plugin

Open this file in Notepad (or any text editor):

```
android\app\src\main\java\com\navdrift\zero\MainActivity.java
```

It will look roughly like this:

```java
package com.navdrift.zero;

import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
}
```

Change it to look like this instead (add the two new lines exactly as shown):

```java
package com.navdrift.zero;

import com.getcapacitor.BridgeActivity;
import android.os.Bundle;

public class MainActivity extends BridgeActivity {
  @Override
  public void onCreate(Bundle savedInstanceState) {
    registerPlugin(NavdriftSensorsPlugin.class);
    super.onCreate(savedInstanceState);
  }
}
```

Save the file.

## Step 6 — add location permissions

Open this file in Notepad:

```
android\app\src\main\AndroidManifest.xml
```

Find the line that says `<manifest ...>` near the top. Right after that line, add these two new
lines:

```xml
    <uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
    <uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />
```

Save the file.

## Step 7 — sync everything

Back in your terminal window:

```
npx cap sync android
```

Wait for it to finish with no red "error" text.

## Step 8 — open the project in Android Studio

```
npx cap open android
```

Android Studio opens with the project loaded. The first time, it will show a progress bar at the
bottom ("Gradle sync") — this can take several minutes the first time. Just wait for it to finish.
Do not click anything while it's running.

## Step 9 — connect your phone

1. On your Android phone: go to Settings, search for "About phone", find "Build number", tap it
   7 times in a row. It will say "You are now a developer."
2. Go back to Settings, find the new "Developer options" entry, open it, turn on
   "USB debugging".
3. Plug your phone into your PC with a USB cable.
4. Your phone will show a popup asking "Allow USB debugging?" — tap Allow (and check "always
   allow from this computer" if it offers that).

## Step 10 — run the app on your phone

Back in Android Studio: at the top of the window there's a dropdown that should now show your
phone's name (it might say "no devices" until the phone is detected — wait a few seconds and it
should appear). Make sure that dropdown shows your phone, not an emulator.

Click the green triangle "Run" button (▶) near that dropdown, or press Shift+F10.

Android Studio will build the app (a progress bar at the bottom) and then install and open it on
your phone automatically. This can take a few minutes the first time.

## Step 11 — what you should see on the phone

The app opens full-screen, looking exactly like the web version. It will ask for location
permission — tap "Allow while using the app" (or "Only this time"). It should then behave exactly
like the browser version: map, GPS lock, live sensor readouts.

To check the native sensors are really being used (not the browser fallback), look at the
"Debug — Sensor Pipeline" section in the app's panel: the POS SOURCE and sensor rows should be
populating with real numbers as you move the phone, and the console log (visible if you connect
Chrome DevTools to the phone, see below) will show a line starting with `[NativeBridge] Native
platform detected`.

## If something goes wrong

- If Step 3 or Step 7 shows red "error" text: copy the exact error text and send it to me, don't
  try to fix it yourself.
- If the app installs but shows a blank white screen: this usually means a JavaScript error.
  Connect your phone's Chrome browser debugger: on your PC, open Chrome, go to
  `chrome://inspect`, your phone should appear there with the NAVDRIFT-0 app listed — click
  "inspect" and send me whatever shows up in red in that console.
- If "Allow USB debugging" never appears on the phone: unplug and replug the cable, and make sure
  you picked "File Transfer" mode (not "Charging only") when the phone asked how to use the USB
  connection.

## What NOT to do

Don't change any code in Android Studio yourself. Don't accept any Android Studio prompt that
offers to "upgrade" the Gradle version or plugin versions — click "Don't ask again" or dismiss it
if it appears, then let me know it showed up, since changing that from what Capacitor generated
can silently break other things. Just get it running as-is first.
