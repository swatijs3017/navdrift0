"""
overwrite_mobile.py  —  replaces frontend/mobile.html with the cloud-fixed version
Run from D:\\navdrift0-main:  python overwrite_mobile.py
"""
import os, sys, urllib.request

# The fixed file delivered to you — download it directly
URL = "https://claude.ai/api/artifacts/file/37a45b25-4ab7-45ff-9b47-8e4b1da4161b"

TARGET = os.path.join('frontend', 'mobile.html')
if not os.path.exists('frontend'):
    sys.exit("Run this from D:\\navdrift0-main (the repo root)")

# Back up the original
backup = TARGET + '.bak'
if os.path.exists(TARGET) and not os.path.exists(backup):
    import shutil
    shutil.copy2(TARGET, backup)
    print(f"Backed up original → {backup}")

print("Downloading fixed mobile.html from Claude…")
try:
    urllib.request.urlretrieve(URL, TARGET)
    print(f"✓ Written to {TARGET}")
except Exception as ex:
    print(f"Download failed: {ex}")
    print("Use the manual copy method below instead.")
    sys.exit(1)

print("\nNow run:")
print("  git add -f frontend/mobile.html")
print("  git commit -m 'fix: IMU gravity-free accel + bias calibration'")
print("  git push origin isro-grade")
