[app]

# (str) Title of your application
title = SeeForMe Assistant

# (str) Package name
package.name = seeformeassistant

# (str) Package domain (needed for android/ios packaging)
package.domain = org.seeformeapp

# (str) Source code where the main.py live
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,kv,atlas,txt,onnx,pt,json

# (str) Application versioning (method 1)
version = 1.0

# (list) Application requirements
# comma separated e.g. requirements = sqlite3,kivy
requirements = python3,kivy,flask,flask-socketio,opencv-python,numpy,onnxruntime,vosk,pyttsx3,gtts,pygame,requests

# (str) Supported platform
# platform = ios
platform = android

# (str) Presplash of the application
#presplash.filename = %(source.dir)s/data/presplash.png

# (str) Icon of the application
#icon.filename = %(source.dir)s/data/icon.png

# (str) Supported orientation (landscape, portrait or all)
orientation = portrait

# (bool) Indicate if the application should be fullscreen or not
fullscreen = 0

[buildozer]

# (int) Log level (0 = error only, 1 = info, 2 = debug (with command output))
log_level = 2

# (int) Display warning if buildozer is run as root (0 = False, 1 = True)
warn_on_root = 1

[android]

# (str) Android entry point, default is ok for Kivy-based app
#android.entrypoint = org.kivy.android.PythonActivity

# (str) Full name including package path of the Java class that implements Python Service
#android.service_main_class = org.kivy.android.PythonService

# (str) Android app theme, default is ok for Kivy-based app
#android.theme = @android:style/Theme.NoTitleBar

# (list) Pattern to whitelist for the whole project
#android.whitelist =

# (bool) Enable AndroidX support. Enable when 'android.gradle_dependencies'
# contains an 'androidx' package, or any package from Kotlin source.
android.enable_androidx = True

# (str) The Android arch to build for, choices: armeabi-v7a, arm64-v8a, x86, x86_64
android.archs = arm64-v8a, armeabi-v7a

# (bool) enables Android auto backup feature (Android API >=23)
android.allow_backup = True

# (int) Target Android API, should be as high as possible.
android.api = 33

# (int) Minimum API your APK / AAB will support.
android.minapi = 21

# (str) Android NDK version to use
android.ndk = 25b

# (bool) Use --private data storage (True) or --dir public storage (False)
android.private_storage = True

# (str) Android additional libraries to copy into libs/armeabi
#android.add_src =

# (list) Gradle dependencies for android build
android.gradle_dependencies = androidx.appcompat:appcompat:1.2.0

# (list) Java classes to add as activities to the manifest.
#android.add_activities =

# (str) OUYA Console category. Should be one of GAME or APP
# If you leave this blank, OUYA support will not be enabled
#android.ouya.category = GAME

# (str) Filename of OUYA Console icon. It must be a 732x412 png image.
#android.ouya.icon.filename = %(source.dir)s/data/ouya_icon.png

# (list) permissions for android
android.permissions = INTERNET,RECORD_AUDIO,CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE

# (str) XML file to include as an intent filter in the MainActivity (optional)
#android.manifest_intent_filters =

# (list) Copy these files to src/main/res/xml/ (used for example with intent-filters)
#android.res_xml =

# (str) bootstrap to use for android builds (default: sdl2)
#android.bootstrap = sdl2

# (str) ANT command (default is "ant")
#android.ant_cmd = ant

# (str) If you want to use a different python for the android environment than the one you're running this with, set this
#android.python_cmd = python3