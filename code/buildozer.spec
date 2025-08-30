[app]
title = AgeGenderExpressionApp
package.name = agegenderexpressionapp
package.domain = com.yourdomain
source.dir = .
source.include_exts = py,png,jpg,tflite
android.add_assets = models/
source.include_dirs = libs
version = 0.1

requirements = kivy==2.3.0,numpy==1.22.3,pillow==8.4.0,tflite-runtime==2.8.0


orientation = portrait
android.api = 30
android.minapi = 21
p4a.branch = master

android.permissions = CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE
android.opengl_es2 = True
android.archs = arm64-v8a
android.accept_sdk_license = True

android.debug = True
android.release = False
android.incremental = True

[buildozer]
log_level = 2
warn_on_root = 1