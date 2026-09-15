[app]
title = Arknights Zero
project_dir = .
input_file = scripts/run_desktop.py
exec_directory = /Users/yihao/Desktop/arkzero/神经网络/dist/prts
project_file = 
icon = /Users/yihao/Desktop/arkzero/神经网络/.venv/lib/python3.12/site-packages/PySide6/scripts/deploy_lib/pyside_icon.icns

[python]
python_path = /Users/yihao/Desktop/arkzero/神经网络/.venv/bin/python
packages = Nuitka==4.2.1
android_packages = buildozer==1.5.0,cython==0.29.33

[qt]
qml_files = 
excluded_qml_plugins = 
modules = Charts,Core,DBus,Gui,OpenGL,OpenGLWidgets,Test,Widgets
plugins = accessiblebridge,egldeviceintegrations,generic,iconengines,imageformats,platforminputcontexts,platforms,platforms/darwin,platformthemes,styles,wayland-decoration-client,wayland-graphics-integration-client,wayland-shell-integration,xcbglintegrations

[android]
wheel_pyside = 
wheel_shiboken = 
plugins = 

[nuitka]
macos.permissions = 
mode = standalone
extra_args = --noinclude-qt-translations --include-package=desktop --include-package=arknights_sim --include-package=agents --include-package=network --include-package=mcts --include-package=training --include-data-dir=data/real=data/real --include-data-files=configs/desktop_default.yaml=configs/desktop_default.yaml --include-data-files=configs/alphazero_001.yaml=configs/alphazero_001.yaml --include-data-dir=desktop/resources=desktop/resources --include-data-files=checkpoints/best.pt=bundled_checkpoints/best.pt --jobs=4 --macos-app-name="Arknights Zero" --macos-app-version=0.3.0

[buildozer]
mode = debug
recipe_dir = 
jars_dir = 
ndk_path = 
sdk_path = 
local_libs = 
arch = 

