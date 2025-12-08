import sys
print(f"Python Version: {sys.version}\n")

packages = {
    'tensorflow': 'tf',
    'numpy': 'np',
    'pandas': 'pd', 
    'cv2': 'cv2',
    'sklearn': None,
    'flask': None,
    'socketio': None,
    'PIL': 'Image'
}
# C:\Program Files\NVIDIA\CUDNN\v9.17\bin\13.1\cudnn64_9.dll
# C:\Program Files\NVIDIA\CUDNN\v9.17\include\13.1\cudnn.h
# C:\Program Files\NVIDIA\CUDNN\v9.17\lib\13.1\x64\cudnn.lib
installed = []
failed = []

for package, alias in packages.items():
    try:
        if package == 'cv2':
            import cv2
            installed.append(f"✓ OpenCV: {cv2.__version__}")
        elif package == 'tensorflow':
            import tensorflow as tf
            installed.append(f"✓ TensorFlow: {tf.__version__}")
            gpus = tf.config.list_physical_devices('GPU')
            installed.append(f"  └─ GPUs: {len(gpus)}")
        elif package == 'PIL':
            from PIL import Image
            installed.append(f"✓ Pillow: {Image.__version__}")
        elif alias:
            exec(f"import {package} as {alias}")
            exec(f"installed.append('✓ {package}: {{{alias}.__version__}}')")
        else:
            exec(f"import {package}")
            installed.append(f"✓ {package}")
    except ImportError:
        failed.append(f"✗ {package}")

print("Installed Packages:")
for item in installed:
    print(item)

if failed:
    print("\nMissing Packages:")
    for item in failed:
        print(item)
else:
    print("\n All core packages installed successfully!")