# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

block_cipher = None

BASE_DIR = os.path.abspath(SPECPATH)

# Collect Ultralytics data (trackers, configs, etc.)
ultralytics_datas = collect_data_files('ultralytics')

# Collect imageio_ffmpeg binaries
ffmpeg_datas = collect_data_files('imageio_ffmpeg')

# Collect project static and template assets
project_datas = [
    (os.path.join(BASE_DIR, 'templates'), 'templates'),
    (os.path.join(BASE_DIR, 'static'), 'static'),
    (os.path.join(BASE_DIR, 'models'), 'models'),
]

version_file = os.path.join(BASE_DIR, 'version.json')
if os.path.exists(version_file):
    project_datas.append((version_file, '.'))

sample_video = os.path.join(BASE_DIR, 'compressed_output_TrafficPolice.mp4')
if os.path.exists(sample_video):
    project_datas.append((sample_video, '.'))

all_datas = ultralytics_datas + ffmpeg_datas + project_datas

# Collect torch and torchvision dynamic libraries (including all CUDA 12.4 runtime DLLs)
torch_bins = collect_dynamic_libs('torch')
torchvision_bins = collect_dynamic_libs('torchvision')

# Exclude unnecessary heavy scientific, compiler & precompiled DLLs to stay within GitHub's 2.0 GB release limit
EXCLUDE_DLLS = {
    'cufft64_11.dll', 'cufftw64_11.dll', 'cusparse64_12.dll',
    'cusolver64_11.dll', 'cusolverMg64_11.dll', 'cudnn_adv64_9.dll',
    'curand64_10.dll', 'cudnn_engines_precompiled64_9.dll',
    'cudnn_engines_runtime_compiled64_9.dll', 'nvrtc64_120_0.dll',
    'nvJitLink_120_0.dll', 'cupti64_2024.1.0.dll', 'nvrtc-builtins64_124.dll'
}
all_bins = [b for b in (torch_bins + torchvision_bins) if not any(b[0].endswith(x) for x in EXCLUDE_DLLS)]

# Hidden imports
hidden_imports = [
    'ultralytics',
    'ultralytics.models',
    'ultralytics.models.yolo',
    'ultralytics.nn',
    'ultralytics.utils',
    'torch',
    'torchvision',
    'torchvision.ops',
    'cv2',
    'flask',
    'jinja2',
    'openpyxl',
    'openpyxl.styles',
    'openpyxl.chart',
    'et_xmlfile',
    'pandas',
    'numpy',
    'scipy',
    'imageio_ffmpeg',
    'tkinter',
    'tkinter.filedialog',
    'vehicle_classes',
    'line_counter',
    'excel_reporter',
    'vehicle_counter',
    'web_server',
    'auto_updater',
    'urllib.request',
]

a = Analysis(
    ['main_launcher.py'],
    pathex=[BASE_DIR],
    binaries=all_bins,
    datas=all_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'IPython', 'notebook', 'nbformat', 'pytest'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Single-file standalone executable (Everything bundled into IndianTrafficAI.exe)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='IndianTrafficAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
