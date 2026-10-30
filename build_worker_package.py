"""
Packages the lightweight Client GPU Worker Node into a portable ZIP distribution (~20 MB).
This ZIP contains everything needed to run GPU acceleration on any client PC (e.g. RTX 5070 / RTX 4070)
without bundling 2.5 GB of PyTorch binaries.
"""

import os
import sys
import zipfile
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ZIP = os.path.join(BASE_DIR, "IndianTrafficAI_GPU_Worker.zip")

RUN_BAT_CONTENT = """@echo off
title Indian Traffic AI - Client GPU Worker Node
cd /d "%~dp0"
echo ========================================================
echo   Starting Indian Traffic AI - Client GPU Worker Node
echo   High-Performance Remote GPU Video Processing Engine
echo ========================================================
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python is not found in system PATH!
    echo     Please install Python 3.10 or 3.11 from python.org
    pause
    exit /b
)
python client_gpu_worker.py
pause
"""

SETUP_BAT_CONTENT = """@echo off
title Setup Indian Traffic AI Client GPU Worker
cd /d "%~dp0"
echo ========================================================
echo   Setting Up Dependencies for Client GPU Worker
echo ========================================================
echo [*] Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo [!] Python 3.10 or 3.11 is required. Please install Python first.
    pause
    exit /b
)

echo [*] Installing PyTorch with CUDA 12.8 (Full Blackwell / RTX 50-series + Ada Lovelace)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo [*] Installing Ultralytics YOLOv11 and supporting libraries...
pip install ultralytics requests opencv-python numpy openpyxl pandas

echo ========================================================
echo [OK] Client GPU Worker setup complete!
echo      Run 'run_worker.bat' to connect to your central server.
echo ========================================================
pause
"""

REQUIREMENTS_CONTENT = """torch>=2.7.0
torchvision>=0.22.0
ultralytics>=8.3.0
requests>=2.31.0
opencv-python>=4.9.0
numpy>=1.24.0
"""

README_CONTENT = """# Indian Traffic AI - Client GPU Worker Node

This lightweight client package (~20 MB) connects your local NVIDIA GPU (e.g., GeForce RTX 5070 / RTX 4070)
to the central Indian Traffic AI Web Server running on the host machine.

## Quick Start (Client PC)

1. **One-Time Setup**:
   Double-click `setup_worker.bat` to install PyTorch CUDA 12.8 and Ultralytics on your system.

2. **Run Worker**:
   Double-click `run_worker.bat`.
   Enter the IP address of the Central Web Server (e.g., `http://192.168.1.111:5000`).

3. **Ready!**:
   Your GPU will automatically appear in the Web Studio dashboard header on the central server.
   Whenever a video is submitted for processing in the browser, your GPU will process it at maximum FPS!
"""


def build_package(output_path=OUTPUT_ZIP):
    print(f"[*] Packaging Client GPU Worker into: {output_path}")
    
    files_to_pack = [
        ("client_gpu_worker.py", "client_gpu_worker.py"),
        ("vehicle_counter.py", "vehicle_counter.py"),
        ("vehicle_classes.py", "vehicle_classes.py"),
        ("line_counter.py", "line_counter.py"),
        ("models/yolo11s_indian_traffic.pt", "models/yolo11s_indian_traffic.pt"),
    ]

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for src_rel, dest_rel in files_to_pack:
            src_full = os.path.join(BASE_DIR, src_rel)
            if os.path.exists(src_full):
                zf.write(src_full, dest_rel)
                print(f" [+] Added: {dest_rel} ({round(os.path.getsize(src_full)/1024, 1)} KB)")
            else:
                print(f" [!] Warning: Missing source file: {src_rel}")

        # Add generated batch and readme files
        zf.writestr("run_worker.bat", RUN_BAT_CONTENT)
        zf.writestr("setup_worker.bat", SETUP_BAT_CONTENT)
        zf.writestr("requirements.txt", REQUIREMENTS_CONTENT)
        zf.writestr("README.md", README_CONTENT)
        print(" [+] Added: run_worker.bat, setup_worker.bat, requirements.txt, README.md")

    size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)
    print(f"[OK] Successfully built Client GPU Worker package: {output_path} ({size_mb} MB)!")
    return output_path


if __name__ == "__main__":
    build_package()
