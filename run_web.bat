@echo off
title Indian Traffic AI Counter - Web Studio (RTX 4070 SUPER CUDA)
cd /d "%~dp0"
echo ========================================================
echo  Starting Indian Traffic AI Web Studio on NVIDIA GPU
echo ========================================================
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" main.py --web
) else (
    python main.py --web
)
pause
