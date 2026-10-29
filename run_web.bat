@echo off
title Indian Traffic AI Counter - Web Server (RTX 4070 SUPER CUDA)
cd /d "%~dp0"
echo ========================================================
echo  Starting Indian Traffic AI Web Server on NVIDIA GPU
echo ========================================================
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
    python web_server.py
) else (
    echo [!] venv not found. Running with default python...
    python web_server.py
)
pause
