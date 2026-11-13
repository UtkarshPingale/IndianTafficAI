@echo off
title Indian Traffic AI - Vehicle Detection & Counting System
cd /d "%~dp0"
echo ======================================================================
echo    Starting Indian Traffic AI - Real-Time Vehicle Analytics
echo    Powered by YOLOv11s + PyTorch CUDA Hardware Engine
echo ======================================================================
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" main.py
) else (
    python main.py
)
pause
