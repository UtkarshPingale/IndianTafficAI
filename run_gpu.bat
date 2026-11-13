@echo off
title Indian Traffic AI - NVIDIA RTX 4070 SUPER CUDA Launcher
cd /d "%~dp0"
echo ======================================================================
echo    Starting Indian Traffic AI with Hardware CUDA Acceleration
echo    NVIDIA GeForce RTX 4070 SUPER (CUDA 12.8 + FP16 Tensor Cores)
echo ======================================================================
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" main.py
) else (
    python main.py
)
pause
