@echo off
title Indian Traffic AI Counter - Interactive Desktop Window
cd /d "%~dp0"
echo ========================================================
echo  Starting Indian Traffic AI Interactive OpenCV Window
echo ========================================================
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" main.py --window
) else (
    python main.py --window
)
pause
