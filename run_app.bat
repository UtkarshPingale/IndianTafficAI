@echo off
title Indian Traffic AI Counter - Streamlit Dashboard (RTX 4070 SUPER CUDA)
cd /d "%~dp0"
echo ========================================================
echo  Starting Indian Traffic AI Streamlit Dashboard
echo ========================================================
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
    streamlit run app.py
) else (
    echo [!] venv not found. Running with default python...
    streamlit run app.py
)
pause
