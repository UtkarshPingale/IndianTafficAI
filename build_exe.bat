@echo off
setlocal
cd /d "%~dp0"
echo ======================================================================
echo           Building Standalone IndianTrafficAI.exe with PyInstaller
echo ======================================================================
echo.

if exist "venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo [*] Cleaning previous build cache...
if exist "build" rmdir /s /q "build"

echo [*] Running PyInstaller with IndianTrafficAI.spec...
pyinstaller --clean --noconfirm IndianTrafficAI.spec

if errorlevel 1 (
    echo.
    echo [ERROR] PyInstaller build failed!
    exit /b 1
)

echo.
echo ======================================================================
echo [OK] Build completed successfully!
echo Single Standalone Executable: dist\IndianTrafficAI.exe
echo ======================================================================
