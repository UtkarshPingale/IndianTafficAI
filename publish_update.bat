@echo off
setlocal
title Publish Indian Traffic AI Update
cd /d "%~dp0"

echo ======================================================================
echo           INDIAN TRAFFIC AI - UPDATE PACKAGING & PUBLISHING
echo ======================================================================
echo.

if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

set /p REBUILD="Did you modify Python code and need to rebuild the .exe? (y/n) [default: y]: "
if /i not "%REBUILD%"=="n" (
    echo.
    echo [*] Rebuilding standalone executable with latest changes...
    call build_exe.bat
    if errorlevel 1 (
        echo [ERROR] Build failed! Aborting update.
        pause
        exit /b 1
    )
)

echo.
set /p VER="Enter new version number (e.g. 1.0.1): "
if "%VER%"=="" (
    echo [!] Version number cannot be empty.
    pause
    exit /b 1
)

set /p NOTES="Enter release notes / changelog: "
if "%NOTES%"=="" (
    set NOTES=Performance improvements, vehicle counting enhancements, and bug fixes.
)

python publish_update.py --version %VER% --notes "%NOTES%"

echo.
set /p PUSH_GIT="Automatically commit and push version.json to GitHub? (y/n) [default: y]: "
if /i not "%PUSH_GIT%"=="n" (
    echo [*] Staging version.json and pushing to GitHub...
    git add version.json
    git commit -m "Release v%VER%: %NOTES%"
    git push origin main
    echo [OK] Pushed v%VER% manifest to GitHub!
)

echo.
echo ======================================================================
echo                           FINAL STEP
echo ======================================================================
echo Upload 'IndianTrafficAI_Update.zip' to your GitHub Release for tag 'v%VER%':
echo https://github.com/UtkarshPingale/IndianTafficAI/releases/new?tag=v%VER%
echo.
echo Once uploaded, ALL client PCs running IndianTrafficAI.exe will
echo automatically detect the update, download it, and restart!
echo ======================================================================
pause
