@echo off
title Publish Indian Traffic AI Update
cd /d "%~dp0"
echo ========================================================
echo       INDIAN TRAFFIC AI - PUBLISH NEW UPDATE
echo ========================================================
set /p VER="Enter new version number (e.g. 1.0.1): "
if "%VER%"=="" (
    echo [!] Version number cannot be empty.
    pause
    exit /b
)
set /p NOTES="Enter release notes / changelog: "
python publish_update.py --version %VER% --notes "%NOTES%"
pause
