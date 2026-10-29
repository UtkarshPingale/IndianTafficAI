@echo off
title Indian Traffic AI - Vehicle Detection & Counting System
cd /d "%~dp0"
echo ========================================================
echo  Starting Indian Traffic AI Standalone Application (.exe)
echo ========================================================
if exist "dist\IndianTrafficAI\IndianTrafficAI.exe" (
    start "" "dist\IndianTrafficAI\IndianTrafficAI.exe"
) else (
    echo [!] IndianTrafficAI.exe not found in dist\IndianTrafficAI.
    pause
)
