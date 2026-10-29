@echo off
title Package Indian Traffic AI to Portable ZIP
cd /d "%~dp0"
echo ========================================================
echo  Compressing dist\IndianTrafficAI to ZIP archive...
echo ========================================================
powershell -Command "Compress-Archive -Path 'dist\IndianTrafficAI\*' -DestinationPath 'IndianTrafficAI_Portable_Windows.zip' -Force"
if exist "IndianTrafficAI_Portable_Windows.zip" (
    echo [✓] Successfully created IndianTrafficAI_Portable_Windows.zip!
    echo [✓] You can now copy or share this ZIP file to any Windows PC.
) else (
    echo [!] Failed to create ZIP file.
)
pause
