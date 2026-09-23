@echo off
setlocal
cd /d "%~dp0"
echo Installing COLMAP plus pretrained-free true 3D Gaussian training.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 (
    echo Setup failed. Read the error above.
    pause
    exit /b 1
)
echo Setup complete. Open Start Studio.cmd.
pause
