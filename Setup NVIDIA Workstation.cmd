@echo off
setlocal
cd /d "%~dp0"
echo This installs CUDA PyTorch and the lightweight Apache-licensed model.
echo For optional research-only SHARP, follow README.md after this setup.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" -Device CUDA
if errorlevel 1 (
    echo Setup failed. Read the error above.
    pause
    exit /b 1
)
echo Setup complete. Open Start Studio.cmd to launch the app.
pause
