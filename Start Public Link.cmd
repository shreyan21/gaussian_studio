@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run Setup NVIDIA Workstation.cmd first.
    pause
    exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_public_link.ps1"
if errorlevel 1 (
    echo.
    echo Public link failed. Read error above.
    pause
    exit /b 1
)
