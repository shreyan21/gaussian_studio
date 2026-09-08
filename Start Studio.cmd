@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo First run: installing the CPU version and the lightweight model.
    echo On the NVIDIA workstation use Setup NVIDIA Workstation.cmd instead.
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" -Device CPU
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" run.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo The app could not start. Read the error above and README.md.
pause
exit /b 1
