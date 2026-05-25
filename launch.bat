@echo off
REM Quick and obvious launcher script for the main PyQt6 Application
cd /d %~dp0
set PYTHONPATH=%~dp0
echo Launching UpstreamDrift Golf Launcher...
python -m src.launchers.golf_launcher
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
