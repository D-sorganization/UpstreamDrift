@echo off
title Golf Simulation Launcher
echo ========================================================
echo Launching Humanoid Golf Simulation GUI...
echo ========================================================

cd /d "%~dp0"

REM Check if python is available
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not found in your PATH.
    echo Please install Python (ensure "Add to PATH" is checked^).
    pause
    exit /b 1
)

REM Run the GUI Wrapper
python gui/deepmind_control_suite_MuJoCo_GUI.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo The GUI exited with an error. 
    echo If 'tkinter' is missing, install Python with tcl/tk support.
    pause
)
