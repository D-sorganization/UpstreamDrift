@echo off
REM Launcher script for MuJoCo Golf Swing Model GUI (Windows)
REM This script activates the conda environment and runs the GUI application

echo ========================================
echo MuJoCo Golf Swing Model - GUI Launcher
echo ========================================
echo.

REM Check if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda not found in PATH
    echo Please install Miniconda/Anaconda or add conda to your PATH
    echo.
    echo Alternatively, activate your environment manually and run:
    echo   python -m python.mujoco_humanoid_golf
    pause
    exit /b 1
)

REM Try to activate conda environment
echo Activating conda environment 'sim-env'...
call conda activate sim-env
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Could not activate conda environment 'sim-env'
    echo Attempting to run with current Python environment...
    echo.
)

REM Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found
    echo Please ensure Python is installed and in your PATH
    pause
    exit /b 1
)

echo.
echo Starting MuJoCo Golf Swing Model GUI...
echo.

REM Run the application
python -m python.mujoco_humanoid_golf

REM If the application exits, pause to see any error messages
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code: %ERRORLEVEL%
    echo.
    pause
)

