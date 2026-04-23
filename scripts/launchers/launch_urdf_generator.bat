@echo off
REM Launch the Interactive URDF Generator using Python 3.12
REM This ensures compatibility with MuJoCo and other engines

echo Launching URDF Generator with Python 3.12...
py -3.12 src/tools/urdf_generator/launch_urdf_generator.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Failed to launch with Python 3.12.
    echo Please ensure Python 3.12 is installed: py -3.12 --version
    pause
    exit /b %ERRORLEVEL%
)
