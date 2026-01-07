@echo off
cd /d "%~dp0"

echo ==========================================
echo      Robotics Environment Builder
echo ==========================================

docker build -t robotics_env .
if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ==========================================
echo      Starting Container
echo ==========================================
echo.
echo Mounting Repositories (Parent Dir) to /workspace
echo Exposing Meshcat/Viz ports: 7000-7010

docker run -it --rm ^
  --name robotics-dev ^
  -p 7000-7010:7000-7010 ^
  -p 8888:8888 ^
  -v "%~dp0..":/workspace ^
  -w /workspace ^
  robotics_env
