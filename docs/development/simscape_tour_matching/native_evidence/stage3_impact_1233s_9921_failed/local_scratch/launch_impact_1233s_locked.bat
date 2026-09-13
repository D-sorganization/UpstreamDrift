@echo off
setlocal

set "RUNTIME_DIR=C:\Users\diete\Repositories\Worktrees\UpstreamDrift-simscape-tour-runtime"
cd /d "%RUNTIME_DIR%"

if not exist scratch mkdir scratch

if exist "C:\Users\diete\SimscapeTour9921\python-r2025b\Scripts\python.exe" (
    set "PYTHON_EXE=C:\Users\diete\SimscapeTour9921\python-r2025b\Scripts\python.exe"
) else (
    set "PYTHON_EXE=C:\Users\diete\AppData\Local\Programs\Python\Python312\python.exe"
)

echo [%DATE% %TIME%] Starting run_impact_1233s_locked.py >> scratch\impact_1233s_locked.log
"%PYTHON_EXE%" scratch\run_impact_1233s_locked.py >> scratch\impact_1233s_locked.log 2>> scratch\impact_1233s_locked.err
echo [%DATE% %TIME%] Finished with errorlevel %ERRORLEVEL% >> scratch\impact_1233s_locked.log

exit /b %ERRORLEVEL%
