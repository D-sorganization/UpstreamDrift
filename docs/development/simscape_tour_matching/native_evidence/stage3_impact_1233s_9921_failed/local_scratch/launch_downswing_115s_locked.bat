@echo off
setlocal

set "RUNTIME_DIR=C:\Users\diete\Repositories\Worktrees\UpstreamDrift-simscape-tour-runtime"
cd /d "%RUNTIME_DIR%"

if not exist scratch mkdir scratch

set "PYTHON_EXE=C:\Users\diete\SimscapeTour9921\python-r2025b\Scripts\python.exe"

echo [%DATE% %TIME%] Starting run_downswing_115s_locked.py >> scratch\downswing_115s_locked.log
"%PYTHON_EXE%" scratch\run_downswing_115s_locked.py >> scratch\downswing_115s_locked.log 2>> scratch\downswing_115s_locked.err
echo [%DATE% %TIME%] Finished with errorlevel %ERRORLEVEL% >> scratch\downswing_115s_locked.log

exit /b %ERRORLEVEL%

