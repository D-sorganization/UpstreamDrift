@echo off
setlocal

set "RUNTIME_DIR=C:\Users\diete\Repositories\Worktrees\UpstreamDrift-simscape-tour-runtime"
cd /d "%RUNTIME_DIR%"

if not exist scratch mkdir scratch

if exist "C:\Users\diete\SimscapeTour9921\python-r2025b\Scripts\python.exe" (
    set "PYTHON_EXE=C:\Users\diete\SimscapeTour9921\python-r2025b\Scripts\python.exe"
) else if exist "C:\Users\diete\SimscapeTour9921\python-r2025b\python.exe" (
    set "PYTHON_EXE=C:\Users\diete\SimscapeTour9921\python-r2025b\python.exe"
) else (
    set "PYTHON_EXE=C:\Users\diete\AppData\Local\Programs\Python\Python312\python.exe"
)

echo [%DATE% %TIME%] Starting run_stage3_impact_1233s.py > scratch\stage3_impact_1233s.log
"%PYTHON_EXE%" scratch\run_stage3_impact_1233s.py --max-nfev 300 --diff-step 0.001 >> scratch\stage3_impact_1233s.log 2>&1
echo [%DATE% %TIME%] Finished with errorlevel %ERRORLEVEL% >> scratch\stage3_impact_1233s.log

exit /b %ERRORLEVEL%
