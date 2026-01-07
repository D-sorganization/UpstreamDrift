@echo off
echo Attempting to launch VcXsrv (X Server) with required settings...
echo Settings: -multiwindow -clipboard -wgl -ac (Disable Access Control)

if exist "C:\Program Files\VcXsrv\vcxsrv.exe" (
    start "" "C:\Program Files\VcXsrv\vcxsrv.exe" :0 -ac -multiwindow -clipboard -wgl
    echo Success: VcXsrv started.
) else if exist "C:\Program Files (x86)\VcXsrv\vcxsrv.exe" (
    start "" "C:\Program Files (x86)\VcXsrv\vcxsrv.exe" :0 -ac -multiwindow -clipboard -wgl
    echo Success: VcXsrv started.
) else (
    echo Error: Could not find VcXsrv installation.
    echo Please install it from https://sourceforge.net/projects/vcxsrv/
    echo or ensure it is in the default Program Files directory.
    pause
)
