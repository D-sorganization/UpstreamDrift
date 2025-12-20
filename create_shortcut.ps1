$WshShell = New-Object -comObject WScript.Shell
$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop "Golf Modeling Suite.lnk"
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
# Derive paths dynamically based on script location
$repoRoot = $PSScriptRoot
$pythonBasePath = Join-Path (Join-Path $env:USERPROFILE "AppData\Local\Programs\Python") "Python313"
$pythonExePath = Join-Path $pythonBasePath "python.exe"
$launcherPath = Join-Path (Join-Path $repoRoot "launchers") "golf_launcher.py"
$iconPath = Join-Path (Join-Path $repoRoot "launchers\assets") "golf_icon.ico"

$Shortcut.TargetPath = $pythonExePath
$Shortcut.Arguments = $launcherPath
$Shortcut.WorkingDirectory = $repoRoot
$Shortcut.Description = "Launch the Golf Modeling Suite"
$Shortcut.IconLocation = $iconPath
$Shortcut.Save()
Write-Host "Shortcut created at $ShortcutPath"
