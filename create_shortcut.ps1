$WshShell = New-Object -comObject WScript.Shell
$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop "Golf Modeling Suite.lnk"
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$userProfile = $env:USERPROFILE
$pythonBasePath = Join-Path (Join-Path $userProfile "AppData\Local\Programs\Python") "Python313"
$pythonExePath = Join-Path $pythonBasePath "python.exe"

# Assuming the script is run from the repo root or we derive it relative to user profile
# The review suggested $repoRoot = Join-Path (Join-Path $userProfile "Repositories") "Golf_Modeling_Suite"
# Let's use that as it matches the user's structure.
$repoRoot = Join-Path (Join-Path $userProfile "Repositories") "Golf_Modeling_Suite"
$launcherPath = Join-Path (Join-Path $repoRoot "launchers") "golf_launcher.py"
$iconPath = Join-Path (Join-Path $repoRoot "launchers\assets") "golf_icon.ico"

$Shortcut.TargetPath = $pythonExePath
$Shortcut.Arguments = $launcherPath
$Shortcut.WorkingDirectory = $repoRoot
$Shortcut.Description = "Launch the Golf Modeling Suite"
$Shortcut.IconLocation = $iconPath
$Shortcut.Save()
Write-Host "Shortcut created at $ShortcutPath"
