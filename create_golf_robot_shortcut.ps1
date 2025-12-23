# Golf Modeling Suite Desktop Shortcut Creator
# Uses the new GolfingRobot icon

$WshShell = New-Object -comObject WScript.Shell
$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop "Golf Modeling Suite.lnk"
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)

# Derive paths dynamically based on script location
$repoRoot = $PSScriptRoot
$pythonBasePath = Join-Path (Join-Path $env:USERPROFILE "AppData\Local\Programs\Python") "Python313"
$pythonExePath = Join-Path $pythonBasePath "python.exe"
$launcherPath = Join-Path (Join-Path $repoRoot "launchers") "golf_launcher.py"

# Use the Windows-optimized GolfingRobot icon for maximum clarity on Windows
$iconCandidates = @(
    (Join-Path (Join-Path $repoRoot "launchers\assets") "golf_robot_windows_optimized.ico"),
    (Join-Path (Join-Path $repoRoot "launchers\assets") "golf_robot_ultra_sharp.ico"),
    (Join-Path (Join-Path $repoRoot "launchers\assets") "golf_robot_cropped_icon.ico"),
    (Join-Path (Join-Path $repoRoot "launchers\assets") "golf_robot_icon.ico"),
    (Join-Path (Join-Path $repoRoot "launchers\assets") "golf_icon.ico")
)

$iconPath = $null
foreach ($candidate in $iconCandidates) {
    if (Test-Path $candidate) {
        $iconPath = $candidate
        Write-Host "Using icon: $iconPath"
        break
    }
}

if (-not $iconPath) {
    Write-Warning "No icon file found, shortcut will use default icon"
    $iconPath = ""
}

# Configure shortcut properties
$Shortcut.TargetPath = $pythonExePath
$Shortcut.Arguments = "`"$launcherPath`""
$Shortcut.WorkingDirectory = $repoRoot
$Shortcut.Description = "Launch the Golf Modeling Suite with GolfingRobot"
# Set icon if available
if ($iconPath -and $iconPath -ne "") {
    $Shortcut.IconLocation = $iconPath
}

# Save the shortcut
$Shortcut.Save()

Write-Host "Golf Modeling Suite shortcut created successfully!"
Write-Host "Location: $ShortcutPath"
Write-Host "Icon: $iconPath"
Write-Host ""
Write-Host "The shortcut uses the new GolfingRobot icon and will launch the Golf Modeling Suite."