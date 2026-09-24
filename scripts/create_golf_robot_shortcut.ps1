# UpstreamDrift Desktop and Start Menu Shortcut Creator
# Delegates to canonical Python shortcut manager (src.launchers.desktop_shortcuts)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent

$pythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $pythonExe) {
    $pythonExe = (Get-Command python3 -ErrorAction SilentlyContinue).Source
}
if (-not $pythonExe) {
    $pythonBasePath = Join-Path (Join-Path $env:USERPROFILE "AppData\Local\Programs\Python") "Python313"
    $pythonExe = Join-Path $pythonBasePath "python.exe"
}

if ($pythonExe -and (Test-Path $pythonExe)) {
    Push-Location $repoRoot
    try {
        & $pythonExe -m src.launchers.desktop_shortcuts
    } finally {
        Pop-Location
    }
} else {
    Write-Error "Python executable not found; could not install UpstreamDrift shortcuts."
    exit 1
}