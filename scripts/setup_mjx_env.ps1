# Set up isolated MJX virtual environment pinned from scripts/config/mjx_env_pins.json
# Usage: powershell -ExecutionPolicy Bypass -File scripts/setup_mjx_env.ps1

$ErrorActionPreference = "Stop"

$ConfigFile = Join-Path $PSScriptRoot "config\mjx_env_pins.json"
if (-not (Test-Path $ConfigFile)) {
    Write-Error "Configuration file not found: $ConfigFile"
    exit 1
}

$Config = Get-Content $ConfigFile -Raw | ConvertFrom-Json
$Packages = $Config.packages.PSObject.Properties

$VenvDir = Join-Path $HOME ".venv-mjx"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$PipExe = Join-Path $VenvDir "Scripts\pip.exe"

Write-Host "Setting up MJX environment in $VenvDir..."

if (-not (Test-Path $PythonExe)) {
    Write-Host "Creating virtual environment at $VenvDir..."
    python -m venv $VenvDir
}

$InstallArgs = @()
foreach ($pkg in $Packages) {
    $InstallArgs += "$($pkg.Name)==$($pkg.Value)"
}

Write-Host "Installing pinned packages: $($InstallArgs -join ' ')..."
& $PipExe install @InstallArgs

Write-Host "`nInstalled package verification:"
$VerifyScript = @"
import defusedxml, jax, mujoco, numpy, pytest, scipy
from mujoco import mjx
print(f'  jax:        {jax.__version__}')
print(f'  mujoco:     {mujoco.__version__}')
print(f'  mujoco-mjx: {mjx.__file__ is not None}')
print(f'  defusedxml: {defusedxml.__version__}')
print(f'  numpy:      {numpy.__version__}')
print(f'  scipy:      {scipy.__version__}')
print(f'  pytest:     {pytest.__version__}')
"@
& $PythonExe -c $VerifyScript

Write-Host "`nMJX environment setup complete at $VenvDir"
