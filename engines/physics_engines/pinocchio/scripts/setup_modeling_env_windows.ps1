# Windows Setup Script for Modeling Environment
# Run this in PowerShell

Write-Host "=== Modeling Stack Setup (Windows) ==="

# 1. Check for Conda
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "Conda not found. Please install Miniforge or Anaconda first."
    exit 1
}

# 2. Create/Update Conda Environment
$EnvName = "pinocchio-golf"
$EnvExists = conda env list | Select-String -Pattern "^$EnvName\s"

if (-not $EnvExists) {
    Write-Host ">>> Creating conda env '$EnvName'..."
    conda create -y -n $EnvName -c conda-forge python=3.10 pinocchio pink crocoddyl numpy scipy matplotlib pytest pytest-cov ruff black mypy pyyaml
} else {
    Write-Host ">>> Updating conda env '$EnvName'..."
    conda install -y -n $EnvName -c conda-forge pinocchio pink crocoddyl numpy scipy matplotlib pytest pytest-cov ruff black mypy pyyaml
}

# 3. Install Pip Dependencies
Write-Host ">>> Installing pip dependencies..."
# Activate env temporarily to run pip
# Note: 'conda activate' might not work in script without shell hook, using 'conda run'
conda run -n $EnvName pip install qpsolvers osqp scs quadprog drake

Write-Host "=== Setup complete. ==="
Write-Host "To activate the environment, run:"
Write-Host "    conda activate $EnvName"
