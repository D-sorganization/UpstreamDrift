# WSL2 Setup for Golf Modeling Suite

This guide explains how to set up WSL2 (Windows Subsystem for Linux) for running the Golf Modeling Suite with full robotics library support including Pinocchio, Drake, Crocoddyl, and Pink.

## Why WSL2?

Some robotics libraries are not available on Windows via pip:
- **Pinocchio** - Rigid body dynamics library
- **Crocoddyl** - Optimal control library
- **Pink** - Python inverse kinematics
- **Gepetto-viewer** - 3D visualization

WSL2 provides a native Linux environment where these packages install easily via conda-forge.

## Prerequisites

- Windows 10 version 2004+ or Windows 11
- WSL2 enabled with Ubuntu 22.04

### Install WSL2 (if not already installed)

```powershell
wsl --install -d Ubuntu-22.04
```

## Setup Instructions

### 1. Install Miniforge in WSL2

Open Ubuntu terminal and run:

```bash
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
rm Miniforge3-Linux-x86_64.sh

# Initialize conda
source ~/miniforge3/etc/profile.d/conda.sh
conda init bash
```

### 2. Create the Golf Suite Environment

```bash
source ~/miniforge3/etc/profile.d/conda.sh

# Create environment with all robotics packages
conda create -n golf_suite python=3.11 \
    pinocchio \
    crocoddyl \
    pink \
    meshcat-python \
    gepetto-viewer \
    gepetto-viewer-corba \
    numpy scipy matplotlib pandas \
    -c conda-forge -y

# Activate and install additional packages
conda activate golf_suite
pip install drake mujoco PyQt6 structlog colorama pyyaml opencv-python PyOpenGL
```

### 3. Fix Library Compatibility (if needed)

If you see errors about `libcoal.so.3.0.1`, create a symlink:

```bash
conda activate golf_suite
cd $CONDA_PREFIX/lib
ln -sf libcoal.so.3.0.2 libcoal.so.3.0.1
```

### 4. Verify Installation

```bash
conda activate golf_suite
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

python -c "
packages = ['pinocchio', 'mujoco', 'pydrake', 'pink', 'crocoddyl', 'meshcat']
for pkg in packages:
    try:
        m = __import__(pkg)
        print(f'{pkg}: OK')
    except ImportError as e:
        print(f'{pkg}: FAILED - {e}')
"
```

Expected output:
```
pinocchio: OK
mujoco: OK
pydrake: OK
pink: OK
crocoddyl: OK
meshcat: OK
```

## Running the Golf Modeling Suite

### Option 1: Use the GUI Launcher (Recommended)

1. Launch the Golf Modeling Suite on Windows:
   ```powershell
   py -3.12 src/launchers/golf_launcher.py
   ```

2. Check the **WSL** checkbox in the top toolbar

3. Click any model tile - it will automatically launch in WSL2

### Option 2: Use the WSL Launch Script

```bash
wsl -d Ubuntu-22.04 -- bash /mnt/c/Users/diete/Repositories/Golf_Modeling_Suite/run_wsl.sh
```

Or with specific engine:
```bash
# Pinocchio GUI
wsl -d Ubuntu-22.04 -- bash /mnt/c/Users/diete/Repositories/Golf_Modeling_Suite/run_wsl.sh --pinocchio

# Drake GUI
wsl -d Ubuntu-22.04 -- bash /mnt/c/Users/diete/Repositories/Golf_Modeling_Suite/run_wsl.sh --drake

# MuJoCo Humanoid
wsl -d Ubuntu-22.04 -- bash /mnt/c/Users/diete/Repositories/Golf_Modeling_Suite/run_wsl.sh --mujoco
```

### Option 3: Manual Launch

```bash
wsl -d Ubuntu-22.04

# Inside WSL
source ~/miniforge3/etc/profile.d/conda.sh
conda activate golf_suite
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/mnt/c/Users/diete/Repositories/Golf_Modeling_Suite"
cd /mnt/c/Users/diete/Repositories/Golf_Modeling_Suite

python src/launchers/golf_launcher.py
```

## Visualization

### Meshcat (Web-based)

When running Pinocchio simulations, meshcat starts a web server. Open in your browser:
```
http://127.0.0.1:7000/static/
```

### Gepetto-viewer (Native)

Gepetto-viewer requires X11. WSLg (Windows 11) handles this automatically. On Windows 10, you may need VcXsrv.

## Troubleshooting

### "libcoal.so.3.0.1: cannot open shared object file"

Create the compatibility symlink:
```bash
cd $CONDA_PREFIX/lib
ln -sf libcoal.so.3.0.2 libcoal.so.3.0.1
```

### GUI windows don't appear

Ensure WSLg is working:
```bash
# Test with a simple GUI
sudo apt install x11-apps
xclock
```

If xclock doesn't show, you may need to:
1. Update Windows to latest version
2. Run `wsl --update`
3. Restart WSL: `wsl --shutdown`

### Import errors after conda install

Always set `LD_LIBRARY_PATH` before running:
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

## Package Versions (Tested)

| Package | Version |
|---------|---------|
| Python | 3.11 |
| Pinocchio | 3.8.0+ |
| MuJoCo | 3.4.0 |
| Drake | 1.49.0 |
| Pink | 3.5.0 |
| Crocoddyl | 3.2.0 |
| Meshcat | Latest |

## See Also

- [Main README](../README.md)
- [Docker Setup](deployment/docker.md)
- [API Documentation](api/README.md)
