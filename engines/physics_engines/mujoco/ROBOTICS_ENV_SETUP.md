# Robotics Environment Setup Guide

## Overview

The `robotics_env` Docker image is a unified environment designed to run all robotics simulations including:
- **MuJoCo Golf Models** (humanoid golf swing simulations)
- **Drake Golf Models** (optimization-based golf analysis)
- **Pinocchio Golf Models** (analytical dynamics)

## Current Environment Status

### ✅ What's Already Installed

The existing `robotics_env` contains:
- **Python 3.10.12** in virtual environment at `/opt/robotics_env/`
- **MuJoCo 3.2.3** - Physics simulation engine
- **Drake 1.48.0** - Model-based design and verification
- **Pinocchio 3.8.0** (libpinocchio) - Fast and flexible implementation of rigid body dynamics algorithms
- **defusedxml 0.7.1** - ✅ **RECENTLY ADDED** - XML parsing security library

### 🔧 Recent Fixes Applied

1. **Added defusedxml dependency** - Fixes the `ModuleNotFoundError: No module named 'defusedxml'` issue
2. **Updated Python path** - GUI now uses `/opt/robotics_env/bin/python` instead of system python
3. **Environment variables** - PATH and VIRTUAL_ENV properly set for the robotics environment

## How to Use the Environment

### From MuJoCo GUI
1. Launch the GUI: `python MuJoCo_Golf_Swing_Model/docker/gui/deepmind_control_suite_MuJoCo_GUI.py`
2. Configure your simulation settings
3. Click "🚀 RUN SIMULATION" - it will automatically use the robotics_env
4. If you get dependency errors, click "🔧 UPDATE ENV" to add missing packages

### From Repository Launcher
1. Launch: `python Repository_Management/launchers/golf_launcher.py`
2. Select "MuJoCo Golf Model" 
3. Click "LAUNCH SIMULATION" - uses the same robotics_env

### Manual Docker Commands
```bash
# Test the environment
docker run --rm robotics_env python -c "import mujoco, drake, defusedxml; print('All libraries available')"

# Run MuJoCo simulation (from MuJoCo repo directory)
docker run --rm -v "$(pwd):/workspace" -w "/workspace/python" robotics_env python -m mujoco_golf_pendulum

# Interactive shell
docker run --rm -it robotics_env bash
```

## Troubleshooting

### If You Get "defusedxml not found" Error:
1. **Option 1**: Click "🔧 UPDATE ENV" button in the MuJoCo GUI
2. **Option 2**: Run `python add_defusedxml_to_robotics_env.py` from MuJoCo directory
3. **Option 3**: Manual update:
   ```bash
   docker run --rm robotics_env /opt/robotics_env/bin/pip install "defusedxml>=0.7.1"
   ```

### If Simulations Don't Start:
1. Check Docker is running: `docker --version`
2. Verify robotics_env exists: `docker images robotics_env`
3. Test environment: `python test_docker_venv.py`
4. Check logs in the GUI for specific error messages

### For X11/Display Issues:
- **Windows**: Make sure VcXsrv is running, or disable "Live Interactive View"
- **Linux**: Ensure X11 forwarding is enabled

## Environment Architecture

```
robotics_env Docker Image
├── Ubuntu 22.04 base
├── Python 3.10.12 virtual environment (/opt/robotics_env/)
├── MuJoCo 3.2.3 (physics simulation)
├── Drake 1.48.0 (optimization & control)
├── Pinocchio 3.8.0 (analytical dynamics)
├── defusedxml 0.7.1 (XML security)
└── Supporting libraries (numpy, scipy, matplotlib, etc.)
```

## Benefits of Unified Environment

1. **Consistency**: Same environment for all robotics models
2. **Efficiency**: No need to rebuild entire environments for small changes
3. **Compatibility**: All libraries tested to work together
4. **Portability**: Works on Windows, Linux, and macOS via Docker

## Adding New Dependencies

To add new packages to the robotics_env:

1. **Quick Method** (for single packages):
   ```bash
   # Create temporary Dockerfile
   echo "FROM robotics_env:latest
   RUN /opt/robotics_env/bin/pip install your-package-name" > Dockerfile
   
   # Build updated image
   docker build -t robotics_env .
   ```

2. **GUI Method**: 
   - Modify the `rebuild_docker()` method in the MuJoCo GUI
   - Add your package to the pip install command
   - Click "🔧 UPDATE ENV"

3. **Script Method**:
   - Use `add_defusedxml_to_robotics_env.py` as a template
   - Modify to install your required packages

## Files Related to Environment

- `MuJoCo_Golf_Swing_Model/docker/gui/deepmind_control_suite_MuJoCo_GUI.py` - Main GUI with Docker integration
- `MuJoCo_Golf_Swing_Model/add_defusedxml_to_robotics_env.py` - Dependency updater script
- `MuJoCo_Golf_Swing_Model/test_docker_venv.py` - Environment testing script
- `Repository_Management/launchers/golf_launcher.py` - Unified launcher for all models

## Next Steps

1. **Test the updated environment** with your golf simulations
2. **Add any missing dependencies** using the update methods above
3. **Consider adding other robotics libraries** you might need (e.g., ROS, Gazebo, etc.)
4. **Document any additional setup** for your specific use cases

The robotics_env is now ready to run all your golf simulation models with proper dependency management!