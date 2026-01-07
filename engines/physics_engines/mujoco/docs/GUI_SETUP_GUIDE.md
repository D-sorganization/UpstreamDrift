# GUI Setup and Usage Guide

Complete guide for setting up and using the MuJoCo Golf Swing Model GUI application.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Software Dependencies](#software-dependencies)
3. [Installation](#installation)
4. [Running the Application](#running-the-application)
5. [Using the GUI](#using-the-gui)
6. [Troubleshooting](#troubleshooting)

## System Requirements

### Operating System
- **Windows**: Windows 10 or later (64-bit)
- **macOS**: macOS 10.15 (Catalina) or later
- **Linux**: Ubuntu 20.04+, Debian 11+, or equivalent

### Hardware
- **CPU**: Modern multi-core processor (Intel/AMD x86_64 or Apple Silicon)
- **RAM**: Minimum 4 GB, recommended 8 GB or more
- **Graphics**: Any graphics card with OpenGL support (integrated graphics sufficient)
- **Display**: 1024x768 minimum resolution

### Python Version
- **Python 3.10 or later** (Python 3.13 recommended)
- Python 3.13.5 is specified in the conda environment

## Software Dependencies

### Core Dependencies

1. **Python 3.10+**
   - Download from [python.org](https://www.python.org/downloads/)
   - Or use conda/miniconda (recommended)

2. **MuJoCo Physics Engine**
   - Python package: `mujoco>=3.0.0`
   - Automatically installed via pip/conda
   - No separate MuJoCo installation needed (uses Python bindings)

3. **PyQt6 GUI Framework**
   - Python package: `PyQt6>=6.6.0`
   - Provides the graphical user interface
   - Cross-platform windowing system

4. **NumPy**
   - Python package: `numpy>=2.1.0`
   - Required for numerical operations and array handling

### Optional Dependencies

- **Matplotlib**: For plotting and visualization (if needed)
- **SciPy**: For advanced scientific computing (if needed)
- **Pandas**: For data analysis (if needed)

### System-Level Dependencies

#### Windows
- No additional system packages required
- PyQt6 includes all necessary Qt libraries

#### macOS
- Xcode Command Line Tools (usually installed automatically)
- If missing, install via: `xcode-select --install`

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxkbcommon-x11-0 \
    libxcb-xinerama0
```

#### Linux (Fedora/RHEL)
```bash
sudo dnf install -y \
    python3-devel \
    python3-pip \
    mesa-libGL \
    glib2 \
    libxkbcommon-x11 \
    libxcb
```

## Installation

### Method 1: Conda (Recommended)

Conda provides the most reliable installation with all dependencies properly managed.

#### Step 1: Install Miniconda or Anaconda

- **Miniconda** (minimal): [Download](https://docs.conda.io/en/latest/miniconda.html)
- **Anaconda** (full distribution): [Download](https://www.anaconda.com/download)

#### Step 2: Create and Activate Environment

```bash
# Navigate to the repository root
cd MuJoCo_Golf_Swing_Model

# Create conda environment from file
conda env create -f python/environment.yml

# Activate the environment
conda activate sim-env
```

#### Step 3: Verify Installation

```bash
# Check Python version
python --version  # Should show Python 3.13.5

# Check installed packages
conda list | grep -E "mujoco|PyQt6|numpy"

# Test imports
python -c "import mujoco; import PyQt6; import numpy; print('All imports successful!')"
```

### Method 2: Python venv with pip

For users who prefer standard Python virtual environments.

#### Step 1: Create Virtual Environment

```bash
# Navigate to the repository root
cd MuJoCo_Golf_Swing_Model

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### Step 2: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all dependencies
pip install -r python/requirements.txt
```

#### Step 3: Verify Installation

```bash
# Test imports
python -c "import mujoco; import PyQt6; import numpy; print('All imports successful!')"
```

### Method 3: System-Wide Installation (Not Recommended)

Only use if you understand the implications of installing packages system-wide.

```bash
pip install --user -r python/requirements.txt
```

## Running the Application

### Quick Start

From the repository root directory:

```bash
# Activate your environment first (if using conda/venv)
conda activate sim-env  # or: source venv/bin/activate

# Run the application
python -m python.mujoco_golf_pendulum
```

### Alternative Methods

#### Method 1: From Python Directory

```bash
cd python
python -m mujoco_golf_pendulum
```

#### Method 2: Direct Python Execution

```bash
cd python/mujoco_golf_pendulum
python __main__.py
```

#### Method 3: Create a Launcher Script

**Windows** (`run_gui.bat`):
```batch
@echo off
call conda activate sim-env
python -m python.mujoco_golf_pendulum
pause
```

**macOS/Linux** (`run_gui.sh`):
```bash
#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh  # Adjust path as needed
conda activate sim-env
python -m python.mujoco_golf_pendulum
```

Make executable:
```bash
chmod +x run_gui.sh
```

### Expected Behavior

When launched successfully, you should see:

1. A window titled "MuJoCo Golf Pendulum (Double & Triple)"
2. Left side: 3D visualization of the golf swing pendulum
3. Right side: Control panel with:
   - Model selector dropdown
   - Play/Pause and Reset buttons
   - Torque sliders for each joint

## Using the GUI

### Interface Overview

```
┌─────────────────────────────────────────────────────────┐
│  MuJoCo Golf Pendulum (Double & Triple)                 │
├──────────────────────┬──────────────────────────────────┤
│                      │  Pendulum Model                  │
│                      │  ┌────────────────────────────┐  │
│                      │  │ Double pendulum (shoulder  │  │
│                      │  │ + wrist)              ▼    │  │
│                      │  └────────────────────────────┘  │
│                      │                                   │
│   3D Visualization   │  Simulation Control              │
│                      │  [Pause]  [Reset]                │
│                      │                                   │
│                      │  Joint Torques (Nm)              │
│                      │  Shoulder: [━━━━━━━━━━] 0        │
│                      │  Elbow:    [━━━━━━━━━━] 0        │
│                      │  Wrist:    [━━━━━━━━━━] 0        │
│                      │                                   │
└──────────────────────┴──────────────────────────────────┘
```

### Controls

#### 1. Model Selection

**Dropdown Menu**: "Pendulum Model"
- **Double Pendulum**: Simplified model with shoulder and wrist joints
  - Faster simulation
  - Good for basic swing analysis
  - Elbow slider is disabled
- **Triple Pendulum**: Full model with shoulder, elbow, and wrist joints
  - More realistic biomechanics
  - All three torque sliders active

**How to Use**:
- Click the dropdown to select a model
- The simulation automatically reloads with the new model
- The pendulum resets to initial backswing position

#### 2. Simulation Control

**Play/Pause Button**:
- **Pause** (default): Simulation is running
  - Click to pause the simulation
  - Button text changes to "Play"
- **Play**: Simulation is paused
  - Click to resume the simulation
  - Button text changes to "Pause"

**Reset Button**:
- Resets the pendulum to initial backswing position
- Sets all velocities to zero
- Does not change current torque settings
- Useful for:
  - Starting a new swing analysis
  - Recovering from unstable states
  - Testing different torque combinations

#### 3. Joint Torque Controls

**Torque Sliders**:
- **Range**: -100 Nm to +100 Nm
- **Resolution**: 1 Nm per step
- **Real-time**: Changes apply immediately while simulation runs

**Shoulder Torque**:
- Controls rotation at the shoulder joint
- Positive values: Forward rotation (downswing)
- Negative values: Backward rotation (backswing)
- Always active in both models

**Elbow Torque**:
- Controls rotation at the elbow joint
- Only active in Triple Pendulum model
- Disabled (grayed out) in Double Pendulum model
- Positive values: Extension
- Negative values: Flexion

**Wrist Torque**:
- Controls rotation at the wrist joint
- Always active in both models
- Positive values: Forward rotation
- Negative values: Backward rotation

**Torque Value Display**:
- Shows current torque value next to each slider
- Updates in real-time as you move the slider
- Format: Integer value in Nm (Newton-meters)

### Typical Workflow

1. **Start the Application**
   ```bash
   python -m python.mujoco_golf_pendulum
   ```

2. **Select a Model**
   - Choose Double or Triple Pendulum from dropdown
   - Wait for model to load (instant)

3. **Set Initial Conditions**
   - Use Reset button to ensure clean starting state
   - Pendulum starts at top of backswing position

4. **Apply Torques**
   - Adjust sliders to apply desired joint torques
   - Start with small values (10-20 Nm) to observe effects
   - Use Pause to freeze simulation while adjusting

5. **Observe Motion**
   - Click Play to start/resume simulation
   - Watch the 3D visualization
   - Adjust torques in real-time to control the swing

6. **Analyze Results**
   - Pause at interesting positions
   - Reset and try different torque combinations
   - Switch between models to compare behaviors

### Tips for Effective Use

1. **Start Small**: Begin with low torque values (10-30 Nm) to understand the system dynamics

2. **Use Pause**: Pause the simulation when making multiple slider adjustments to avoid confusion

3. **Reset Frequently**: Reset between experiments to ensure consistent starting conditions

4. **Model Comparison**: Switch between Double and Triple models to understand the effect of the elbow joint

5. **Realistic Torques**: Golf swing torques are typically:
   - Shoulder: 50-150 Nm during downswing
   - Elbow: 20-80 Nm
   - Wrist: 10-50 Nm

6. **Performance**: If simulation is slow:
   - Close other applications
   - Reduce window size
   - Use Double Pendulum model (faster)

## Troubleshooting

### Common Issues

#### Issue: "ModuleNotFoundError: No module named 'mujoco'"

**Solution**:
```bash
# Verify environment is activated
conda activate sim-env  # or: source venv/bin/activate

# Reinstall mujoco
pip install --upgrade mujoco
```

#### Issue: "ModuleNotFoundError: No module named 'PyQt6'"

**Solution**:
```bash
# Install PyQt6
pip install PyQt6>=6.6.0

# On Linux, you may also need system packages (see System Dependencies above)
```

#### Issue: Application Window Doesn't Appear

**Possible Causes**:
1. Window opened off-screen
   - Check taskbar/dock for the application
   - Try Alt+Tab (Windows/Linux) or Cmd+Tab (macOS)

2. Display server issues (Linux)
   ```bash
   # Check if X11/Wayland is running
   echo $DISPLAY

   # If empty, you may need to run via SSH with X11 forwarding
   ssh -X user@host
   ```

3. Graphics driver issues
   - Update graphics drivers
   - Try software rendering: `QT_QUICK_BACKEND=software python -m python.mujoco_golf_pendulum`

#### Issue: "Segmentation Fault" or Application Crashes

**Possible Causes**:
1. Incompatible Qt version
   ```bash
   # Reinstall PyQt6
   pip uninstall PyQt6
   pip install PyQt6>=6.6.0
   ```

2. Corrupted MuJoCo installation
   ```bash
   pip uninstall mujoco
   pip install --upgrade mujoco
   ```

3. Memory issues
   - Close other applications
   - Restart your computer

#### Issue: Simulation Runs Very Slowly

**Solutions**:
1. Use Double Pendulum model (simpler, faster)
2. Reduce window size
3. Close other applications
4. Check CPU usage (should be low for this simulation)
5. Update graphics drivers

#### Issue: "Permission Denied" on Linux/macOS

**Solution**:
```bash
# Make scripts executable
chmod +x scripts/*.sh

# If using conda, ensure conda is in PATH
export PATH="$HOME/miniconda3/bin:$PATH"
```

#### Issue: Conda Environment Not Found

**Solution**:
```bash
# List all environments
conda env list

# If sim-env doesn't exist, recreate it
conda env create -f python/environment.yml

# Activate it
conda activate sim-env
```

### Getting Help

If you encounter issues not covered here:

1. **Check the Logs**: Look for error messages in the terminal
2. **Verify Installation**: Run the verification commands in the Installation section
3. **Check Dependencies**: Ensure all system-level dependencies are installed
4. **Update Packages**: Try updating all packages:
   ```bash
   pip install --upgrade mujoco PyQt6 numpy
   ```
5. **Create an Issue**: Report the problem on GitHub with:
   - Operating system and version
   - Python version (`python --version`)
   - Error message or screenshot
   - Steps to reproduce

### Performance Optimization

For best performance:

1. **Use Conda**: Conda environments are optimized for scientific computing
2. **Update Drivers**: Keep graphics drivers up to date
3. **Close Background Apps**: Free up system resources
4. **Use Double Pendulum**: For faster simulations during development
5. **Monitor Resources**: Use system monitor to identify bottlenecks

## Advanced Usage

### Command Line Arguments

Currently, the application doesn't support command-line arguments, but you can modify the code to:
- Set initial window size
- Choose default model
- Set initial torque values
- Configure simulation parameters

### Customization

You can customize the application by editing:
- `python/mujoco_golf_pendulum/models.py`: Modify pendulum models
- `python/mujoco_golf_pendulum/__main__.py`: Change GUI layout and controls
- `python/mujoco_golf_pendulum/sim_widget.py`: Adjust rendering and simulation parameters

### Integration with Other Tools

The simulation can be integrated with:
- Data logging scripts
- Analysis tools (MATLAB, Python scripts)
- Optimization algorithms
- Machine learning frameworks

See the main README.md for development guidelines.
