# MuJoCo Golf Simulation - GUI Architecture Documentation

## Overview

This document provides a comprehensive guide to all GUI applications in the Golf Modeling Suite's MuJoCo physics engine, their purposes, features, and when to use each one.

---

## 🎯 Primary GUIs

### 1. **Humanoid Golf Launcher** (RECOMMENDED for Humanoid Customization)

**File**: `engines/physics_engines/mujoco/python/humanoid_launcher.py`

**Framework**: PyQt6

**Purpose**: Dedicated launcher for the Carnegie Mellon/MuJoCo humanoid golf simulation with full customization support.

**Key Features**:
- ✅ **Body Segment Color Customization**
  - Shirt, pants, shoes, skin, eyes, club colors
  - Real-time color picker with visual preview
- ✅ **Physical Customization**
  - Height scaling (0.5m - 3.0m)
  - Weight scaling (50% - 200%)
- ✅ **Equipment Customization**
  - Club length (0.5m - 1.5m)
  - Club mass (0.1kg - 2.0kg)
- ✅ **Advanced Model Features**
  - Two-handed grip constraints
  - Enhanced facial features
  - Articulated finger segments
- ✅ **Polynomial Control Generator** ⭐ NEW
  - Visual polynomial function generation
  - Draw trends manually or input equations
  - Add and manipulate control points
  - Generate 6th-order polynomials for joint control
  - Per-joint polynomial configuration
  - Automatic coefficient saving
- ✅ **Simulation Modes**
  - Live Interactive View (requires VcXsrv on Windows)
  - Headless Mode (OSMesa rendering, no X11 required)
- ✅ **Control Systems**
  - PD Controller (default)
  - LQR Controller
  - Polynomial Controller (with visual generator)
- ✅ **State Management**
  - Save simulation states to disk
  - Load previous states
- ✅ **Docker Integration**
  - Automatic Docker container management
  - Real-time log streaming
  - Environment rebuild capability

**How to Launch**:
```bash
cd engines/physics_engines/mujoco/python
python humanoid_launcher.py
```

**Backend**: Uses `docker/src/humanoid_golf/sim.py` which properly reads `simulation_config.json`

**Configuration File**: `engines/physics_engines/mujoco/docker/src/simulation_config.json`

**Output Files**:
- Video: `engines/physics_engines/mujoco/docker/src/humanoid_golf.mp4`
- Data: `engines/physics_engines/mujoco/docker/src/golf_data.csv`

---

### 2. **Advanced Golf Analysis Window** (For Multi-Model Analysis)

**File**: `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/advanced_gui.py`

**Framework**: PyQt6

**Purpose**: Professional golf swing analysis tool with support for multiple biomechanical models.

**Key Features**:
- ✅ **Multiple Model Support**
  - Chaotic Pendulum (2 DOF)
  - Double Pendulum (2 DOF)
  - Triple Pendulum (3 DOF)
  - Upper Body + Arms (10 DOF)
  - Full Body with Legs (15 DOF)
  - Advanced Biomechanical (28 DOF)
  - MyoSuite Models (290+ muscles)
- ✅ **Interactive Manipulation**
  - Real-time pose editing
  - Drag-and-drop body segments
  - IK-based manipulation
  - Joint angle sliders
  - Constraint-aware positioning
- ✅ **Advanced Analysis**
  - Real-time biomechanical metrics
  - Force/torque visualization
  - Energy analysis
  - Joint angle tracking
- ✅ **Visualization**
  - Multiple camera presets
  - Force vector rendering
  - Contact force visualization
  - Actuator torque display
- ✅ **Data Export**
  - CSV export
  - JSON telemetry
  - Video recording
  - State snapshots
- ❌ **NO Color Customization** (uses default model colors)
- ❌ **NO Height/Weight Scaling** (uses model defaults)

**How to Launch**:
```bash
cd engines/physics_engines/mujoco/python
python -m mujoco_humanoid_golf
```

**Entry Point**: `mujoco_humanoid_golf/__main__.py` → `main()` function

**Use Case**: When you need to analyze different golf swing models, compare biomechanical approaches, or perform detailed swing analysis with real-time manipulation.

---

### 3. **Legacy DeepMind Control Suite GUI** (Tkinter Version)

**File**: `engines/physics_engines/mujoco/docker/gui/deepmind_control_suite_MuJoCo_GUI.py`

**Framework**: Tkinter

**Purpose**: Original Docker-based launcher (predecessor to Humanoid Golf Launcher).

**Key Features**:
- ✅ Body segment color customization
- ✅ Height/weight scaling
- ✅ Club parameters
- ✅ Docker integration
- ⚠️ **Older UI** (Tkinter vs PyQt6)
- ⚠️ **Less polished** than modern PyQt6 version

**How to Launch**:
```bash
cd engines/physics_engines/mujoco/docker/gui
python deepmind_control_suite_MuJoCo_GUI.py
```

**Status**: **DEPRECATED** - Use `humanoid_launcher.py` instead

**Note**: This was the original working version. The functionality has been ported to the modern PyQt6 `humanoid_launcher.py`.

---

## 🔧 Backend Simulation Modules

### Working Humanoid Golf Simulation

**File**: `engines/physics_engines/mujoco/docker/src/humanoid_golf/sim.py`

**Purpose**: Core simulation engine that applies customization settings.

**Features**:
- Reads `simulation_config.json`
- Applies color customization via `utils.customize_visuals()`
- Handles height/weight scaling
- Implements PD, LQR, and Polynomial controllers
- Supports live viewer (dm_control.viewer) and headless modes
- Generates video (MP4) and data (CSV) outputs

**Key Functions**:
- `run_simulation()`: Main entry point
- `utils.load_humanoid_with_props()`: Model loading with customization
- `utils.customize_visuals()`: Applies color settings
- `PDController`, `LQRController`, `PolynomialController`: Control systems

---

### CLI Runner (Headless Multi-Model)

**File**: `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/cli_runner.py`

**Purpose**: Headless batch simulation runner for multiple models.

**Features**:
- Command-line interface
- Batch processing support
- Model catalog (pendulums, upper body, full body, etc.)
- Telemetry export (JSON/CSV)
- Summary statistics

**How to Use**:
```bash
python -m mujoco_humanoid_golf.cli_runner --model full_body --duration 5.0 --summary
```

**Note**: Does NOT support color customization or height/weight scaling.

---

## 🎨 Interactive Manipulation Features

The **Advanced Golf Analysis Window** includes sophisticated interactive manipulation capabilities:

### Drag-and-Drop Manipulation

**File**: `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/interactive_manipulation.py`

**Features**:
- **IK-Based Dragging**: Click and drag body segments to new positions
- **Constraint-Aware**: Respects joint limits and kinematic constraints
- **Nullspace Posture**: Maintains natural poses while dragging
- **Orientation Lock**: Option to maintain segment orientation
- **Multi-Body Support**: Drag any body segment in the model

**How to Use**:
1. Launch Advanced Golf Analysis Window
2. Go to "Manipulation" tab
3. Enable "Enable Drag Manipulation"
4. Click on a body segment in the viewer
5. Drag to desired position
6. Release to apply IK solution

### Joint Angle Editor

**Features**:
- Real-time joint angle sliders
- Grouped by body part (legs, torso, arms, etc.)
- Numerical input fields
- Reset to default pose
- Save/load custom poses

**How to Use**:
1. Use sliders in the "Control" tab
2. Adjust joint angles in real-time
3. Click "Save State" to preserve pose
4. Click "Load State" to restore saved pose

### State Management

**Features**:
- Save simulation states (`.pkl` files)
- Load previous states
- Preserve joint angles, velocities, and positions
- Compatible across sessions

**File Format**: Python pickle format containing MuJoCo state data

---

## 📈 Polynomial Control Generator

The **Humanoid Golf Launcher** includes a visual polynomial function generator for creating complex control profiles.

### Overview

**File**: `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/polynomial_generator.py`

**Purpose**: Visual interface for generating 6th-order polynomial functions for joint control without manual coefficient entry.

### Features

- **Visual Drawing**: Draw control trends directly on the graph
- **Control Points**: Add and manipulate control points with mouse
- **Equation Input**: Enter mathematical equations (e.g., `sin(t)`, `t**2`)
- **Curve Manipulation**: Drag points to adjust polynomial shape
- **Per-Joint Configuration**: Generate different polynomials for each joint
- **Automatic Fitting**: Generates 6th-order polynomial coefficients automatically
- **Real-time Preview**: See polynomial curve as you design it

### How to Use

1. **Launch Humanoid Launcher**:
   ```bash
   cd engines/physics_engines/mujoco/python
   python humanoid_launcher.py
   ```

2. **Select Polynomial Mode**:
   - Go to "Simulation" tab
   - Set Control Mode to "poly"
   - Click "📊 Configure Polynomial" button

3. **Generate Polynomial**:
   - Select target joint from dropdown
   - Choose interaction mode:
     - **Draw**: Click and drag to draw trend
     - **Points**: Click to add control points
     - **Equation**: Enter mathematical expression
   - Click "Fit Polynomial" to generate coefficients
   - Coefficients are automatically saved to config

4. **Run Simulation**:
   - Close polynomial generator
   - Click "RUN SIMULATION"
   - Simulation uses generated polynomials for control

### Workflow Example

**Creating a Swing Motion**:

1. Select "rhumerusrx" (right shoulder) joint
2. Draw a smooth curve representing the swing:
   - Start at 0 (address position)
   - Rise to peak (backswing)
   - Drop sharply (downswing)
   - Level off (follow-through)
3. Fit polynomial
4. Repeat for other joints (elbow, wrist, etc.)
5. Run simulation to see coordinated swing

### Mathematical Background

- **Polynomial Order**: 6th order (7 coefficients: c0 through c6)
- **Function Form**: `u(t) = c0 + c1*t + c2*t² + c3*t³ + c4*t⁴ + c5*t⁵ + c6*t⁶`
- **Time Range**: Typically 0 to simulation duration
- **Fitting Method**: Least squares polynomial regression

### Available Joints

The generator supports all humanoid joints:
- **Spine**: lowerbackrx, upperbackrx
- **Legs**: rtibiarx, ltibiarx, rfemurrx, lfemurrx, rfootrx, lfootrx
- **Arms**: rhumerusrx, lhumerusrx, rhumerusrz, lhumerusrz, rhumerusry, lhumerusry
- **Forearms**: rradiusrx, lradiusrx

### Tips for Best Results

1. **Start Simple**: Begin with one or two joints
2. **Smooth Curves**: Avoid sharp discontinuities
3. **Reasonable Magnitudes**: Keep torques within ±100 Nm
4. **Test Incrementally**: Generate, test, refine
5. **Save Configurations**: Use state save/load for good profiles

---

## 📊 Comparison Matrix

| Feature | Humanoid Launcher | Advanced Analysis | Legacy Tkinter |
|---------|------------------|-------------------|----------------|
| **Framework** | PyQt6 | PyQt6 | Tkinter |
| **Color Customization** | ✅ | ❌ | ✅ |
| **Height/Weight Scaling** | ✅ | ❌ | ✅ |
| **Polynomial Generator** | ✅ | ❌ | ❌ |
| **Interactive Manipulation** | ❌ | ✅ | ❌ |
| **Multiple Models** | ❌ (Humanoid only) | ✅ | ❌ (Humanoid only) |
| **Biomechanical Analysis** | ❌ | ✅ | ❌ |
| **Real-time Plotting** | ❌ | ✅ | ❌ |
| **Docker Integration** | ✅ | ❌ | ✅ |
| **State Save/Load** | ✅ | ✅ | ✅ |
| **Modern UI** | ✅ | ✅ | ❌ |
| **Status** | **ACTIVE** | **ACTIVE** | DEPRECATED |

---

## 🚀 Quick Start Guide

### For Humanoid Customization (Colors, Size, Equipment)
```bash
cd engines/physics_engines/mujoco/python
python humanoid_launcher.py
```
1. Customize appearance in "Appearance" tab
2. Adjust equipment in "Equipment" tab
3. Configure simulation in "Simulation" tab
4. Click "RUN SIMULATION"

### For Interactive Pose Editing
```bash
cd engines/physics_engines/mujoco/python
python -m mujoco_humanoid_golf
```
1. Select model from dropdown
2. Use joint sliders to adjust pose
3. Or enable drag mode and click-drag body segments
4. Save state when satisfied

### For Batch Analysis
```bash
cd engines/physics_engines/mujoco/python
python -m mujoco_humanoid_golf.cli_runner --model full_body --duration 5.0 --output-csv results.csv
```

---

## 🐛 Troubleshooting

### X11 Segmentation Faults (Live View Mode)

**Problem**: Simulation crashes with segfault when using Live Interactive View

**Solution**:
1. Ensure VcXsrv is running (Windows)
2. Check "Disable access control" in VcXsrv settings
3. Verify VcXsrv icon is in system tray
4. **Alternative**: Use Headless Mode (uncheck "Live Interactive View")

### Pixelated/Jumbled Display

**Problem**: GUI appears pixelated or distorted over X11

**Solution**: Already fixed in `humanoid_launcher.py` with Qt scaling environment variables:
- `QT_AUTO_SCREEN_SCALE_FACTOR=0`
- `QT_SCALE_FACTOR=1`
- `QT_QPA_PLATFORM=xcb`

### Configuration Not Applied

**Problem**: Color/size changes don't appear in simulation

**Solution**: Ensure you're using `humanoid_launcher.py`, not the advanced GUI. The advanced GUI doesn't support customization.

### Output Files Not Found

**Problem**: "Open Video" or "Open Data" buttons report file not found

**Solution**: Check `engines/physics_engines/mujoco/docker/src/` for output files. They're generated in the Docker working directory.

---

## 📝 Configuration File Format

**Location**: `engines/physics_engines/mujoco/docker/src/simulation_config.json`

**Example**:
```json
{
  "colors": {
    "shirt": [0.6, 0.6, 0.6, 1.0],
    "pants": [0.4, 0.2, 0.0, 1.0],
    "shoes": [0.1, 0.1, 0.1, 1.0],
    "skin": [0.8, 0.6, 0.4, 1.0],
    "eyes": [1.0, 1.0, 1.0, 1.0],
    "club": [0.8, 0.8, 0.8, 1.0]
  },
  "height_m": 1.8,
  "weight_percent": 100.0,
  "control_mode": "pd",
  "live_view": false,
  "save_state_path": "",
  "load_state_path": "",
  "club_length": 1.0,
  "club_mass": 0.5,
  "two_handed": false,
  "enhance_face": false,
  "articulated_fingers": false
}
```

**Color Format**: RGBA values in range [0.0, 1.0]

---

## 🔄 Migration Guide

### From Legacy Tkinter GUI to Modern PyQt6

If you were using `deepmind_control_suite_MuJoCo_GUI.py`:

1. **Switch to**: `humanoid_launcher.py`
2. **Same features**: All customization features are preserved
3. **Better UI**: Modern PyQt6 interface with improved styling
4. **Same backend**: Uses the same `humanoid_golf.sim` module

### From Advanced GUI to Humanoid Launcher

If you need color customization but are using the advanced GUI:

1. **Use**: `humanoid_launcher.py` for customization
2. **Then**: Use advanced GUI for pose editing if needed
3. **Workflow**: Customize → Simulate → Analyze

---

## 📚 Additional Resources

- **MuJoCo Documentation**: https://mujoco.readthedocs.io/
- **DeepMind Control Suite**: https://github.com/deepmind/dm_control
- **PyQt6 Documentation**: https://www.riverbankcomputing.com/static/Docs/PyQt6/

---

## 🤝 Contributing

When adding new GUI features:

1. **Document** in this README
2. **Update** comparison matrix
3. **Add** troubleshooting section if needed
4. **Test** with Docker integration
5. **Verify** configuration file compatibility

---

**Last Updated**: 2025-12-20

**Maintainer**: Golf Modeling Suite Team
