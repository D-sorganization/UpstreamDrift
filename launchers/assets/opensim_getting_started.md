# OpenSim Golf - Getting Started Guide

Welcome to OpenSim Golf Simulation! This guide will help you get started with the OpenSim physics engine for golf swing biomechanics.

## Prerequisites

### OpenSim Installation

OpenSim is a powerful open-source software for musculoskeletal modeling and simulation. To use OpenSim features:

**Option 1: Conda (Recommended)**

```bash
conda install -c opensim-org opensim
```

**Option 2: Manual Installation**

1. Download from [OpenSim Official Site](https://opensim.stanford.edu/)
2. Follow the installation guide for your platform
3. Ensure Python bindings are installed

### Alternative Engines

Don't have OpenSim? The Golf Modeling Suite supports multiple engines:

- **MuJoCo**: Fast, accurate, easy to install (`pip install mujoco`)
- **Pinocchio**: Analytical Jacobians, ideal for control (`pip install pin`)
- **Drake**: Block diagram modeling for complex systems

---

## Quick Start

### Step 1: Launch OpenSim GUI

From the main Golf Suite Launcher:

1. Click on the "OpenSim Golf" tile
2. The GUI will launch in **Getting Started** mode

### Step 2: Load an Existing Model

1. Click the **"Load Model"** button
2. Navigate to your `.osim` model file
3. Sample models are available in: `shared/models/opensim/`

### Step 3: Run Simulation

Once a model is loaded:

1. Click **"Run Simulation"**
2. View results in the integrated plots
3. Export data for further analysis

---

## Creating a New Model

### Option A: URDF Generator

The easiest way to create a new biomechanical model:

1. Launch **"URDF Generator"** from the main launcher
2. Design your golf swing model visually
3. Export as URDF format
4. Convert to `.osim` using OpenSim's import tools

### Option B: OpenSim GUI

For detailed musculoskeletal modeling:

1. Use the standalone OpenSim GUI application
2. Create your model with muscles, joints, and constraints
3. Save as `.osim` file
4. Load into Golf Modeling Suite

---

## Understanding OpenSim Models

### Key Components

| Component   | Description                   | Units   |
| ----------- | ----------------------------- | ------- |
| **Bodies**  | Rigid segments (arm, club)    | kg, m   |
| **Joints**  | Connections between bodies    | degrees |
| **Muscles** | Hill-type actuators           | N, m    |
| **Markers** | Virtual motion capture points | m       |

### File Format

OpenSim uses `.osim` files (XML-based):

```xml
<OpenSimDocument Version="40000">
  <Model name="golfer_swing">
    <BodySet>...</BodySet>
    <JointSet>...</JointSet>
    <ForceSet>...</ForceSet>
    <MarkerSet>...</MarkerSet>
  </Model>
</OpenSimDocument>
```

---

## Muscle Analysis Features

### Available Analysis

- **Muscle Forces**: Hill-type force-length-velocity curves
- **Moment Arms**: Torque contribution per muscle
- **Induced Acceleration**: Per-muscle contribution to joint motion
- **Grip Modeling**: Wrapping geometry for hand-club interface

### Example Code

```python
from engines.physics_engines.opensim.python.opensim_physics_engine import (
    OpenSimPhysicsEngine
)

engine = OpenSimPhysicsEngine("path/to/model.osim")
analyzer = engine.get_muscle_analyzer()
report = analyzer.analyze_all()

print(f"Biceps force: {report.muscle_forces['biceps']} N")
print(f"Total muscle torque: {report.total_muscle_torque} N·m")
```

---

## Troubleshooting

### "OpenSim Not Installed"

1. Verify installation: `python -c "import opensim; print(opensim.__version__)"`
2. Check conda environment is activated
3. Ensure Python version compatibility (3.8-3.11 recommended)

### "Model Load Failed"

1. Verify file exists and is readable
2. Check file is valid `.osim` format
3. Ensure geometry files are present (if referenced)

### "Simulation Not Available"

Full simulation integration requires additional setup. For quick results:

- Use MuJoCo or Pinocchio engines
- Or contact support for assistance

---

## References

- [OpenSim Documentation](https://simtk.org/projects/opensim)
- [Golf Modeling Suite Wiki](../../docs/README.md)
- [Section J: OpenSim Requirements](../../docs/assessments/project_design_guidelines.qmd)

---

## Getting Help

For assistance:

1. Check the `docs/` folder for detailed documentation
2. Review `OPENSIM_COMPLETE_SUMMARY.md` for implementation details
3. Open an issue on the repository

---

_Last Updated: January 2026_
