# MuJoCo Golf Swing Biomechanical Analysis Suite

A professional-grade physics-based golf swing simulation and analysis system using MuJoCo, featuring comprehensive biomechanical analysis, force/torque visualization, and advanced plotting capabilities.

## Overview

This repository contains:
- **Eight Progressive Swing Models**: From educational pendulums (2 DOF) to research-grade biomechanical (28 DOF) and complete musculoskeletal models (290 muscles)
- **Musculoskeletal Models**: MyoSuite integration with realistic muscle-tendon units and physiological constraints
- **Advanced Analysis Suite**: Comprehensive biomechanical analysis with force/torque extraction and visualization
- **Professional GUI**: Tabbed interface with controls, visualization, analysis, and plotting
- **Real-Time Data Recording**: Capture complete swing kinematics and kinetics
- **Advanced Plotting**: 10+ plot types including energy analysis, phase diagrams, and 3D trajectories
- **Data Export**: CSV and JSON export for external analysis
- **Force/Torque Visualization**: Adjustable 3D vector rendering of forces and torques
- **MATLAB Support**: Additional analysis tools and scripts

### 🦾 **NEW: Complete Musculoskeletal Models**

The simulator now includes three physiologically realistic musculoskeletal models from [MyoSuite](https://github.com/MyoHub/myosuite):

#### Available Models
- **MyoUpperBody (19 DOF, 20 actuators)**: Torso + head + bilateral arms with muscle-based control
- **MyoBody Full (52 DOF, 290 muscles)**: Complete full-body model with all major muscle groups
- **MyoArm Simple (14 DOF, 14 actuators)**: Simplified bilateral arms for rapid analysis

#### Features
- **Realistic Muscle Dynamics**: Hill-type muscle models with force-length-velocity relationships
- **Physiological Accuracy**: Converted from validated OpenSim models (MoBL-ARMS, Rajagopal)
- **High Performance**: 60-4000× faster than OpenSim equivalents
- **Biomechanical Analysis**: Study muscle recruitment, co-contraction, and power generation
- **Research Applications**: Golf biomechanics, injury prevention, performance optimization

See **[Musculoskeletal Models Guide](docs/MUSCULOSKELETAL_MODELS.md)** for complete documentation.

### 🔗 **NEW: Universal & Gimbal Joint Models**

**Latest Addition** introduces advanced joint models for studying constraint forces and torque transmission:

#### Universal Joint Models
- **Two-Link Inclined Plane**: Universal joint at wrist for studying torque wobble
- **Constraint Force Analysis**: Real-time force/torque measurement at joints
- **Torque Transmission**: Analyze velocity and torque ratios through U-joints
- **Realistic Wrist Mechanics**: 2-DOF universal joints matching human anatomy

#### Gimbal Joint Models
- **3-DOF Gimbal Demonstration**: Nested ring structure for full orientation control
- **Gimbal Lock Detection**: Automatic singularity warnings
- **Shoulder Joints**: Anatomically accurate 3-DOF shoulder implementation

#### Flexible Golf Clubs
- **Multi-Segment Beam Model**: 1-5 flexible shaft segments
- **Realistic Club Configurations**: Driver, 7-Iron, Wedge with proper inertia
- **Configurable Stiffness**: Based on real club measurements
- **Rigid vs Flexible**: Compare shaft dynamics effects

See **[Joint Models Guide](docs/JOINT_MODELS_GUIDE.md)** and **[Quick Start](docs/ADVANCED_JOINTS_README.md)** for complete documentation.


**Version 2.1** adds complete motion capture workflow and kinematic force analysis:

#### 📹 Motion Capture Integration
- **Motion Capture Loading**: CSV, JSON formats with automatic parsing
- **Motion Retargeting**: IK-based mapping from markers to joint angles
- **Kinematic Processing**: Filtering, derivatives, time normalization
- **Marker-Based Analysis**: Support for standard golf biomechanics marker sets

#### ⚡ Kinematic Force Analysis
- **Coriolis Forces**: Compute velocity-dependent coupling forces
- **Centrifugal Forces**: Rotation-induced outward forces
- **Gravitational Forces**: Configuration-dependent weight effects
- **Club Head Forces**: Apparent forces in rotating reference frame
- **Power Analysis**: Energy dissipation from dynamic effects
- **NO Inverse Dynamics Required**: Analyze forces from kinematics alone!

#### 🔧 Inverse Dynamics Framework
- **Full Inverse Dynamics**: Complete torque computation for any motion
- **Partial Solutions**: Handle parallel mechanism constraints
- **Force Decomposition**: Break down into inertial, Coriolis, gravity
- **Validation Tools**: Verify solutions with forward dynamics

See **[Motion Capture Guide](docs/MOTION_CAPTURE_GUIDE.md)** for complete workflow documentation.

### 🤖 **Advanced Robotics Features (v2.0)**

**Version 2.0** brought professional-grade parallel mechanism robotics and advanced control capabilities:

#### 🤖 Advanced Kinematics & Analysis
- **Constraint Jacobian Analysis**: Analyze closed-chain systems (two-handed grip as parallel mechanism)
- **Manipulability Analysis**: Singularity detection, condition number analysis, manipulability ellipsoids
- **Inverse Kinematics**: Professional IK solver with Damped Least-Squares and nullspace optimization
- **Task-Space Control**: Control end-effector in Cartesian coordinates with redundancy resolution

#### 🎮 Multiple Control Schemes
- **Impedance Control**: Position-based compliant control with adjustable stiffness/damping
- **Admittance Control**: Force-based control for compliant interaction
- **Hybrid Force-Position Control**: Simultaneous force and position objectives
- **Computed Torque Control**: Model-based feedforward using inverse dynamics
- **Operational Space Control**: Advanced task-space control with inertia compensation

#### 🎯 Motion Optimization & Planning
- **Trajectory Optimization**: Generate optimal swings maximizing club speed, minimizing energy
- **Multi-Objective Optimization**: Balance speed, accuracy, smoothness, and efficiency
- **Motion Synthesis**: Suggest optimal motion inputs for desired swing characteristics
- **Motion Primitive Libraries**: Store and blend pre-computed optimal motions

See **[Advanced Robotics Guide](docs/ADVANCED_ROBOTICS_GUIDE.md)** for complete documentation and examples.

### 📊 Biomechanical Analysis Suite

The core analysis features:
- **Force & Torque Visualization**: Real-time 3D vector rendering with user-adjustable scaling
- **Comprehensive Plotting**: Summary dashboards, time series, phase diagrams, 3D trajectories
- **Biomechanical Metrics**: Club head speed, energy analysis, power curves, ground reaction forces
- **Data Export**: Export complete swing data to CSV/JSON for advanced analysis
- **Professional GUI**: Tabbed interface with controls, visualization, analysis, and plotting panels

See **[Analysis Suite Documentation](docs/ANALYSIS_SUITE.md)** for complete feature details and workflow guide.

### Model Progression

1. **Double Pendulum (2 DOF)** - Educational: shoulder + wrist
2. **Triple Pendulum (3 DOF)** - Educational: shoulder + elbow + wrist
3. **Upper Body Golf Swing (10 DOF)** - Realistic: pelvis + spine + both arms + club
4. **Full Body Golf Swing (15 DOF)** - Complete: legs + pelvis + torso + both arms + club
5. **Advanced Biomechanical (28 DOF)** - Research: scapulae + 3-DOF shoulders + 2-DOF wrists + 3-DOF spine + flexible shaft

All models include USGA-regulation golf balls with realistic contact physics.

## Project Structure

```
MuJoCo_Golf_Swing_Model/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Root requirements (template - use python/requirements.txt)
├── .gitignore                   # Git ignore rules
│
├── python/                      # Python implementation
│   ├── requirements.txt        # Python dependencies (USE THIS)
│   ├── environment.yml         # Conda environment specification
│   ├── mujoco_golf_pendulum/   # Main package
│   │   ├── __init__.py
│   │   ├── __main__.py         # Entry point for GUI application
│   │   ├── models.py           # MuJoCo XML model definitions
│   │   └── sim_widget.py       # Qt widget for simulation rendering
│   ├── src/                    # Shared utilities
│   │   ├── constants.py        # Physical constants and golf specifications
│   │   └── logger_utils.py     # Logging utilities
│   └── tests/                  # Unit tests
│
├── matlab/                     # MATLAB scripts and analysis
│   ├── run_all.m              # Main MATLAB script
│   └── tests/                 # MATLAB tests
│
├── docs/                       # Documentation
│   ├── README.md              # Development guidelines
│   ├── CHANGELOG.md           # Version history
│   └── GUARDRAILS_GUIDELINES.md
│
├── data/                       # Data files
│   └── raw/                   # Raw data (gitignored)
│
├── output/                     # Simulation outputs
│
└── scripts/                    # Utility scripts
    ├── quality_check.py
    ├── setup_precommit.sh
    └── snapshot.sh
```

## Quick Start

### Prerequisites

- Python 3.10+ (Python 3.13 recommended)
- MuJoCo Python bindings
- PyQt6
- See [GUI Setup Guide](docs/GUI_SETUP_GUIDE.md) for detailed system requirements and installation instructions

### Installation

#### Option 1: Using Conda (Recommended)

```bash
# Create and activate conda environment
conda env create -f python/environment.yml
conda activate sim-env

# Install additional dependencies (if not already in environment.yml)
pip install mujoco PyQt6
```

#### Option 2: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r python/requirements.txt
```

### Running the Simulation

#### Quick Launch (Recommended)

Use the launcher scripts for easy startup:

**Windows:**
```batch
run_gui.bat
```

**macOS/Linux:**
```bash
./run_gui.sh
```

#### Manual Launch

```bash
# From the repository root
python -m python.mujoco_golf_pendulum

# Or from the python directory
cd python
python -m mujoco_golf_pendulum
```

The Advanced Analysis GUI will launch with:
- **Controls Tab**: Model selection, simulation controls, recording, actuator torque sliders
- **Visualization Tab**: Camera views, force/torque vector visualization with adjustable scaling
- **Analysis Tab**: Real-time biomechanical metrics, data export (CSV/JSON)
- **Plotting Tab**: 10+ plot types including dashboards, time series, phase diagrams, 3D trajectories

**For detailed setup instructions, troubleshooting, and usage guide, see [GUI Setup Guide](docs/GUI_SETUP_GUIDE.md)**
**For complete analysis features and workflow guide, see [Analysis Suite Documentation](docs/ANALYSIS_SUITE.md)**

### Quick Start with Advanced Robotics Features

Run the comprehensive examples demonstrating all advanced robotics capabilities:

```bash
cd python
python -m mujoco_golf_pendulum.examples_advanced_robotics
```

This will run 8 examples covering:
1. **Constraint Jacobian Analysis** - Parallel mechanism analysis of two-handed grip
2. **Manipulability Analysis** - Singularity detection and condition number analysis
3. **Inverse Kinematics** - Professional IK solver with nullspace optimization
4. **Impedance Control** - Compliant position control
5. **Hybrid Force-Position Control** - Simultaneous force and position objectives
6. **Trajectory Optimization** - Generate optimal swings
7. **Motion Primitive Library** - Store and blend motion primitives
8. **Singularity Analysis** - Comprehensive workspace analysis

Or use individual features in your code:

```python
from mujoco_golf_pendulum.advanced_kinematics import AdvancedKinematicsAnalyzer
from mujoco_golf_pendulum.advanced_control import AdvancedController, ControlMode
from mujoco_golf_pendulum.motion_optimization import SwingOptimizer

# Analyze constraint Jacobian
analyzer = AdvancedKinematicsAnalyzer(model, data)
constraint_data = analyzer.compute_constraint_jacobian()

# Use impedance control
controller = AdvancedController(model, data)
controller.set_control_mode(ControlMode.IMPEDANCE)
tau = controller.compute_control(target_position, target_velocity)

# Optimize swing trajectory
optimizer = SwingOptimizer(model, data)
result = optimizer.optimize_swing_for_speed(target_speed=45.0)  # m/s
```

See **[Advanced Robotics Guide](docs/ADVANCED_ROBOTICS_GUIDE.md)** for complete documentation.

### Quick Start with Motion Capture

Run the motion capture workflow examples:

```bash
cd python
python -m mujoco_golf_pendulum.examples_motion_capture
```

This runs 6 examples covering:
1. **Load Motion Capture** - CSV and JSON format support
2. **Motion Retargeting** - IK-based mapping to model
3. **Kinematic Forces** - Coriolis, centrifugal, gravity (KEY FEATURE!)
4. **Inverse Dynamics** - Required torques with decomposition
5. **Complete Pipeline** - Full workflow from mocap to forces
6. **Swing Comparison** - Quantitative comparison of different swings

Or analyze your own captured motion:

```python
from mujoco_golf_pendulum.motion_capture import MotionCaptureLoader, MotionRetargeting
from mujoco_golf_pendulum.kinematic_forces import KinematicForceAnalyzer
from mujoco_golf_pendulum.inverse_dynamics import InverseDynamicsSolver

# Load motion capture
mocap_seq = MotionCaptureLoader.load_csv('player_swing.csv')

# Retarget to model
retargeting = MotionRetargeting(model, data, marker_set)
times, joint_traj, success = retargeting.retarget_sequence(mocap_seq)

# Analyze kinematic forces (NO inverse dynamics needed!)
analyzer = KinematicForceAnalyzer(model, data)
force_data = analyzer.analyze_trajectory(times, positions, velocities, accelerations)

# See Coriolis and centrifugal forces
for fd in force_data:
    print(f"Coriolis forces: {fd.coriolis_forces}")
    print(f"Centrifugal forces: {fd.centrifugal_forces}")
    print(f"Coriolis power: {fd.coriolis_power} W")
```

See **[Motion Capture Guide](docs/MOTION_CAPTURE_GUIDE.md)** for complete workflow.

## Features

- **Physics Simulation**: Accurate MuJoCo physics engine with RK4 integration
- **Five Model Complexities**:
  - **Simple models**: Double and triple pendulum for education
  - **Upper body model**: 10 DOF with realistic torso and arm biomechanics
  - **Full body model**: 15 DOF with legs, enabling weight transfer analysis
  - **Advanced biomechanical**: 28 DOF research-grade model with scapulae and flexible shaft
- **Biomechanically Realistic**:
  - Anthropometric body segment dimensions and masses (de Leva 1996)
  - Physiologically accurate joint ranges (Kapandji 2019)
  - Two-handed club grip with equality constraints
  - USGA-regulation ball and club specifications
  - Research-grade inertia tensors from literature
- **Advanced Features** (Research Model):
  - **Scapular kinematics**: 2-DOF scapulae for realistic shoulder mechanics
  - **3-DOF shoulders**: Full ball-and-socket range of motion
  - **2-DOF wrists**: Flexion/extension + radial/ulnar deviation
  - **3-DOF spine**: Lateral bend, sagittal bend, axial rotation
  - **2-DOF ankles**: Plantarflexion + inversion/eversion
  - **Flexible golf shaft**: 3-segment model with stiffness gradient
  - **Realistic club head**: Hosel, face, crown, sole, alignment aid
- **Interactive Controls**:
  - Dynamic GUI adapts to model complexity
  - Grouped controls (Legs, Torso/Spine, Scapulae, Arms, Club/Shaft)
  - Real-time torque adjustment for all actuators
  - Up to 28 independent actuator sliders
- **Advanced Visualization**:
  - Five camera views (side, front, top, follow, down-the-line)
  - 3D rendering at 60 FPS
  - High-quality materials and lighting with specularity
  - **Force/torque vector visualization** with user-adjustable scaling
  - **Contact force visualization** for ground reactions
  - Real-time biomechanical metrics display
- **Comprehensive Analysis Suite**:
  - **Real-time data recording** of complete biomechanical state
  - **10+ plot types**: Dashboards, time series, phase diagrams, 3D trajectories
  - **Biomechanical metrics**: Joint kinematics, kinetics, energy, power
  - **Club head analysis**: Speed tracking, trajectory visualization
  - **Energy analysis**: Kinetic, potential, total energy over time
  - **Phase space analysis**: State-space plots for any joint
  - **Data export**: CSV and JSON formats for external analysis
- **Extensible Framework**: Ready for motion capture integration, controller development, and swing optimization

See [Golf Swing Models Documentation](docs/GOLF_SWING_MODELS.md) and [Advanced Biomechanical Model](docs/ADVANCED_BIOMECHANICAL_MODEL.md) for detailed technical specifications.

## Development

### Code Quality

This project uses:
- `ruff` for linting and formatting
- `mypy` for type checking
- `pytest` for testing
- `pre-commit` hooks for automated checks

### Running Tests

```bash
# Python tests
cd python
pytest

# Validate all MuJoCo models
python validate_models.py

# MATLAB tests
cd matlab
# Run tests/test_example.m
```

### Pre-commit Setup

```bash
bash scripts/setup_precommit.sh
```

## Documentation

- **[Motion Capture Guide](docs/MOTION_CAPTURE_GUIDE.md)** ⭐ **NEW v2.1** - Complete workflow for motion capture integration, kinematic force analysis, and inverse dynamics
- **[Advanced Robotics Guide](docs/ADVANCED_ROBOTICS_GUIDE.md)** ⭐ **v2.0** - Professional-grade parallel mechanism analysis, control schemes, and motion optimization
- **[Biomechanical Analysis Suite](docs/ANALYSIS_SUITE.md)** - Complete guide to force/torque visualization, plotting, data recording, and analysis workflows
- **[Advanced Biomechanical Model](docs/ADVANCED_BIOMECHANICAL_MODEL.md)** - Complete specification for the 28-DOF research model with scapulae, flexible shaft, and anthropometric data
- **[Golf Swing Models Technical Documentation](docs/GOLF_SWING_MODELS.md)** - Detailed biomechanical specifications for all models, features, and development opportunities
- **[GUI Setup and Usage Guide](docs/GUI_SETUP_GUIDE.md)** - Complete guide for installing and using the GUI application
- [Development Guidelines](docs/README.md) - Setup and workflow
- [Changelog](docs/CHANGELOG.md) - Version history
- [Guardrails Guidelines](docs/GUARDRAILS_GUIDELINES.md) - Safety practices

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes following the project's coding standards
3. Run quality checks: `python scripts/quality_check.py`
4. Commit with descriptive messages
5. Push and create a pull request

## Notes

- The MuJoCo models are defined as XML strings in `python/mujoco_golf_pendulum/models.py`
- Physical constants are in `python/src/constants.py` with full citations
- Simulation outputs go to `output/` directory
- Raw data should be placed in `data/raw/` (gitignored)
