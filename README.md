# Golf Modeling Suite 🤖⛳

**A unified platform for golf swing analysis across multiple physics engines and modeling approaches**

> ✅ **STABLE**: The codebase migration is complete. See [Migration Status](docs/plans/migration_status.md) for details.

## Overview

The Golf Modeling Suite consolidates multiple golf swing modeling implementations into a single, cohesive platform. This repository provides comprehensive biomechanical analysis capabilities through:

- **5 Physics Engines**: MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite
- **Multiple Model Complexities**: From 2-DOF educational pendulums to 290-muscle musculoskeletal models
- **Advanced Biomechanics**: Muscle dynamics, inverse kinematics/dynamics, motion capture integration
- **Cross-Engine Validation**: Compare results across different physics engines
- **Professional GUI**: Interactive visualization and analysis tools
- **MATLAB Integration**: Simscape Multibody models for additional analysis

For detailed documentation, please visit the **[Documentation Hub](docs/README.md)**.

## Key Features

### 🦾 Musculoskeletal Modeling

- **MyoSuite Integration**: Hill-type muscle models with 290 muscles (full body)
- **OpenSim Integration**: Biomechanical model validation and analysis
- **Muscle Dynamics**: Force-length-velocity relationships, activation dynamics
- **Research-Grade**: Converted from validated OpenSim models (MoBL-ARMS, Rajagopal)

### 🎯 Advanced Analysis

- **Motion Capture**: Load and retarget mocap data (CSV, JSON, C3D)
- **Inverse Kinematics**: Professional IK solver with nullspace optimization
- **Inverse Dynamics**: Complete torque computation with force decomposition
- **Kinematic Forces**: Coriolis, centrifugal, and gravitational force analysis
- **Trajectory Optimization**: Generate optimal swings for speed, accuracy, or efficiency

### 🎮 Control & Robotics

- **Multiple Control Schemes**: Impedance, admittance, hybrid force-position, operational space
- **Constraint Analysis**: Parallel mechanism analysis of two-handed grip
- **Manipulability Analysis**: Singularity detection and workspace characterization
- **Task-Space Control**: End-effector control with redundancy resolution

### 📊 Visualization & Export

- **Real-Time 3D Rendering**: Multiple camera views with force/torque vectors
- **Comprehensive Plotting**: 10+ plot types including energy, phase diagrams, 3D trajectories
- **Data Export**: CSV and JSON formats for external analysis
- **Cross-Engine Comparison**: Validate results across different physics engines

## Quick Start

### Prerequisites

- **Python** 3.11+ (Python 3.13 recommended)
- **Git** with Git LFS
- **MATLAB** R2023a+ with Simulink and Simscape Multibody (optional, for MATLAB models)

### Installation

```bash
git clone https://github.com/D-sorganization/Golf_Modeling_Suite.git
cd Golf_Modeling_Suite
git lfs install
git lfs pull
pip install -r requirements.txt
```

### Launching the Suite

```bash
# Unified launcher (recommended) - select engine and model
python launchers/golf_launcher.py

# MuJoCo-specific launcher with advanced GUI
cd engines/physics_engines/mujoco/python
python -m mujoco_golf_pendulum

# Drake launcher
python engines/physics_engines/drake/python/golf_swing_launcher.py

# Pinocchio launcher
python engines/physics_engines/pinocchio/python/golf_swing_launcher.py
```

## Available Physics Engines

### MuJoCo (Recommended for Biomechanics)

- ✅ Full musculoskeletal models (MyoSuite integration)
- ✅ Contact dynamics (ground, ball)
- ✅ 2-28 DOF models with flexible shafts
- ✅ Advanced robotics features
- ✅ Motion capture workflow
- **See**: [engines/physics_engines/mujoco/README.md](engines/physics_engines/mujoco/README.md)

### Drake (Model-Based Design)

- ✅ Trajectory optimization
- ✅ Contact modeling
- ✅ System analysis tools
- ✅ URDF support
- **See**: [engines/physics_engines/drake/README.md](engines/physics_engines/drake/README.md)

### Pinocchio (Fast Rigid Body Algorithms)

- ✅ High-performance dynamics
- ✅ Jacobians and derivatives
- ✅ Constrained systems
- ✅ PINK inverse kinematics
- **See**: [engines/physics_engines/pinocchio/README.md](engines/physics_engines/pinocchio/README.md)

### OpenSim (Biomechanical Validation)

- ✅ Model validation against OpenSim
- ✅ Biomechanical analysis
- ✅ Integration with established workflows
- **See**: [engines/physics_engines/opensim/README.md](engines/physics_engines/opensim/README.md)

### MyoSuite (Muscle Modeling)

- ✅ Realistic muscle dynamics
- ✅ 290-muscle full body models
- ✅ MuJoCo-based simulation
- **See**: [engines/physics_engines/myosuite/README.md](engines/physics_engines/myosuite/README.md)

📖 **See [Engine Selection Guide](docs/engine_selection_guide.md) for detailed comparison and use cases**.

## Documentation

- **[User Guide](docs/user_guide/README.md)**: Installation, running simulations, and using the GUI
- **[Engines](docs/engines/README.md)**: Detailed engine documentation and comparison
- **[Development](docs/development/README.md)**: Contributing, architecture, and testing
- **[API Reference](docs/api/README.md)**: Code documentation and interfaces
- **[Plans & Roadmap](docs/plans/README.md)**: Implementation plans and future development
- **[Assessments](docs/assessments/README.md)**: Project reviews and implementation summaries
- **[Technical Docs](docs/technical/README.md)**: Engine reports and control strategies

### Recent Integration Guides

- **[MyoSuite Integration](docs/MYOSUITE_INTEGRATION.md)** - Biomechanics features (January 2026)
- **[OpenSim Integration](docs/OPENSIM_INTEGRATION.md)** - Musculoskeletal modeling (January 2026)

## Repository Structure

```
Golf_Modeling_Suite/
├── docs/                         # Comprehensive documentation
│   ├── user_guide/              # User documentation
│   ├── engines/                 # Engine-specific guides
│   ├── development/             # Development guides and PR docs
│   ├── plans/                   # Implementation plans
│   ├── assessments/             # Project reviews and summaries
│   ├── technical/               # Technical reports
│   └── api/                     # API documentation
├── launchers/                    # Unified launch applications
├── engines/
│   ├── physics_engines/         # Python physics engines
│   │   ├── mujoco/             # MuJoCo implementation
│   │   ├── drake/              # Drake implementation
│   │   ├── pinocchio/          # Pinocchio implementation
│   │   ├── opensim/            # OpenSim integration
│   │   └── myosuite/           # MyoSuite integration
│   ├── Simscape_Multibody_Models/ # MATLAB/Simulink models
│   └── pendulum_models/         # Simplified pendulum models
├── shared/                       # Common utilities and resources
│   ├── python/                  # Shared Python code
│   └── data/                    # Shared data and models
└── tools/                        # Analysis and development tools
```

## Contributing

We welcome contributions! Please see:

- [Contributing Guide](docs/development/contributing.md)
- [Development Guidelines](docs/development/README.md)
- [Testing Guide](docs/testing-guide.md)

## Citation

If you use this software in your research, please cite:

```bibtex
@software{golf_modeling_suite,
  title = {Golf Modeling Suite: A Unified Platform for Biomechanical Golf Swing Analysis},
  author = {[Authors]},
  year = {2026},
  url = {https://github.com/D-sorganization/Golf_Modeling_Suite}
}
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

This project integrates and builds upon several open-source projects:

- [MuJoCo](https://mujoco.org/) - Physics simulation
- [Drake](https://drake.mit.edu/) - Model-based design and control
- [Pinocchio](https://stack-of-tasks.github.io/pinocchio/) - Rigid body dynamics
- [MyoSuite](https://github.com/MyoHub/myosuite) - Musculoskeletal models
- [OpenSim](https://opensim.stanford.edu/) - Biomechanical modeling
