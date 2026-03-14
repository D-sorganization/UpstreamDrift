# UpstreamDrift

<p align="center">
  <img src="assets/branding/logo.png" alt="UpstreamDrift Logo" width="200"/>
</p>

<p align="center">
  <a href="https://github.com/D-sorganization/UpstreamDrift/actions/workflows/ci-standard.yml"><img src="https://github.com/D-sorganization/UpstreamDrift/actions/workflows/ci-standard.yml/badge.svg" alt="CI Standard"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
</p>

<p align="center">
  <strong>A unified platform for golf swing analysis across multiple physics engines and modeling approaches</strong>
</p>

---

## Overview

UpstreamDrift (formerly Golf Modeling Suite) consolidates multiple golf swing modeling implementations into a single, cohesive platform. This repository provides comprehensive biomechanical analysis capabilities through:

- **Tiered Engine Support**: MuJoCo as the supported default, Drake/Pinocchio for extended cross-engine work, OpenSim/MyoSuite as experimental integrations
- **Multiple Model Complexities**: From 2-DOF educational pendulums to 290-muscle musculoskeletal models
- **Advanced Biomechanics**: Muscle dynamics, inverse kinematics/dynamics, motion capture integration
- **Cross-Engine Validation**: Compare results across different physics engines
- **Professional GUI**: Interactive visualization and analysis tools
- **MATLAB Integration**: Simscape Multibody models for additional analysis

For detailed documentation, please visit the **[Documentation Hub](docs/README.md)**.

## Key Features

### Musculoskeletal Modeling

- **MyoSuite Integration**: Experimental muscle-modeling surface for future biomechanics work
- **OpenSim Integration**: Experimental biomechanics validation surface
- **Muscle Dynamics**: Force-length-velocity relationships, activation dynamics
- **Research Lineage**: Includes conversions from established OpenSim model sources where integrations remain under active development

### Advanced Analysis

- **Motion Capture**: Load and retarget mocap data (CSV, JSON, C3D) using OpenPose or MediaPipe.
- **Model Explorer**: Interactive browser for Humanoid, Pendulum, and Robotic models.
- **Inverse Kinematics**: Professional IK solver with nullspace optimization
- **Inverse Dynamics**: Complete torque computation with force decomposition
- **Kinematic Forces**: Coriolis, centrifugal, and gravitational force analysis
- **Trajectory Optimization**: Generate optimal swings for speed, accuracy, or efficiency

### Control and Robotics

- **Multiple Control Schemes**: Impedance, admittance, hybrid force-position, operational space
- **Constraint Analysis**: Parallel mechanism analysis of two-handed grip
- **Manipulability Analysis**: Singularity detection and workspace characterization
- **Task-Space Control**: End-effector control with redundancy resolution

### Visualization and Export

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

**Recommended: Conda** (handles binary dependencies like MuJoCo)

```bash
git clone https://github.com/D-sorganization/UpstreamDrift.git
cd UpstreamDrift
git lfs install && git lfs pull

# Create conda environment (most reliable)
conda env create -f environment.yml
conda activate upstream-drift

# Verify installation
python scripts/verify_installation.py
```

**Alternative: Pip**

```bash
# Matches the required PR CI surface (MuJoCo + dev tooling)
pip install -e ".[dev]"

# Adds Drake and Pinocchio for cross-engine work
pip install -e ".[dev,all-engines]"

# Adds OpenSim and MyoSuite integrations on a best-effort basis
pip install -e ".[dev,biomechanics]"
```

**Light Installation** (for UI development without heavy physics engines)

```bash
pip install -e .
export GOLF_USE_MOCK_ENGINE=1
```

**Troubleshooting**: See [docs/troubleshooting/installation.md](docs/troubleshooting/installation.md) for common issues.

### Supported Engine Tiers

| Tier         | Engines           | Install Profile                        | Validation                                              |
| ------------ | ----------------- | -------------------------------------- | ------------------------------------------------------- |
| Supported    | MuJoCo            | `pip install -e ".[dev]"`              | Required PR CI                                          |
| Extended     | Drake, Pinocchio  | `pip install -e ".[dev,all-engines]"`  | Nightly cross-engine validation and targeted local runs |
| Experimental | OpenSim, MyoSuite | `pip install -e ".[dev,biomechanics]"` | Best-effort local validation                            |

See [docs/engines/support_tiers.md](docs/engines/support_tiers.md) for the
full contract and [docs/engines/engine_capabilities.md](docs/engines/engine_capabilities.md)
for feature-level support.

### Development Setup

Use the Makefile for common development tasks:

```bash
make help      # Show available targets
make install   # Install dependencies
make check     # Run linters and tests
make format    # Format code with black and ruff
```

### Launching the Suite

The suite now features a **Unified Launcher** that provides access to all engines and tools from a single interface.

```bash
# Unified launcher (recommended) - select engine and model
python3 launch_golf_suite.py

# Alternative: Direct launch of specific engines
python3 src/engines/physics_engines/mujoco/python/humanoid_launcher.py
python3 src/engines/physics_engines/drake/python/src/golf_gui.py
```

## Available Physics Engines

### MuJoCo (Recommended for Biomechanics)

- Default simulation stack for contact-rich dynamics and day-to-day development
- Contact dynamics (ground, ball)
- 2-28 DOF models with flexible shafts
- Advanced robotics features
- Motion capture workflow (OpenPose & MediaPipe)
- **See**: [src/engines/physics_engines/mujoco/README.md](src/engines/physics_engines/mujoco/README.md)

### Drake (Model-Based Design)

- Trajectory optimization
- Contact modeling
- System analysis tools
- URDF support
- **See**: [src/engines/physics_engines/drake/README.md](src/engines/physics_engines/drake/README.md)

### Pinocchio (Fast Rigid Body Algorithms)

- High-performance dynamics
- Jacobians and derivatives
- Constrained systems
- PINK inverse kinematics
- **See**: [src/engines/physics_engines/pinocchio/README.md](src/engines/physics_engines/pinocchio/README.md)

### OpenSim (Experimental Biomechanics)

- Stub-oriented integration surface for future biomechanics validation
- Not part of the required PR CI contract today
- Use when explicitly working on biomechanics integration tasks
- **See**: [src/engines/physics_engines/opensim/README.md](src/engines/physics_engines/opensim/README.md)

### MyoSuite (Experimental Muscle Modeling)

- Stub-oriented integration surface for future muscle-modeling work
- Not part of the required PR CI contract today
- Use when explicitly working on biomechanics integration tasks
- **See**: [src/engines/physics_engines/myosuite/README.md](src/engines/physics_engines/myosuite/README.md)

**See [Engine Selection Guide](docs/engine_selection_guide.md) for detailed comparison and use cases.**
**See [Supported Engine Tiers](docs/engines/support_tiers.md) for the support and validation contract.**

## Documentation

- **[User Guide](docs/user_guide/README.md)**: Installation, running simulations, and using the GUI
- **[Engines](docs/engines/README.md)**: Detailed engine documentation and comparison
- **[Supported Engine Tiers](docs/engines/support_tiers.md)**: Install profiles and CI coverage expectations
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
UpstreamDrift/
├── docs/                         # Comprehensive documentation
│   ├── user_guide/              # User documentation
│   ├── engines/                 # Engine-specific guides
│   ├── development/             # Development guides and PR docs
│   ├── plans/                   # Implementation plans
│   ├── assessments/             # Project reviews and summaries
│   ├── technical/               # Technical reports
│   └── api/                     # API documentation
├── src/
│   ├── launchers/               # Unified launch applications
│   ├── engines/
│   │   ├── physics_engines/     # Python physics engines
│   │   │   ├── mujoco/          # MuJoCo implementation
│   │   │   ├── drake/           # Drake implementation
│   │   │   ├── pinocchio/       # Pinocchio implementation
│   │   │   ├── opensim/         # OpenSim integration
│   │   │   └── myosuite/        # MyoSuite integration
│   │   ├── Simscape_Multibody_Models/ # MATLAB/Simulink models
│   │   └── pendulum_models/     # Simplified pendulum models
│   ├── shared/                  # Source-level shared code
│   └── tools/                   # Source-level tools
├── shared/                      # Shared assets and vendored dependencies
└── tools/                       # Root-level helper scripts and workflows
```

## Contributing

We welcome contributions! Please see:

- [Contributing Guide](docs/development/contributing.md)
- [Development Guidelines](docs/development/README.md)
- [Testing Guide](docs/testing-guide.md)

## Citation

If you use this software in your research, please cite:

```bibtex
@software{upstream_drift,
  title = {UpstreamDrift: A Unified Platform for Biomechanical Golf Swing Analysis},
  author = {Dieter Olson},
  year = {2026},
  url = {https://github.com/D-sorganization/UpstreamDrift}
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
