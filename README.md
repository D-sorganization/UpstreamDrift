# UpstreamDrift

<p align="center">
  <img src="assets/branding/logo.png" alt="UpstreamDrift Logo" width="200"/>
</p>

<p align="center">
  <a href="https://github.com/D-sorganization/UpstreamDrift/actions/workflows/ci-standard.yml"><img src="https://github.com/D-sorganization/UpstreamDrift/actions/workflows/ci-standard.yml/badge.svg" alt="CI Standard"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
</p>

<p align="center">
  <strong>A unified platform for golf swing analysis across multiple physics engines and modeling approaches</strong>
</p>

---

## Overview

UpstreamDrift (formerly Golf Modeling Suite) consolidates multiple golf swing modeling implementations into a single, cohesive platform. This repository provides comprehensive biomechanical analysis capabilities through:

> **For AI agents and contributors:** Before writing new code, read [`AGENTS.md`](AGENTS.md) for a directory of shared infrastructure and a discovery workflow to avoid duplicating existing work.

- **Tiered Engine Support**: MuJoCo as the supported default, Drake/Pinocchio for extended cross-engine work, OpenSim/MyoSuite as experimental integrations
- **Multiple Model Complexities**: From 2-DOF educational pendulums to 290-muscle musculoskeletal models
- **Advanced Biomechanics**: Muscle dynamics, inverse kinematics/dynamics, motion capture integration
- **Cross-Engine Validation**: Compare results across different physics engines
- **Professional GUI**: Interactive visualization and analysis tools
- **MATLAB Integration**: Simscape Multibody models for additional analysis

For detailed documentation, please visit the **[Documentation Hub](https://upstream-drift.readthedocs.io)**.
For a focused reviewer walkthrough, start with the
**[golf modeling portfolio demo](docs/portfolio/golf_modeling_demo.md)**.

## Key Features

### Musculoskeletal Modeling

- **MyoSuite Integration**: Experimental muscle-modeling surface for future biomechanics work
- **OpenSim Integration**: Experimental biomechanics validation surface
- **Muscle Dynamics**: Force-length-velocity relationships, activation dynamics
- **Research Lineage**: Includes conversions from established OpenSim model sources where integrations remain under active development

### Advanced Analysis

- **Motion Capture**: Load and retarget mocap data (CSV, JSON, C3D) using OpenPose or MediaPipe.
  - **[Motion Pipeline Guide](docs/motion_pipeline/README.md)** — From video to tracked motion in 5 commands
- **Model Explorer**: Interactive browser for Humanoid, Pendulum, and Robotic models.
- **Inverse Kinematics**: Professional IK solver with nullspace optimization
- **Inverse Dynamics**: Complete torque computation with force decomposition
- **Kinematic Forces**: Coriolis, centrifugal, and gravitational force analysis
- **Trajectory Optimization**: Run trajectory-optimization experiments to compare candidate swing objectives for speed, accuracy, or efficiency

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

The recommended entry point is the **web UI**:

```bash
python3 launch_golf_suite.py
```

This starts the local API server (default port `8000`) and opens the React UI
in your default browser.

### Other entry points

| Command | What it launches |
|---------|------------------|
| `python3 launch_golf_suite.py` | Web UI (recommended) |
| `python3 launch_golf_suite.py --classic` | Classic PyQt6 desktop launcher |
| `python3 launch_golf_suite.py --api-only` | API server without auto-opening a UI |
| `python3 launch_golf_suite.py --engine <name>` | Direct engine launch (legacy) |
| `python3 -m src.tools.pose_studio` | Pose Studio standalone |

Additional flags: `--port <N>` to override the API port, `--no-browser` to skip
auto-opening the browser. The classic PyQt6 launcher remains supported as a
fallback.

**Hiring Manager or Reviewer?** See the [Golf Modeling Portfolio Demo](docs/portfolio/golf_modeling_demo.md) for a focused, reproducible showcase of the physics capabilities.

### Prerequisites

- **Python** 3.11 or 3.12 for the supported pip and lockfile workflow
- **Git** with Git LFS
- **MATLAB** R2023a+ with Simulink and Simscape Multibody (optional, for MATLAB models)
  See the canonical
  **[production artifact and compatibility matrix](docs/operations/production-readiness.md)**
  for supported Python, OS, engine tier, and hardware combinations. Git LFS is
  required for model assets; MATLAB/Simscape models are research references and
  are not production artifacts.

### Installation

**Recommended: Pip** (canonical dependency source: `pyproject.toml`)

```bash
git clone https://github.com/D-sorganization/UpstreamDrift.git
cd UpstreamDrift
git lfs install && git lfs pull

# Install the supported default surface
pip install -e ".[dev]"

# Verify installation
python scripts/verify_installation.py
```

**Conda convenience wrapper**

```bash
conda env create -f environment.yml
conda activate upstream-drift
```

`environment.yml` is generated from `pyproject.toml`; edit Python dependencies
in `pyproject.toml` and run `make sync-deps`.

**Light Installation** (for UI development without heavy physics engines)

```bash
pip install -e .
export GOLF_USE_MOCK_ENGINE=1
```

**Troubleshooting**: See [docs/troubleshooting/installation.md](docs/troubleshooting/installation.md) for common issues.

### Rust Kernel Quickstart

Rust development now works from a clean `UpstreamDrift` clone. The shared
`tools-core` crate is fetched automatically from a pinned `D-sorganization/Tools`
git revision, so you do not need a sibling `../Tools` checkout just to run the
Rust build or Python bindings workflow.

```bash
cargo build

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip maturin

cd rust_core/upstream-physics
python -m maturin develop --features python
python -c "import upstream_physics; print(upstream_physics.IntegratorConfig())"
```

If you are also iterating on local cross-repository Python integrations from
`D-sorganization/Tools`, use `scripts/setup_tools_workspace.sh` to wire the
optional sibling workspace and `PYTHONPATH` helpers.

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
make format    # Format code with Ruff
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

- Experimental integration interface for biomechanics validation
- Not part of the required PR CI contract today
- Use when explicitly working on biomechanics integration tasks
- **See**: [src/engines/physics_engines/opensim/README.md](src/engines/physics_engines/opensim/README.md)

### MyoSuite (Experimental Muscle Modeling)

- Experimental integration interface for future muscle-modeling work
- Not part of the required PR CI contract today
- Use when explicitly working on biomechanics integration tasks
- **See**: [src/engines/physics_engines/myosuite/README.md](src/engines/physics_engines/myosuite/README.md)

**See [Engine Selection Guide](docs/engine_selection_guide.md) for detailed comparison and use cases.**
**See [Supported Engine Tiers](docs/engines/support_tiers.md) for the support and validation contract.**

## Documentation

- **[Project Map](docs/PROJECT_MAP.md)**: Complete guide to every feature, module, and integration in the platform
- **[User Guide](docs/user_guide/README.md)**: Installation, running simulations, and using the GUI
- **[Character Builder Quickstart](docs/user_guide/character_builder_quickstart.md)**: Generate humanoid URDFs in 5 minutes
- **[Engines](docs/engines/README.md)**: Detailed engine documentation and comparison
- **[Supported Engine Tiers](docs/engines/support_tiers.md)**: Install profiles and CI coverage expectations
- **[Development](docs/development/README.md)**: Contributing, architecture, and testing
- **[API Reference](docs/api/README.md)**: Code documentation and interfaces
- **[Plans & Roadmap](docs/plans/README.md)**: Implementation plans and future development
- **[Assessments](docs/assessments/README.md)**: Project reviews and implementation summaries
- **[Technical Docs](docs/technical/README.md)**: Engine reports and control strategies

### Recent Integration Guides

- **[MyoSuite Integration](docs/development/MYOSUITE_INTEGRATION.md)** - Biomechanics features (January 2026)
- **[OpenSim Integration](docs/development/OPENSIM_INTEGRATION.md)** - Musculoskeletal modeling (January 2026)

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
- [Testing Guide](docs/testing/testing-guide.md)

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
