# Golf Modeling Suite 🤖

**A unified platform for golf swing analysis across multiple physics engines and modeling approaches**

![GolfingRobot](GolfingRobot.png)

> ⚠️ **BETA STATUS**: The codebase migration is complete (Status: 98% Complete / Validation). See [Migration Status](docs/plans/migration_status.md) for details.

## Overview

The Golf Modeling Suite consolidates multiple golf swing modeling implementations into a single, cohesive platform. This repository combines MATLAB Simscape multibody models with Python-based physics engines (MuJoCo, Drake, Pinocchio) to provide comprehensive golf swing analysis capabilities.

For detailed documentation, please visit the **[Documentation Hub](docs/README.md)**.

## Quick Start

### Prerequisites
- **Python** 3.11+ (for physics engines)
- **MATLAB** R2023a+ with Simulink and Simscape Multibody (optional, for MATLAB models)
- **Git** with Git LFS

### Installation
```bash
git clone https://github.com/D-sorganization/Golf_Modeling_Suite.git
cd Golf_Modeling_Suite
git lfs install
git lfs pull
pip install -r requirements.txt
```

### Launching Models
```bash
# Unified launcher (recommended)
python launchers/golf_launcher.py

# Local Python launcher  
python launchers/golf_suite_launcher.py
```

## Documentation

- **[User Guide](docs/user_guide/README.md)**: Installation, running simulations, and using the GUI.
- **[Engines](docs/engines/README.md)**: Details on Mu JoCo, Drake, Pinocchio, and Simscape models.
- **[Development](docs/development/README.md)**: Contributing, architecture, and testing.
- **[API Reference](docs/api/README.md)**: Code documentation.
- **[Engine Selection Guide](docs/engine_selection_guide.md)**: Choose the right physics engine for your needs.

## Engine Compatibility

### ⚠️ Important: Biomechanics Features

**MyoSuite biomechanics (Hill-type muscles, grip modeling) are MuJoCo-only**. This includes:
- ✅ Human URDF models with muscles
- ✅ Muscle activation dynamics  
- ✅ Muscle-induced acceleration analysis

**Other engines (Drake, Pinocchio)** support:
- ✅ Kinematics and dynamics
- ✅ Joint-level analysis
- ✅ URDF loading
- ❌ Muscle-level biomechanics (not available)

📖 **See [Engine Selection Guide](docs/engine_selection_guide.md) for detailed comparison and migration strategies**.

## Repository Structure

```
Golf_Modeling_Suite/
├── docs/                        # Comprehensive documentation
├── launchers/                   # Unified launch applications
├── engines/
│   ├── Simscape_Multibody_Models/ # MATLAB/Simulink models
│   ├── physics_engines/         # Python physics engines (MuJoCo, Drake, Pinocchio)
│   └── pendulum_models/         # Simplified pendulum models
├── shared/                      # Common utilities and resources
└── tools/                       # Analysis and development tools
```

## License

MIT License - See [LICENSE](LICENSE) for details.
