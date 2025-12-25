# Golf Modeling Suite 🤖

**A unified platform for golf swing analysis across multiple physics engines and modeling approaches**

![GolfingRobot](GolfingRobot.png)

> ⚠️ **BETA STATUS**: The codebase migration is complete (Status: 98% Complete / Validation). See [Migration Status](docs/plans/migration_status.md) for details.

## Overview

The Golf Modeling Suite consolidates multiple golf swing modeling implementations into a single, cohesive platform. This repository combines MATLAB Simscape multibody models with Python-based physics engines (MuJoCo, Drake, Pinocchio) to provide comprehensive golf swing analysis capabilities.

For detailed documentation, please visit the **[Documentation Hub](docs/README.md)**.

## Quick Start

### Prerequisites
- **Python** 3.10+ (for physics engines)
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
- **[Engines](docs/engines/README.md)**: Details on MuJoCo, Drake, Pinocchio, and Simscape models.
- **[Development](docs/development/README.md)**: Contributing, architecture, and testing.
- **[API Reference](docs/api/README.md)**: Code documentation.

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
