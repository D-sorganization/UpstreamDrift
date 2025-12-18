# Golf Modeling Suite

**A unified platform for golf swing analysis across multiple physics engines and modeling approaches**

## Overview

The Golf Modeling Suite consolidates multiple golf swing modeling implementations into a single, cohesive platform. This repository combines MATLAB Simscape multibody models with Python-based physics engines to provide comprehensive golf swing analysis capabilities.

## Quick Start

### Prerequisites
- **MATLAB** R2023a+ with Simulink and Simscape Multibody (for MATLAB models)
- **Python** 3.11+ (for physics engines)
- **Docker** (optional, for containerized physics engines)
- **Git** with Git LFS

### Installation
```bash
git clone https://github.com/D-sorganization/Golf_Modeling_Suite.git
cd Golf_Modeling_Suite
git lfs install
git lfs pull
pip install -r shared/python/requirements.txt
```

### Launching Models
```bash
# Unified launcher (recommended)
python launchers/golf_launcher.py

# Local Python launcher  
python launchers/golf_suite_launcher.py
```

## Repository Structure

```
Golf_Modeling_Suite/
├── launchers/                    # Unified launch applications
├── engines/
│   ├── matlab_simscape/         # MATLAB/Simulink models
│   ├── physics_engines/         # Python physics engines (MuJoCo, Drake, Pinocchio)
│   └── pendulum_models/         # Simplified pendulum models
├── shared/                      # Common utilities and resources
└── tools/                      # Analysis and development tools
```

## Migration Status

This repository is currently under active migration from separate repositories. See [MIGRATION_STATUS.md](MIGRATION_STATUS.md) for detailed progress.

## License

MIT License - See [LICENSE](LICENSE) for details.