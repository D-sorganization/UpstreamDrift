# Unified Golf Biomechanics Platform Integration

## Overview

This PR integrates three existing projects (MuJoCo Golf Model, Pinocchio Golf Model, and Golf_Model reference) into a single, unified platform housed within the Pinocchio_Golf_Model repository. This creates a comprehensive "full-body, physics-grounded, IK/Dynamics-driven Golfer Simulation Toolkit" that supports multiple physics engines and visualization tools.

## Key Features

### 🎯 Canonical Model Specification

- **Single Source of Truth**: YAML-based canonical model specification (`models/spec/golfer_canonical.yaml`)
- **Multi-Backend Export**: URDF and MJCF exporters for generating backend-specific models
- **Complete Model Definition**: Includes segments, joints, constraints, contacts, and named frames

### 🔧 Backend Integration

- **PinocchioBackend**: Full dynamics support (RNEA, ABA, CRBA, Jacobians)
- **MuJoCoBackend**: Forward simulation with contact dynamics
- **PINKBackend**: IK solver foundation (stub implementation ready for expansion)
- **BackendFactory**: Unified interface for creating backend instances

### 🎨 Visualization

- **MeshCatViewer**: Browser-based visualization wrapper
- **GeppettoViewer**: Desktop visualization wrapper
- **Unified GUI**: PySide6 application with tabbed interface (Model Viewer, IK, Dynamics, Counterfactuals, ML, Settings)

### 📊 MATLAB Integration

- **Simscape Extraction**: MATLAB script to extract parameters from Simscape models
- **Data Import**: Python utilities for loading `.mat` and `.c3d` files

### 🧪 Testing & Quality

- **Comprehensive Test Structure**: Unit, integration, validation, and performance test directories
- **Code Quality**: Unified ruff.toml, mypy.ini, .pre-commit-config.yaml with strictest settings
- **Documentation**: Sphinx API documentation foundation, quick start guide, updated README

## Repository Consolidation

### Files Added

- **Data**: Copied from Golf_Model (Rob Neal data, Gears Tour Average)
- **Models**: MATLAB Simulink model copied to `models/matlab/`
- **MuJoCo Code**: Migrated to `python/mujoco_golf_pendulum/`
- **Pinocchio Code**: Reorganized to `python/pinocchio_golf/`

### Core Library Structure (`python/dtack/`)

```
dtack/
├── backends/      # Physics engine wrappers
├── sim/          # Simulation modules
├── viz/          # Visualization wrappers
├── ik/           # Inverse kinematics
├── constraints/  # Loop closures
├── ml/           # Machine learning
├── gui/          # GUI application
└── utils/        # Utilities (exporters, importers)
```

## Breaking Changes

- Existing Pinocchio GUI files moved from `python/` to `python/pinocchio_golf/`
  - `pinnochio_GUI.py` → `pinocchio_golf/gui.py`
  - `coppelia_pinnochio_meshcat_bridge.py` → `pinocchio_golf/coppelia_bridge.py`
  - `pinnochio_polynomial_torque_fitter.py` → `pinocchio_golf/torque_fitting.py`

## Testing

- ✅ Integration test foundations created
- ✅ Validation test foundations created
- ✅ Test structure verified (unit/, integration/, validation/, performance/)

## Documentation

- ✅ Updated README.md with installation and quick start
- ✅ Created quick start guide (`docs/user_guides/QUICK_START.md`)
- ✅ Sphinx API documentation foundation (`docs/api/`)
- ✅ All code follows strict type hints and documentation standards

## Commits

This PR includes 10 commits:

1. `feat: Add backend wrappers for Pinocchio, MuJoCo, and PINK`
2. `feat: Add URDF exporter and dtack module structure`
3. `feat: Add MJCF exporter and visualization wrappers`
4. `feat: Add MATLAB integration utilities`
5. `test: Add integration and validation test foundations`
6. `feat: Add unified GUI and update README`
7. `docs: Add quick start guide`
8. `refactor: Move Pinocchio files to pinocchio_golf/ and update configs`
9. `feat: Add canonical YAML model specification`
10. `docs: Add Sphinx API documentation foundation`

## Next Steps

After merge:

1. Fill in stub implementations (PINK IK solver, counterfactuals, ML modules)
2. Add comprehensive unit tests for each module
3. Generate full API documentation with Sphinx
4. Connect GUI to backend functionality
5. Validate model exporters with real models

## Related

- Consolidates code from MuJoCo_Golf_Swing_Model
- References Golf_Model for data and model structure
- Follows architecture outlined in `docs/Pinocchio_Project_Outline.md`
