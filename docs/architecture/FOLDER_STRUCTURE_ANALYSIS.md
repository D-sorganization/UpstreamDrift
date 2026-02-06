# UpstreamDrift Folder Structure Analysis

## Overview

UpstreamDrift is a biomechanical golf simulation platform that integrates multiple physics engines. The codebase follows a **hybrid organizational approach** combining domain-driven design with layer-based separation.

## Top-Level Directory Structure

```
UpstreamDrift/
├── archive/              # Deprecated/archived code
├── assets/               # Branding and static assets
├── data/                 # Data files (motion capture, models)
├── docs/                 # Comprehensive documentation
├── examples/             # Tutorial and example scripts
├── installer/            # Platform-specific installers
├── issues/               # Issue tracking files
├── output/               # Simulation output directory
├── reports/              # Security and audit reports
├── scripts/              # Utility and maintenance scripts
├── shared/               # Shared models and data (legacy location)
├── src/                  # Main source code
├── tests/                # Test suite
└── ui/                   # Web-based frontend (React/TypeScript)
```

## Organizational Patterns

### 1. Domain-Driven Organization (Primary)

The codebase organizes code by **physics engine domain** within `src/engines/`:

```
src/engines/
├── physics_engines/          # Core simulation engines
│   ├── mujoco/              # MuJoCo implementation
│   ├── drake/               # Drake implementation
│   ├── pinocchio/           # Pinocchio implementation
│   ├── opensim/             # OpenSim integration
│   └── myosuite/            # MyoSuite muscle models
├── Simscape_Multibody_Models/ # MATLAB/Simulink models
└── pendulum_models/          # Simplified educational models
```

Each engine directory is self-contained with:

- `python/` - Python implementation
- `python/src/` - Core engine code
- `python/tests/` - Engine-specific tests
- `README.md` - Engine documentation

### 2. Layer-Based Organization (Secondary)

Within `src/`, code is also organized by architectural layer:

```
src/
├── api/                  # REST API layer (FastAPI)
│   ├── auth/            # Authentication
│   ├── middleware/      # Request processing
│   ├── models/          # Request/response schemas
│   ├── routes/          # API endpoints
│   ├── services/        # Business logic
│   └── utils/           # API utilities
├── config/              # Configuration files
├── launchers/           # Application entry points
├── shared/              # Shared libraries
│   ├── python/          # Shared Python modules
│   │   ├── ai/          # AI/ML adapters and tools
│   │   ├── analysis/    # Analysis utilities
│   │   ├── optimization/# Optimization algorithms
│   │   ├── plotting/    # Visualization
│   │   └── ui/          # UI components
│   └── models/          # Shared model files (URDF, OpenSim)
└── tools/               # Development tools
    ├── model_generation/# URDF/model generation
    ├── model_explorer/  # Interactive model browser
    └── humanoid_character_builder/
```

### 3. Feature-Based Organization (UI)

The frontend follows React conventions:

```
ui/
├── public/               # Static assets
└── src/
    ├── api/             # API client
    ├── assets/          # Images, icons
    ├── components/      # React components
    │   ├── simulation/  # Simulation widgets
    │   └── visualization/ # 3D viewers
    ├── pages/           # Route pages
    └── test/            # Frontend tests
```

## Test Organization

Tests follow a **type-based hierarchy**:

```
tests/
├── unit/                 # Unit tests by module
│   ├── ai/              # AI module tests
│   └── engines/         # Engine-specific unit tests
├── integration/          # Integration tests
│   └── isolated/        # Isolated integration tests
├── acceptance/           # End-to-end acceptance tests
├── analytical/           # Mathematical validation tests
├── benchmarks/           # Performance benchmarks
├── cross_engine/         # Multi-engine comparison tests
├── fixtures/             # Test fixtures and models
├── headless/             # Headless GUI tests
├── physics_validation/   # Physics accuracy tests
├── security/             # Security-focused tests
└── *.py                  # Root-level test files
```

## Documentation Structure

Extensive documentation organized by audience and purpose:

```
docs/
├── user_guide/           # End-user documentation
├── development/          # Developer guides
├── api/                  # API reference
├── architecture/         # System design
├── engines/              # Engine-specific docs
├── technical/            # Technical reports
├── plans/                # Implementation roadmaps
├── assessments/          # Code reviews and audits
├── tutorials/            # Learning resources
├── troubleshooting/      # Problem resolution
├── testing/              # Testing documentation
└── sphinx/               # Generated API docs
```

## Configuration Files

Root-level configuration:

| File                      | Purpose                                    |
| ------------------------- | ------------------------------------------ |
| `pyproject.toml`          | Python package configuration (hatch build) |
| `environment.yml`         | Conda environment definition               |
| `setup.py`                | Legacy pip installation                    |
| `.pre-commit-config.yaml` | Pre-commit hooks                           |
| `mypy.ini`                | Type checking configuration                |
| `Makefile`                | Development task automation                |

## CI/CD and Automation

Extensive GitHub Actions automation in `.github/workflows/`:

- **CI Pipelines**: `ci-standard.yml`, `ci-fast-tests.yml`
- **Jules Bot Workflows**: 30+ automated workflows for code quality, documentation, issue management
- **Security**: `docker-security-scan.yml`, `critical-files-guard.yml`
- **Maintenance**: `stale-cleanup.yml`, `nightly-cross-engine.yml`

## Key Conventions

### 1. Shared Code Pattern

Common utilities live in `src/shared/python/` and are imported across engines.

### 2. Model Files Location

- OpenSim models: `src/shared/models/opensim/opensim-models/`
- MyoSuite models: `src/shared/models/myosuite/`
- URDF models: `src/shared/urdf/`

### 3. Entry Points

- Primary launcher: `launch_golf_suite.py`
- Engine-specific launchers: `src/launchers/`
- CLI entry: `upstream-drift` command

### 4. Namespace Pattern

Python packages use `src/` as the root:

- `src.api` - API modules
- `src.engines` - Physics engines
- `src.shared` - Shared utilities

## Strengths of This Organization

1. **Clear separation of physics engines** - Each engine is isolated and independently testable
2. **Comprehensive test coverage** - Multiple test types ensure quality
3. **Extensive automation** - Jules bot handles routine tasks
4. **Rich documentation** - Multiple documentation types for different audiences
5. **Flexible installation** - Optional dependencies for different use cases

## Areas for Potential Improvement

1. **Duplicate paths** - `shared/` exists at both root and `src/shared/`
2. **Deep nesting** - Some paths are very deep (e.g., Simscape models)
3. **Mixed conventions** - Some engines use `python/src/`, others use flat structure
4. **Test file location** - Some tests are at root of `tests/`, others in subdirectories

## Summary

UpstreamDrift uses a **multi-paradigm organizational approach**:

- **Domain-driven** for physics engine separation
- **Layer-based** for API and shared code
- **Feature-based** for UI components
- **Type-based** for test organization

This hybrid approach effectively manages the complexity of integrating five different physics engines while maintaining a cohesive codebase.
