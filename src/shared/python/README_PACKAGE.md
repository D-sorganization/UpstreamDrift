# upstream-drift-shared

Shared Python modules for the Golf Modeling Suite (UpstreamDrift).

This package is the **single source of truth** for modules consumed by:
- [UpstreamDrift](https://github.com/D-sorganization/Golf_Modeling_Suite) (source)
- [Tools](https://github.com/D-sorganization/Tools) (consumer)
- [Gasification_Model](https://github.com/D-sorganization/Gasification_Model) (consumer)

## Installation

```bash
# Core (logging, error handling, validation)
pip install upstream-drift-shared

# With theme support (requires PyQt6)
pip install upstream-drift-shared[theme]

# With plotting support (requires matplotlib)
pip install upstream-drift-shared[plotting]

# Everything
pip install upstream-drift-shared[all]
```

## Module Ownership

| Module | Description | Owner |
|--------|-------------|-------|
| `core/` | Exceptions, constants, contracts, structured logging | UpstreamDrift |
| `theme/` | PyQt6 theme system (colors, typography, dialogs) | UpstreamDrift |
| `plotting/` | Matplotlib plot theming and animation | UpstreamDrift |
| `logging_pkg/` | Centralized logging configuration | UpstreamDrift |
| `validation_pkg/` | Input validation utilities | UpstreamDrift |
| `security/` | Security utilities and subprocess safety | UpstreamDrift |
| `engine_core/` | Physics engine interfaces and registry | UpstreamDrift |
