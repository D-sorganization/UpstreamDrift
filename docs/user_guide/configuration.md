# Configuration

The Golf Modeling Suite is designed to be highly configurable. Parameters can be adjusted programmatically via the `PhysicsParameterRegistry` or through configuration files.

## Physics Parameters

The core physics parameters are managed by a central registry found in `shared/python/physics_parameters.py`.

### Default Parameters

The suite comes with reasonable defaults based on:
- **USGA Rules of Golf**: For ball mass and size.
- **NIST Standards**: For gravity and physical constants.
- **Biomechanics Literature**: For human swing characteristics.

### Modifying Parameters Programmatically

You can modify parameters in your scripts before running a simulation:

```python
from shared.python.physics_parameters import get_registry

registry = get_registry()

# Change club mass
registry.set("CLUB_MASS", 0.350)  # kg

# Verify value
param = registry.get("CLUB_MASS")
print(f"New Club Mass: {param.value} {param.unit}")
```

### Parameter Validation

The registry enforces safety checks:
- **Type Checking**: Ensures numeric values for physical quantities.
- **Range Checking**: Prevents unrealistic values (e.g., negative mass).
- **Constants**: Prevents modification of fixed physical laws or strict rules (e.g., Gravity).

## Engine Configuration

Each engine has its specific configuration assets located in `engines/physics_engines/<engine_name>/`.

- **MuJoCo**: `assets/*.xml` (MJCF files)
- **Drake**: `assets/*.sdf` or `*.urdf`
- **Pinocchio**: `models/*.urdf`

To use a custom model, place your asset file in the respective directory and reference it by name when loading the engine.
