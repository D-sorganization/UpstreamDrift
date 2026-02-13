# Dependency Direction Rules

The architecture fitness gate is defined in `scripts/config/dependency_direction_rules.json` and enforced by `scripts/check_dependency_direction.py`.

## Current boundaries

- `src/shared/python` must not import from engine implementation layers (`src.engines`, `src.robotics`).
- `src/api` must not import from `src.launchers` directly; use service abstractions.

## Allowed patterns

```python
# Allowed: type-only import via TYPE_CHECKING
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.engines.some_engine import EngineImpl
```

```python
# Allowed: lazy runtime import inside function body

def make_launcher_service():
    from src.api.services.launcher_service import LauncherService
    return LauncherService(...)
```

## Violation examples

```python
# Not allowed in shared layer
from src.engines.physics_engines.mujoco import mujoco_physics_engine
```

```python
# Not allowed in API route/module-level scope
from src.launchers.launcher_process_manager import ProcessManager
```

## Exception process

Use `exceptions` in `scripts/config/dependency_direction_rules.json` only when needed, and include:

- `path`
- `owner`
- `reason`
- `expires_on`

Expired exceptions fail CI.
