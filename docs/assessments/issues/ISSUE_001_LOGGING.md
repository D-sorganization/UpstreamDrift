---
title: "Replace excessive print statements with logging"
labels: ["jules:assessment", "needs-attention", "tech-debt"]
assignee: "jules"
---

## Problem Description
The repository assessment identified **1,346** occurrences of `print()` statements used for debugging and status updates, compared to only 395 uses of the standard `logging` module.

Using `print` for production logging is problematic because:
1.  **No Severity Levels**: Cannot filter by INFO, WARN, ERROR.
2.  **No Context**: Lacks timestamps, module names, and line numbers.
3.  **Hard to Capture**: Difficult to route to log management systems without capturing all stdout.
4.  **Performance**: Blocking I/O on stdout can affect performance in tight loops.

## Affected Areas
- `launchers/golf_launcher.py`
- `shared/python/plotting_core.py`
- `engines/physics_engines/mujoco/python/humanoid_launcher.py`
- And many others.

## Proposed Resolution
1.  Configure the root logger in the entry points (`launchers/*.py`, `api/server.py`).
2.  In each module, replace `print(f"...")` with:
    ```python
    import logging
    logger = logging.getLogger(__name__)
    ...
    logger.info(f"...")
    ```
3.  Remove `T201` ignores from `pyproject.toml` as files are fixed.

## Acceptance Criteria
- `print()` statements are replaced with `logger` calls in key modules.
- `pyproject.toml` has fewer `T201` exclusions.
