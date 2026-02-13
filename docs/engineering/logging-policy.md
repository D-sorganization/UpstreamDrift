# Logging Policy

## Scope

Applies to production Python modules under:

- `src/api`
- `src/shared/python`
- `src/robotics`
- `src/launchers`
- `src/tools`

## Rules

1. Do not use `print()` in production code paths.
2. Use module loggers: `logger = get_logger(__name__)`.
3. Use level conventions:

- `debug`: internal diagnostics and high-volume traces
- `info`: lifecycle or business milestones
- `warning`: recoverable anomalies or degraded behavior
- `error`: failed operations that impact caller behavior
- `critical`: unrecoverable conditions requiring intervention

4. Include relevant context as structured fields in the message arguments.
5. Log once at the boundary where failure is handled to avoid duplicate noise.

## Examples

```python
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

logger.info("engine loaded", engine=engine_name)
logger.warning("probe unavailable", engine=engine_name, reason=reason)
logger.error("simulation failed", task_id=task_id, exc_info=True)
```

## Enforcement

CI enforces:

- `scripts/check_no_print_calls.py` for net-new `print()` usage in changed production files.
