# Logging Style Guide

This document defines the conventions for logging throughout the UpstreamDrift
codebase. All new code should follow these guidelines; existing code should be
migrated as part of normal maintenance.

## Quick start

```python
from src.shared.python.logging_pkg.logging_config import setup_logging, get_logger

setup_logging()                  # call once at application entry point
logger = get_logger(__name__)    # one logger per module
```

For structured logging via `structlog` (preferred for new code):

```python
from src.shared.python.core import setup_structured_logging, get_logger

setup_structured_logging()
logger = get_logger(__name__)
logger.info("simulation_started", engine="mujoco", step_count=500)
```

## Log levels

| Level      | When to use                                                   | Example                                               |
| ---------- | ------------------------------------------------------------- | ----------------------------------------------------- |
| `DEBUG`    | Detailed diagnostic data useful only during active debugging. | `logger.debug("matrix_shape", rows=3, cols=3)`        |
| `INFO`     | Normal operational milestones and state transitions.          | `logger.info("engine_loaded", engine="drake")`        |
| `WARNING`  | Unexpected but recoverable situations; degraded behaviour.    | `logger.warning("fallback_used", reason="timeout")`   |
| `ERROR`    | A failure that prevents a specific operation from completing. | `logger.error("file_not_found", path="/data/in.csv")` |
| `CRITICAL` | System-wide failure requiring immediate operator attention.   | `logger.critical("database_unreachable")`             |

### Rules of thumb

1. **INFO is the default production level.** Anything that would overwhelm a
   production log stream belongs at `DEBUG`.
2. **Never log secrets.** The `SensitiveDataFilter` will redact known patterns
   (`password`, `api_key`, `secret_token`, ...) but do not rely on it as the
   sole safeguard. Avoid passing credential values to log calls at all.
3. **Use structured key-value pairs** instead of f-strings when using structlog:

   ```python
   # Good
   logger.info("request_completed", duration_ms=42, status=200)

   # Avoid
   logger.info(f"Request completed in 42ms with status 200")
   ```

4. **One logger per module.** Declare `logger = get_logger(__name__)` at
   module scope.
5. **Bind context for request/session scoped work:**
   ```python
   req_logger = logger.bind(request_id=req_id)
   req_logger.info("processing_started")
   ```

## Log rotation

Use `add_rotating_file_handler` when writing to files in long-running
processes:

```python
from src.shared.python.logging_pkg.logging_config import add_rotating_file_handler

add_rotating_file_handler(
    filename="logs/app.log",
    max_bytes=10 * 1024 * 1024,   # 10 MB
    backup_count=5,
)
```

Defaults: **10 MB** per file, **5** backup files retained.

## Sensitive data redaction

The `SensitiveDataFilter` is automatically attached to handlers created via
`setup_logging` and `add_rotating_file_handler`. It redacts values matching
patterns such as `password=...`, `api_key=...`, `access_token=...`.

If you add a handler manually, attach the filter explicitly:

```python
from src.shared.python.logging_pkg.logging_config import SensitiveDataFilter

handler = logging.StreamHandler()
handler.addFilter(SensitiveDataFilter())
```

## JSON output (production)

For production deployments that feed logs into an aggregator (ELK, Datadog,
etc.), enable JSON rendering:

```python
setup_logging(json_output=True, dev_mode=False)
```

Or with structlog directly:

```python
setup_structured_logging(json_output=True, dev_mode=False)
```

## Migration from raw `logging`

When updating a module from raw `logging` to structured logging:

1. Replace `import logging` / `logging.getLogger` with the centralized
   `get_logger`.
2. Convert `logger.info("Did X with %s", value)` to
   `logger.info("did_x", value=value)`.
3. Remove any per-module `basicConfig` calls -- they conflict with the
   centralized setup.
