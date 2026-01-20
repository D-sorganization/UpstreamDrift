# Assessment L: Logging

## Grade: 8/10

## Summary
Logging is implemented using the standard library `logging` module, with proper configuration at the application entry point.

## Strengths
- **Usage**: Modules use `logging.getLogger(__name__)` to allow granular control.
- **Configuration**: `api/server.py` sets up basic logging.
- **Linting**: Tools are configured to flag `print` statements (`flake8-print`), enforcing the use of proper logging.

## Weaknesses
- **Consistency**: `structlog` is listed in dependencies, but the sampled code (`api/server.py`, `signal_processing.py`) uses the standard `logging` library. This suggests a potential mix of logging styles or underutilization of structured logging capabilities.

## Recommendations
- Standardize on `structlog` for application-level logging to produce machine-readable (JSON) logs, which are easier to query in production environments.
