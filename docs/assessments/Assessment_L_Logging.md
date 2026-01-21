# Assessment L: Logging

## Grade: 9/10

## Summary
Logging is excellent. The `core.py` module defines `setup_structured_logging` using `structlog`, providing JSON output for production and readable output for dev.

## Strengths
- **Structured Logging**: Uses `structlog` for machine-readable logs.
- **Contextual Info**: Logs include timestamps, log levels, and can include arbitrary context.
- **Centralized Config**: A single setup function ensures consistency.

## Weaknesses
- **Adoption**: Need to ensure all modules use `get_logger` and not `print` or raw `logging`. `ruff` checks for `print` statements, which is good.

## Recommendations
1. **Audit Logs**: Periodically review logs to ensure they contain useful debugging info without being too verbose.
