# Assessment: Logging (Category L)

## Grade: 4/10

## Analysis
Logging is a significant weakness in the current codebase.
- **Print Statements**: There are **1,346** occurrences of `print()` in the codebase, compared to only **395** calls to `logging.`.
- **Configuration**: While `api/server.py` and `golf_launcher.py` configure logging, many modules rely on `print` for debugging and status updates.
- **Risk**: `print` statements clutter stdout, cannot be easily filtered or piped to files in production, and lack timestamp/severity context.
- **CI Ignores**: `pyproject.toml` has numerous `T201` (print found) ignores to make the linter pass, masking the debt.

## Recommendations
1. **Migration**:Systematically replace `print()` with `logger.info()`, `logger.debug()`, etc., across the codebase.
2. **Standardization**: Ensure every module does `logger = logging.getLogger(__name__)`.
3. **Structured Logging**: Consider using `structlog` (which is in dependencies) more consistently for machine-readable logs.
