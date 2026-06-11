"""Pytest configuration for MuJoCo physics engine tests.

Path configuration is centralized in pyproject.toml [tool.pytest.ini_options].
This follows DRY principles from The Pragmatic Programmer.
Optional runtime dependencies must be stubbed by scoped per-test fixtures, not
by module-level sys.modules mutation during collection.
"""
