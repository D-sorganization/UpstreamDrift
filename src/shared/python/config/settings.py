"""Canonical configuration reference for the Golf Modeling Suite.

This module documents the single source of truth for every category of
configuration in the codebase.  It re-exports the most commonly needed
accessors so that call-sites can import from one place.

Configuration Landscape
-----------------------
The project uses several config formats; this table shows what belongs where:

+----------------------------+-------------------------------------+----------+
| Setting category           | Canonical location                  | Format   |
+============================+=====================================+==========+
| API secret key             | env: GOLF_API_SECRET_KEY            | env var  |
| Fallback secret key        | env: SECRET_KEY                     | env var  |
| Deployment environment     | env: ENVIRONMENT                    | env var  |
| Admin password             | env: GOLF_ADMIN_PASSWORD            | env var  |
| Database URL               | env: DATABASE_URL                   | env var  |
| API host / port            | env: GOLF_API_HOST / GOLF_API_PORT  | env var  |
| Auth disabled flag         | env: GOLF_AUTH_DISABLED             | env var  |
| Headless mode              | env: HEADLESS                       | env var  |
| Log level                  | env: LOG_LEVEL                      | env var  |
+----------------------------+-------------------------------------+----------+
| CORS origins (defaults)    | src/config/interim_config.yaml      | YAML     |
| Trusted hosts (defaults)   | src/config/interim_config.yaml      | YAML     |
| Rate-limit defaults        | src/config/interim_config.yaml      | YAML     |
| Quota tiers                | src/config/interim_config.yaml      | YAML     |
| Simulation engine order    | src/config/interim_config.yaml      | YAML     |
| Video analysis defaults    | src/config/interim_config.yaml      | YAML     |
+----------------------------+-------------------------------------+----------+
| Tool / lint / test config  | pyproject.toml                      | TOML     |
| Coverage thresholds        | pyproject.toml                      | TOML     |
+----------------------------+-------------------------------------+----------+
| Physical constants (SI)    | shared/python/core/physics_constants| Python   |
| Simulation config defaults | shared/python/config/               |          |
|                            |   configuration_manager.py          | Python   |
| File-size budgets          | scripts/config/file_size_budget.json| JSON     |
| Module-size baselines      | scripts/config/                     |          |
|                            |   module_size_budget_baseline.json  | JSON     |
+----------------------------+-------------------------------------+----------+

Resolution precedence for runtime settings
------------------------------------------
1. Environment variable (always wins if set and non-empty)
2. YAML default from ``src/config/interim_config.yaml`` (documents intent;
   not loaded automatically — callers that need YAML values must load the
   file explicitly via PyYAML)
3. Hard-coded default in the accessor function

Note: ``src/config/interim_config.yaml`` contains ``auth.secret_key:
"${GOLF_API_SECRET_KEY}"`` as documentation only.  The variable
interpolation is **not** performed automatically.  The actual secret key is
always read from the environment by
``src.shared.python.config.environment.get_secret_key``.

Startup validation
------------------
``src.shared.python.security.env_validator.validate_environment`` must be
called during application startup (it is called in ``src/api/server.py``
inside the FastAPI lifespan handler).  It validates required env vars,
checks secret-key strength, and logs a full report.

Accessor quick-reference
------------------------
All env-var accessors live in ``src.shared.python.config.environment``.
Import from there (or from this module for convenience):

    from src.shared.python.config.settings import (
        get_secret_key,
        get_environment,
        is_production,
        get_database_url,
        get_api_host,
        get_api_port,
        get_log_level,
        get_setting,
        load_settings,
        save_settings,
    )
"""

from __future__ import annotations

import logging
from typing import Any

from src.shared.python.config.environment import (
    get_admin_password,
    get_api_host,
    get_api_port,
    get_database_url,
    get_dbc_level,
    get_display,
    get_env_bool,
    get_env_float,
    get_env_int,
    get_env_list,
    get_environment,
    get_golf_port,
    get_golf_suite_mode,
    get_golf_ui_dist,
    get_log_level,
    get_secret_key,
    is_auth_disabled,
    is_browser_suppressed,
    is_development,
    is_docker,
    is_headless,
    is_production,
    is_wsl,
    require_env,
)

_logger = logging.getLogger(__name__)

# In-process settings overlay: persists only for the lifetime of the process.
# Use environment variables or persistent config files for cross-process state.
_settings_store: dict[str, Any] = {}


def get_setting(key: str, default: Any = None) -> Any:
    """Return the value for *key* from the in-process settings store.

    Preconditions:
        key must be a non-empty string.

    Args:
        key: The setting name.
        default: Value to return when *key* is absent. Defaults to ``None``.

    Returns:
        The stored value, or *default* if *key* is not present.

    Raises:
        TypeError: If *key* is not a string.

    Example:
        >>> get_setting("simulation_timeout", default=60)
        60
    """
    if not isinstance(key, str):
        raise TypeError(f"key must be a str, got {type(key).__name__!r}")
    return _settings_store.get(key, default)


def load_settings() -> dict[str, Any]:
    """Return a snapshot of the current in-process settings store.

    Returns:
        A shallow copy of the settings dict so callers cannot mutate internal
        state directly.

    Example:
        >>> settings = load_settings()
        >>> isinstance(settings, dict)
        True
    """
    return dict(_settings_store)


def save_settings(data: dict[str, Any]) -> None:
    """Merge *data* into the in-process settings store.

    Existing keys are overwritten; keys not in *data* are preserved.

    Preconditions:
        data must be a dict.

    Args:
        data: Mapping of setting names to values.

    Raises:
        TypeError: If *data* is not a dict.

    Example:
        >>> save_settings({"simulation_timeout": 120})
    """
    if not isinstance(data, dict):
        raise TypeError(f"data must be a dict, got {type(data).__name__!r}")
    _settings_store.update(data)
    _logger.debug("Settings updated: %d key(s) merged", len(data))


__all__ = [
    "get_admin_password",
    "get_api_host",
    "get_api_port",
    "get_database_url",
    "get_dbc_level",
    "get_display",
    "get_environment",
    "get_env_bool",
    "get_env_float",
    "get_env_int",
    "get_env_list",
    "get_golf_port",
    "get_golf_suite_mode",
    "get_golf_ui_dist",
    "get_log_level",
    "get_secret_key",
    "get_setting",
    "is_auth_disabled",
    "is_browser_suppressed",
    "is_development",
    "is_docker",
    "is_headless",
    "is_production",
    "is_wsl",
    "load_settings",
    "require_env",
    "save_settings",
]
