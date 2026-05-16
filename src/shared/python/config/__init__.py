"""Configuration management, environment, and model registry.

Public surface
--------------
The most commonly needed accessors are re-exported here so that call-sites
can import from a single place::

    from src.shared.python.config import get_setting, load_settings, save_settings

For the full catalogue of environment-variable accessors see
``src.shared.python.config.settings``.
"""

from src.shared.python.config.settings import (
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
    get_setting,
    is_auth_disabled,
    is_browser_suppressed,
    is_development,
    is_docker,
    is_headless,
    is_production,
    is_wsl,
    load_settings,
    require_env,
    save_settings,
)

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
