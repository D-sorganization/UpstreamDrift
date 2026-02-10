"""Backward compatibility shim - module moved to config.environment."""

import sys as _sys

from .config import environment as _real_module  # noqa: E402
from .config.environment import (  # noqa: F401
    EnvironmentError,
    T,
    get_admin_password,
    get_api_host,
    get_api_port,
    get_database_url,
    get_env,
    get_env_bool,
    get_env_float,
    get_env_int,
    get_env_list,
    get_environment,
    get_log_level,
    get_secret_key,
    is_development,
    is_docker,
    is_production,
    is_wsl,
    require_env,
)

_sys.modules[__name__] = _real_module
