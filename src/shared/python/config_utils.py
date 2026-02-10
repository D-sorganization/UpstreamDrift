"""Backward compatibility shim - module moved to config.config_utils."""

import sys as _sys

from .config import config_utils as _real_module  # noqa: E402
from .config.config_utils import (  # noqa: F401
    ConfigLoader,
    T,
    load_json_config,
    load_yaml_config,
    logger,
    merge_configs,
    save_json_config,
    save_yaml_config,
    validate_config,
)

_sys.modules[__name__] = _real_module
