"""Backward compatibility shim - module moved to security.env_validator."""

import sys as _sys

from .security import env_validator as _real_module  # noqa: E402
from .security.env_validator import (  # noqa: F401
    APIKeyValidationResults,
    DatabaseKeyValidationResults,
    EnvironmentValidationResults,
    generate_secure_key_command,
    logger,
    print_validation_report,
    validate_api_security,
    validate_database_config,
    validate_environment,
    validate_production_checklist,
    validate_secret_key_strength,
)

_sys.modules[__name__] = _real_module
