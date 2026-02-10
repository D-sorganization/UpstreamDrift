"""Backward compatibility shim - module moved to validation_pkg.validation_utils."""

import sys as _sys

from .validation_pkg import validation_utils as _real_module  # noqa: E402
from .validation_pkg.validation_utils import (  # noqa: F401
    logger,
    validate_all,
    validate_array_dimensions,
    validate_array_length,
    validate_array_shape,
    validate_dict_keys,
    validate_directory_exists,
    validate_extension,
    validate_file_exists,
    validate_not_none,
    validate_numeric,
    validate_positive,
    validate_range,
    validate_type,
)

_sys.modules[__name__] = _real_module
