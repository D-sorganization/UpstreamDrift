"""Backward compatibility shim - module moved to validation_pkg.validation_data."""

import sys as _sys

from .validation_pkg import validation_data as _real_module  # noqa: E402
from .validation_pkg.validation_data import (  # noqa: F401
    ALL_VALIDATION_DATA,
    AMATEUR_AVERAGES,
    PGA_TOUR_2024,
    DataSource,
    ValidationDataPoint,
    get_validation_data_for_club,
    print_validation_summary,
)

_sys.modules[__name__] = _real_module
