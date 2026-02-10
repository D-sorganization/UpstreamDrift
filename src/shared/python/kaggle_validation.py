"""Backward compatibility shim - module moved to validation_pkg.kaggle_validation."""

import sys as _sys

from .validation_pkg import kaggle_validation as _real_module  # noqa: E402
from .validation_pkg.kaggle_validation import (  # noqa: F401
    ShotRecord,
    compare_all_models_to_dataset,
    get_clean_shots,
    get_dataset_statistics,
    load_kaggle_dataset,
    logger,
    print_validation_report,
    validate_model_against_dataset,
)

_sys.modules[__name__] = _real_module
