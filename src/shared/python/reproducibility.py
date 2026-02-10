"""Backward compatibility shim - module moved to data_io.reproducibility."""

import sys as _sys

from .data_io import reproducibility as _real_module  # noqa: E402
from .data_io.reproducibility import (  # noqa: F401
    DEFAULT_SEED,
    MAX_SEED,
    ensure_reproducibility,
    get_rng,
    log_execution_time,
    logger,
    set_seeds,
)

_sys.modules[__name__] = _real_module
