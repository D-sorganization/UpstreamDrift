"""Backward compatibility shim - module moved to data_io.common_utils."""

import sys as _sys

from .data_io import common_utils as _real_module  # noqa: E402
from .data_io.common_utils import (  # noqa: F401
    CONVERSION_FACTORS,
    DataFormatError,
    EngineNotFoundError,
    GolfModelingError,
    convert_units,
    ensure_output_dir,
    get_logger,
    get_shared_urdf_path,
    load_golf_data,
    normalize_z_score,
    plot_joint_trajectories,
    save_golf_data,
    setup_logging,
    setup_structured_logging,
    standardize_joint_angles,
)

_sys.modules[__name__] = _real_module
