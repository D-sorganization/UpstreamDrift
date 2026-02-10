"""Backward compatibility shim - module moved to data_io.data_utils."""

import sys as _sys

from .data_io import data_utils as _real_module  # noqa: E402
from .data_io.data_utils import (  # noqa: F401
    DataLoader,
    convert_to_dataframe,
    load_c3d_data,
    load_csv_data,
    load_json_data,
    load_numpy_data,
    logger,
    resample_data,
    save_csv_data,
    save_json_data,
    save_numpy_data,
)

_sys.modules[__name__] = _real_module
