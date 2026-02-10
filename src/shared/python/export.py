"""Backward compatibility shim - module moved to data_io.export."""

import sys as _sys

from .data_io import export as _real_module  # noqa: E402
from .data_io.export import (  # noqa: F401
    export_recording_all_formats,
    export_to_c3d,
    export_to_hdf5,
    export_to_matlab,
    get_available_export_formats,
    logger,
)

_sys.modules[__name__] = _real_module
