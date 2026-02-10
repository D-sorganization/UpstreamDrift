"""Backward compatibility shim - module moved to data_io.provenance."""

import sys as _sys

from .data_io import provenance as _real_module  # noqa: E402
from .data_io.provenance import (  # noqa: F401
    ProvenanceInfo,
    add_provenance_header_file,
    add_provenance_to_csv,
)

_sys.modules[__name__] = _real_module
