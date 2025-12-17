"""Project package init."""

from typing import Any, List

# Define stubs first to avoid MyPy redefinition errors
_C3D_AVAILABLE = False
_C3DDataReader: Any = None
_C3DEvent: Any = None
_C3DMetadata: Any = None
_load_tour_average_reader: Any = None

try:
    from .c3d_reader import (
        C3DDataReader as _C3DDataReader_imported,
    )
    from .c3d_reader import (
        C3DEvent as _C3DEvent_imported,
    )
    from .c3d_reader import (
        C3DMetadata as _C3DMetadata_imported,
    )
    from .c3d_reader import (
        load_tour_average_reader as _load_tour_average_reader_imported,
    )

    _C3D_AVAILABLE = True
    _C3DDataReader = _C3DDataReader_imported
    _C3DEvent = _C3DEvent_imported
    _C3DMetadata = _C3DMetadata_imported
    _load_tour_average_reader = _load_tour_average_reader_imported
except ImportError:
    # ezc3d not available (e.g., Python 3.9)
    # Stubs remain as None - will raise RuntimeError when used
    pass

# Export the names (either imported or None stubs)
C3DDataReader = _C3DDataReader
C3DEvent = _C3DEvent
C3DMetadata = _C3DMetadata
load_tour_average_reader = _load_tour_average_reader

__all__: list[str] = [
    "C3DDataReader",
    "C3DEvent",
    "C3DMetadata",
    "load_tour_average_reader",
]
