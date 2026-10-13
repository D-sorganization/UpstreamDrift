"""Stable ``data_io`` entry point for the shared ezc3d native-DLL guard (#12065).

The implementation is owned by Tools (``sidekick.utils.native_dll_guard``,
Tools PR #5581) and arrives through the ``vendor/ud-tools`` pin.  This module
only re-exports it so existing UpstreamDrift import paths stay stable without a
second copy of the logic (DRY).  Do not add behaviour here.

Every ``ezc3d`` import site should go through :func:`import_ezc3d`.
"""

from __future__ import annotations

from src.shared.python.sidekick.utils.native_dll_guard import (
    EZC3D_DLL,
    NativeDllConflictError,
    import_ezc3d,
    loaded_dll_path,
)

__all__ = [
    "EZC3D_DLL",
    "NativeDllConflictError",
    "import_ezc3d",
    "loaded_dll_path",
]
