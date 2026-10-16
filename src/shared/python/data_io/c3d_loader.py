"""Single lazy route to the C3D reading stack (#12883).

Importing ``sidekick.lab.bio`` imports ``ezc3d`` at module load.  On Windows the
Python ``ezc3d`` extension and the native OpenSim SDK ship ABI-incompatible
``ezc3d.dll`` builds, so whichever loads first owns the DLL name.  Importing
the C3D stack at launcher startup therefore breaks OpenSim for the whole
session.  Every UpstreamDrift call site that needs the C3D reader or its model
types goes through these helpers, at call time, so ``ezc3d`` is only loaded when
a C3D file is actually read.

This module imports only the standard library.
"""

from __future__ import annotations

from typing import Any

__all__ = ["c3d_reader_class", "marker_set_types"]

_BIO_PACKAGE = "src.shared.python.sidekick.lab.bio"


def c3d_reader_class() -> Any:
    """Return :class:`C3DDataReader`, importing ``ezc3d`` on first use."""
    from src.shared.python.sidekick.lab.bio.c3d_reader import (  # noqa: PLC0415 - lazy by design
        C3DDataReader,
    )

    return C3DDataReader


def marker_set_types() -> tuple[Any, Any]:
    """Return ``(MarkerSet, MarkerSetMismatchError)`` lazily."""
    from src.shared.python.sidekick.lab.bio._c3d_marker_set import (  # noqa: PLC0415 - lazy by design
        MarkerSet,
        MarkerSetMismatchError,
    )

    return MarkerSet, MarkerSetMismatchError
