"""OpenSim-facing alias of the shared TRC reader/writer (#12065).

The implementation lives in :mod:`src.shared.python.motion_matching.trc_io`
so C3D preparation can write TRC without importing this engine package (whose
import probes the native ``opensim`` SDK and its bundled ``ezc3d.dll``).
Existing import paths keep working through this re-export.
"""

from __future__ import annotations

from src.shared.python.motion_matching.trc_io import read_trc, write_trc

__all__ = ["read_trc", "write_trc"]
