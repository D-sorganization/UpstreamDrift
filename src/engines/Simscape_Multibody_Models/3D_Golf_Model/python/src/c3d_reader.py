"""Backwards-compatible shim. Use the canonical reader instead.

The canonical implementation lives at
``src/shared/python/sidekick/lab/bio/c3d_reader.py`` (issue #4484).
This module re-exports the public API plus a project-local
``load_tour_average_reader`` convenience that resolves the Tour-average C3D
relative to this engine's repo layout.
"""

from __future__ import annotations

from pathlib import Path

from src.shared.python.sidekick.lab.bio.c3d_reader import (  # noqa: F401
    BIOMECHANICAL_MARKER_MAX_M,
    BIOMECHANICAL_MARKER_MIN_M,
    SCHEMA_VERSION,
    C3DDataReader,
    C3DEvent,
    C3DMapping,
    C3DMetadata,
)

__all__ = [
    "BIOMECHANICAL_MARKER_MAX_M",
    "BIOMECHANICAL_MARKER_MIN_M",
    "SCHEMA_VERSION",
    "C3DDataReader",
    "C3DEvent",
    "C3DMapping",
    "C3DMetadata",
    "load_tour_average_reader",
]


def load_tour_average_reader(base_directory: Path | None = None) -> C3DDataReader:
    """Convenience loader for the repository's Tour average capture.

    Args:
        base_directory: Optional base directory containing the repository
            files. If omitted, the repository root is derived from this
            module's location.

    Returns:
        A configured :class:`C3DDataReader` pointing to the Tour average
        capture file.
    """
    base_path = base_directory or Path(__file__).resolve().parents[2]
    default_path = (
        base_path / "matlab" / "Data" / "Mocap C3D Files" / "C3DExport Tour average.c3d"
    )  # noqa: E501
    return C3DDataReader(default_path)
