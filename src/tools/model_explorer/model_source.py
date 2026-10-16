"""Turn any explorable model file into URDF text (#12878).

The Model Explorer renders URDF; this module is the one place that converts an
OpenSim ``.osim`` or MuJoCo MJCF file into URDF first, so the widget stays a
thin view and the conversion can be tested without a display.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["model_as_urdf"]

_MJCF_SUFFIXES = (".xml", ".mjcf")


def _is_mjcf(path: Path) -> bool:
    return path.suffix.lower() in _MJCF_SUFFIXES and "<mujoco" in path.read_text(
        encoding="utf-8", errors="replace"
    )


def _mjcf_as_urdf(path: Path) -> str:
    from src.shared.python.model_generation.library.unified_loader import (
        UnifiedModelLoader,
    )

    result = UnifiedModelLoader().load_file(path)
    if not result.success or result.model is None:
        raise ValueError(result.error or f"cannot parse MJCF '{path.name}'")
    return result.model.to_urdf()


def model_as_urdf(path: Path) -> str:
    """Return URDF text for a URDF, MJCF or OSIM file.

    Postcondition: the returned text defines at least one ``<link``.

    Raises:
        OSError: the file cannot be read.
        ValueError: the file is not a loadable model or defines no bodies.
    """
    if path.suffix.lower() == ".osim":
        from .osim_loader import OsimLoader

        urdf_content = OsimLoader().to_urdf(path)
    elif _is_mjcf(path):
        urdf_content = _mjcf_as_urdf(path)
    else:
        urdf_content = path.read_text(encoding="utf-8")
    if "<link" not in urdf_content:
        raise ValueError(f"'{path.name}' defines no bodies, nothing to explore")
    return urdf_content
