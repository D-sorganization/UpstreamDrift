"""Compatibility namespace for the Simscape C3D viewer app package."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_simscape_apps = (
    Path(__file__).resolve().parent.parent
    / "src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps"
)
if _simscape_apps.exists():
    __path__.append(str(_simscape_apps))  # type: ignore[attr-defined]
