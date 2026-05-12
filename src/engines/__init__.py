"""Engine package - provides physics engine implementations."""

from __future__ import annotations
import importlib
from pathlib import Path

__all__ = ["get_engine_catalog", "is_fit_capable"]


def get_engine_catalog() -> dict[str, dict[str, bool]]:
    """Return a catalog of engines and their capabilities.

    Dynamically scans src/engines/physics_engines/ to determine fit_swing capabilities.
    """
    catalog: dict[str, dict[str, bool]] = {}
    engine_dir = Path(__file__).parent / "physics_engines"
    if not engine_dir.exists():
        return catalog

    for path in engine_dir.iterdir():
        if path.is_dir() and not path.name.startswith("_") and path.name != "tests":
            engine_name = path.name
            fit_capable = True

            # Check for explicitly marked FIT_INCAPABLE in _tier.py
            tier_file = path / "_tier.py"
            if tier_file.exists():
                try:
                    mod = importlib.import_module(
                        f"src.engines.physics_engines.{engine_name}._tier"
                    )
                    if getattr(mod, "FIT_INCAPABLE", False):
                        fit_capable = False
                except ImportError:
                    pass

            # If capable, try to import the provider to trigger registration
            if fit_capable:
                try:
                    importlib.import_module(
                        f"src.engines.physics_engines.{engine_name}.python.motion_matching.provider"
                    )
                except ImportError:
                    pass

            catalog[engine_name] = {"fit_capable": fit_capable}

    return catalog


def is_fit_capable(engine_name: str) -> bool:
    """Return True if the engine is fit_swing capable."""
    catalog = get_engine_catalog()
    return catalog.get(engine_name, {}).get("fit_capable", False)
