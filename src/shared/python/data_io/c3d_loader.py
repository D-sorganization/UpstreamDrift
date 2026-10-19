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

import importlib
import importlib.util
from pathlib import Path
import sys
import types
from typing import Any

__all__ = ["c3d_reader_class", "marker_set_types"]

_BIO_PACKAGE = "src.shared.python.sidekick.lab.bio"


def _ensure_bio_paths() -> None:
    """Ensure UpstreamDrift's local bio directory is present on bio package __path__."""
    local_bio = str(Path(__file__).resolve().parents[1] / "sidekick" / "lab" / "bio")
    for mod_name in (
        "src.shared.python.sidekick.lab.bio",
        "shared.python.sidekick.lab.bio",
        "sidekick.lab.bio",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "__path__"):
            if local_bio not in mod.__path__:
                mod.__path__.append(local_bio)


def _is_populated_bio_submodule(
    submodule_name: str, mod: types.ModuleType | None
) -> bool:
    """Return True if mod is a valid, populated instance of the bio submodule."""
    if mod is None:
        return False
    if submodule_name == "c3d_reader":
        return hasattr(mod, "C3DDataReader")
    if submodule_name == "_c3d_marker_set":
        return hasattr(mod, "MarkerSet")
    return True


def _load_bio_submodule(submodule_name: str) -> types.ModuleType:
    """Load a submodule of sidekick.lab.bio, resilient to Tools tree pollution (#13348)."""
    _ensure_bio_paths()

    full_src_name = f"{_BIO_PACKAGE}.{submodule_name}"
    aliases = (
        full_src_name,
        f"shared.python.sidekick.lab.bio.{submodule_name}",
        f"sidekick.lab.bio.{submodule_name}",
    )
    for name in aliases:
        mod = sys.modules.get(name)
        if mod is not None and _is_populated_bio_submodule(submodule_name, mod):
            return mod

    try:
        mod = importlib.import_module(full_src_name)
        if _is_populated_bio_submodule(submodule_name, mod):
            return mod
    except ModuleNotFoundError:
        pass

    try:
        mod = importlib.import_module(f"sidekick.lab.bio.{submodule_name}")
        if _is_populated_bio_submodule(submodule_name, mod):
            return mod
    except ModuleNotFoundError:
        pass

    file_path = (
        Path(__file__).resolve().parents[1]
        / "sidekick"
        / "lab"
        / "bio"
        / f"{submodule_name}.py"
    )
    if not file_path.is_file():
        raise ModuleNotFoundError(f"No module named {full_src_name!r}")

    spec = importlib.util.spec_from_file_location(full_src_name, file_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(
            f"Cannot create spec for {full_src_name!r} from {file_path}"
        )

    mod = importlib.util.module_from_spec(spec)
    for alias in aliases:
        sys.modules[alias] = mod

    try:
        spec.loader.exec_module(mod)
    except BaseException:
        for alias in aliases:
            sys.modules.pop(alias, None)
        raise

    return mod


def c3d_reader_class() -> Any:
    """Return :class:`C3DDataReader`, importing ``ezc3d`` on first use."""
    mod = _load_bio_submodule("c3d_reader")
    return mod.C3DDataReader


def marker_set_types() -> tuple[Any, Any]:
    """Return ``(MarkerSet, MarkerSetMismatchError)`` lazily."""
    mod = _load_bio_submodule("_c3d_marker_set")
    return mod.MarkerSet, mod.MarkerSetMismatchError
