"""Fail-closed import guard preventing sibling checkouts from shadowing UD (#15014)."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

_SHARED_PYTHON_ROOT = Path(__file__).resolve().parents[2]


def verify_local_anthropometry_module(mod: Any) -> Any:
    """Reject ``humanoid_character_builder`` resolved outside UD ``src/shared/python``."""
    mod_file = getattr(mod, "__file__", None)
    if not mod_file or not Path(mod_file).resolve().is_relative_to(_SHARED_PYTHON_ROOT):
        raise ImportError(
            "humanoid_character_builder.core.anthropometry resolved to "
            f"{mod_file!r}, outside UpstreamDrift src/shared/python "
            f"({_SHARED_PYTHON_ROOT}). Uninstall any sibling editable "
            "Tools/ud_tools checkout or prepend src/shared/python to PYTHONPATH."
        )
    return mod


def load_local_anthropometry_module(caller_name: str) -> Any:
    """Load and verify the repo-local ``humanoid_character_builder.core.anthropometry``."""
    prefix = (
        "src.shared.python." if caller_name.startswith("src.shared.python.") else ""
    )
    target = f"{prefix}humanoid_character_builder.core.anthropometry"
    return verify_local_anthropometry_module(importlib.import_module(target))
