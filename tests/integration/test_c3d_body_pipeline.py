"""Integration tests: load every available C3D file as a BodyTarget."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.motion_matching import (
    AlignOptions,
    BodyTarget,
    load_body_target_c3d,
)
from src.shared.python.motion_matching.loaders.c3d_body import (
    default_anatomical_marker_set,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _discover_c3d_files() -> list[Path]:
    """Return every ``.c3d`` file shipped with the repo."""
    candidates: list[Path] = []
    candidates.extend(sorted((REPO_ROOT / "data").glob("*.c3d")))
    extra_root = (
        REPO_ROOT
        / "src"
        / "engines"
        / "physics_engines"
        / "pinocchio"
        / "data"
        / "optical_tour_average"
    )
    if extra_root.is_dir():
        candidates.extend(sorted(extra_root.glob("*.c3d")))
    return candidates


C3D_FILES = _discover_c3d_files()
