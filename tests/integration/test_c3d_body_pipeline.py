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
        / "gears_tour_average"
    )
    if extra_root.is_dir():
        candidates.extend(sorted(extra_root.glob("*.c3d")))
    return candidates


C3D_FILES = _discover_c3d_files()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not C3D_FILES, reason="no C3D fixtures shipped in this checkout")
@pytest.mark.parametrize("c3d_path", C3D_FILES, ids=lambda p: p.name)
def test_all_c3d_files_load_as_body_target(c3d_path: Path) -> None:
    """Every C3D file loads, validates, and yields the default marker set."""
    opts = AlignOptions(
        sample_rate_hz=1000.0,
        simulation_time_s=0.3,
        time_alignment="impact",
        impact_target_t_s=0.25,
    )
    bt = load_body_target_c3d(c3d_path, opts)
    assert isinstance(bt, BodyTarget)
    assert bt.marker_names == default_anatomical_marker_set()
    # Resampled grid: int(0.3 * 1000) + 1 = 301 samples.
    assert bt.time.shape == (301,)
    assert bt.marker_xyz.shape[0] == 301
    assert bt.marker_xyz.shape[2] == 3
    assert 0 <= bt.impact_idx < bt.time.shape[0]
