"""Every C3D file shipped in ``data/`` must load through the C3D adapter.

The repo carries two distinct capture types:

* Tour-Average golf swings (``data/C3D_TA_*.c3d``) — 38 markers,
  float-mode coordinates declared in **meters** via a 1-D
  ``POINT:UNITS`` char parameter.
* CMU academic mocap (``data/cmu_mocap/subject_64/*.c3d``) — 45
  markers, coordinates declared in **millimeters**.

A units-parsing regression in the Rust parser scaled the meter-based
golf files down 1000x (a 2 mm golf swing) while leaving the mm-based
CMU files coincidentally correct, so this suite pins human-scale
magnitudes for every file rather than just successful parsing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.sources.c3d_adapter import (
    _HAS_C3D_BACKEND,
    C3DAdapter,
)

pytestmark = [pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

ALL_C3D_FILES = sorted(DATA_DIR.rglob("*.c3d"))

# Marker counts per capture type; anything new in data/ falls back to
# the generic >0 assertion.
EXPECTED_MARKERS = {"C3D_TA_": 38, "64_": 45}

# A human capture (golf swing or locomotion) expressed in meters must
# span more than half a meter and stay well under 10 m from origin.
HUMAN_SCALE_MIN_M = 0.5
HUMAN_SCALE_MAX_M = 10.0


def _expected_marker_count(path: Path) -> int | None:
    for prefix, count in EXPECTED_MARKERS.items():
        if path.name.startswith(prefix):
            return count
    return None


def test_data_folder_contains_both_capture_types() -> None:
    """Guard: the two known capture families are present in data/."""
    names = {p.name for p in ALL_C3D_FILES}
    assert "C3D_TA_Driver.c3d" in names
    assert "C3D_TA_Iron.c3d" in names
    assert any(n.startswith("64_") for n in names), "CMU subject_64 files missing"


@pytest.mark.skipif(not _HAS_C3D_BACKEND, reason="no C3D backend installed")
@pytest.mark.parametrize("c3d_path", ALL_C3D_FILES, ids=[p.name for p in ALL_C3D_FILES])
def test_every_data_c3d_loads_at_human_scale(c3d_path: Path) -> None:
    adapter = C3DAdapter()
    assert adapter.supports(c3d_path)

    meta = adapter.metadata(c3d_path)
    assert meta.fps > 0
    assert meta.frame_count > 0

    trajectory = adapter.load(c3d_path)
    assert len(trajectory.frames) == meta.frame_count

    expected_markers = _expected_marker_count(c3d_path)
    # Occlusion drops markers per frame — and some CMU takes have a
    # marker occluded for the whole capture — so compare the best frame
    # against a small tolerance below the declared marker count.
    max_markers = max(len(f.markers) for f in trajectory.frames)
    if expected_markers is not None:
        assert expected_markers - 3 <= max_markers <= expected_markers, (
            f"{c3d_path.name}: expected ~{expected_markers} markers, got {max_markers}"
        )
    else:
        assert max_markers > 0

    coords = [
        abs(c)
        for frame in trajectory.frames
        for marker in frame.markers.values()
        for c in (marker.x, marker.y, marker.z)
    ]
    assert coords, f"{c3d_path.name}: no finite marker coordinates"
    max_abs = max(coords)
    assert HUMAN_SCALE_MIN_M < max_abs < HUMAN_SCALE_MAX_M, (
        f"{c3d_path.name}: max |coord| = {max_abs:.4f} m is not human-scale; "
        "units were likely mis-parsed (meters-vs-millimeters regression)"
    )


@pytest.mark.skipif(not _HAS_C3D_BACKEND, reason="no C3D backend installed")
def test_units_metadata_distinguishes_both_types() -> None:
    """The golf files report meters; the CMU files report millimeters."""
    adapter = C3DAdapter()
    golf = adapter.load(DATA_DIR / "C3D_TA_Driver.c3d")
    cmu = adapter.load(DATA_DIR / "cmu_mocap" / "subject_64" / "64_01.c3d")
    assert golf.metadata["units"].startswith("m")
    assert not golf.metadata["units"].startswith("mm")
    assert cmu.metadata["units"].startswith("mm")
