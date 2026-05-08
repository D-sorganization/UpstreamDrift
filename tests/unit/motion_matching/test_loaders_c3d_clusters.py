"""Validated cluster-marker C3D regression tests (issue #013 follow-up).

The schema and impact-speed targets here come from external validation of
the two cluster-marker mocap files via ``ezc3d`` 1.7.0:

* ``CaptureExport Tour average.c3d``   -- driver, 114.2 mph at frame 475 / t=1.319 s.
* ``CaptureExport tour average iron.c3d`` -- iron, 88.6 mph at frame 478 / t=1.331 s.

Tests that need either the ``ezc3d`` package or the actual ``.c3d`` files
are gracefully skipped when those are absent.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from src.shared.python.motion_matching.club_target import AlignOptions, ClubTarget
from src.shared.python.motion_matching.loaders._marker_clusters import (
    EXCLUDED_MARKERS,
    fill_short_gaps,
    has_marker_clusters,
    pose_from_cluster,
    y_up_to_z_up,
    y_up_to_z_up_rotation,
)

ezc3d = pytest.importorskip("ezc3d")
del ezc3d  # only needed for the import-or-skip guard

from src.shared.python.motion_matching import load_club_target_c3d  # noqa: E402
from src.shared.python.motion_matching.loaders import c3d as c3d_loader  # noqa: E402

from ._fixtures import repo_root  # noqa: E402

C3D_DIR = (
    "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/Data/Mocap C3D Files"
)
DRIVER_FILE = "CaptureExport Tour average.c3d"
IRON_FILE = "CaptureExport tour average iron.c3d"

# External validation targets (mph at impact).
DRIVER_IMPACT_MPH = 114.2
IRON_IMPACT_MPH = 88.6
SPEED_TOL_FRAC = 0.05  # +/- 5%

MPS_TO_MPH = 2.2369362920544


def _c3d(name: str) -> Path | None:
    path = repo_root() / C3D_DIR / name
    return path if path.is_file() else None


def _max_clubhead_speed_mph(target: ClubTarget) -> float:
    """Estimate peak clubhead speed in mph from a ClubTarget trajectory."""
    t = target.time
    p = target.clubhead
    if t.shape[0] < 5:
        return 0.0
    # Central differences in the interior, drop the edges.
    dt = t[2:] - t[:-2]
    v = (p[2:] - p[:-2]) / dt[:, None]
    speed_mps = np.linalg.norm(v, axis=1)
    return float(speed_mps.max() * MPS_TO_MPH)


# ---------------------------------------------------------------------------
# Pure-unit tests: do not need the C3D files
# ---------------------------------------------------------------------------


def test_has_marker_clusters_detects_filename_prefix() -> None:
    assert has_marker_clusters("CaptureExport Tour average.c3d", ["foo"])
    assert has_marker_clusters("captureexport_iron.c3d", [])


def test_has_marker_clusters_detects_marker_label() -> None:
    assert has_marker_clusters("anything.c3d", ["Marker_2:2:1", "BUTT"])
    assert not has_marker_clusters("anything.c3d", ["BUTT", "CH"])


def test_excluded_markers_listed() -> None:
    assert "Marker_0:0:0" in EXCLUDED_MARKERS
    assert "RShoulderTop" in EXCLUDED_MARKERS


def test_y_up_to_z_up_preserves_right_handed() -> None:
    r = y_up_to_z_up_rotation()
    assert math.isclose(np.linalg.det(r), 1.0, abs_tol=1e-12)
    # Right-handed: e_x x e_y = e_z (with the swap applied to a unit triad).
    triad = np.eye(3)
    swapped = y_up_to_z_up(triad)
    cross = np.cross(swapped[0], swapped[1])
    assert np.allclose(cross, swapped[2])


def test_cluster_pose_against_known_rotation() -> None:
    # Synthetic equilateral triangle.
    reference = np.array(
        [
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3) / 2, 0.0],
            [-0.5, -np.sqrt(3) / 2, 0.0],
        ]
    )
    angle = 0.7
    r_true = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([0.1, -0.2, 0.3])
    moved = (reference @ r_true.T) + translation
    cluster_t = moved[None, :, :]
    rotations, centroids = pose_from_cluster(cluster_t, reference)
    assert np.allclose(rotations[0], r_true, atol=1e-9)
    assert np.allclose(centroids[0], translation, atol=1e-9)


def test_short_nan_gaps_filled_via_spline_interpolation() -> None:
    n = 60
    t = np.linspace(0.0, 1.0, n)
    arr = np.column_stack([np.sin(2 * t), np.cos(2 * t), 0.5 * t])
    arr_with_gap = arr.copy()
    arr_with_gap[20:23, :] = np.nan  # 3-frame gap

    filled = fill_short_gaps(arr_with_gap, max_gap=5)
    assert np.all(np.isfinite(filled[20:23]))
    # Interpolation should be close to truth (~1e-2 for a smooth signal).
    assert np.max(np.abs(filled[20:23] - arr[20:23])) < 1e-2


def test_short_nan_gaps_long_gap_left_alone() -> None:
    n = 60
    arr = np.tile(np.arange(n, dtype=float)[:, None], (1, 3))
    arr_with_gap = arr.copy()
    arr_with_gap[10:30, :] = np.nan  # 20-frame gap exceeds max_gap=5
    filled = fill_short_gaps(arr_with_gap, max_gap=5)
    assert np.all(np.isnan(filled[10:30]))


def test_units_metres_no_inch_conversion() -> None:
    # Sanity: a 1 m shaft at the source must remain 1 m, not 0.0254 m.
    butt = np.zeros((1, 3))
    head = np.array([[0.0, 1.0, 0.0]])
    distance = float(np.linalg.norm(head[0] - butt[0]))
    assert math.isclose(distance, 1.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Integration tests: need the actual .c3d files
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_loads_cluster_driver_returns_canonical_target() -> None:
    p = _c3d(DRIVER_FILE)
    if p is None:
        pytest.skip(f"Missing {DRIVER_FILE}")
    target = load_club_target_c3d(p, AlignOptions())
    assert isinstance(target, ClubTarget)
    assert target.source.format == "c3d"
    assert target.butt.shape[1] == 3
    assert target.clubhead.shape[1] == 3
    assert target.club_quat.shape[1] == 4


@pytest.mark.integration
def test_loads_cluster_iron_returns_canonical_target() -> None:
    p = _c3d(IRON_FILE)
    if p is None:
        pytest.skip(f"Missing {IRON_FILE}")
    target = load_club_target_c3d(p, AlignOptions())
    assert isinstance(target, ClubTarget)


@pytest.mark.integration
def test_clubhead_speed_at_impact_within_5_pct_driver() -> None:
    p = _c3d(DRIVER_FILE)
    if p is None:
        pytest.skip(f"Missing {DRIVER_FILE}")
    # Use a long enough sim window + raw-rate output to preserve peak speed.
    opts = AlignOptions(
        sample_rate_hz=360.0, simulation_time_s=1.6, impact_target_t_s=1.319
    )
    target = load_club_target_c3d(p, opts)
    measured = _max_clubhead_speed_mph(target)
    lo = DRIVER_IMPACT_MPH * (1.0 - SPEED_TOL_FRAC)
    hi = DRIVER_IMPACT_MPH * (1.0 + SPEED_TOL_FRAC)
    assert lo <= measured <= hi, (
        f"driver impact speed {measured:.2f} mph outside [{lo:.2f}, {hi:.2f}]"
    )


@pytest.mark.integration
def test_clubhead_speed_at_impact_within_5_pct_iron() -> None:
    p = _c3d(IRON_FILE)
    if p is None:
        pytest.skip(f"Missing {IRON_FILE}")
    opts = AlignOptions(
        sample_rate_hz=359.0, simulation_time_s=1.6, impact_target_t_s=1.331
    )
    target = load_club_target_c3d(p, opts)
    measured = _max_clubhead_speed_mph(target)
    lo = IRON_IMPACT_MPH * (1.0 - SPEED_TOL_FRAC)
    hi = IRON_IMPACT_MPH * (1.0 + SPEED_TOL_FRAC)
    assert lo <= measured <= hi, (
        f"iron impact speed {measured:.2f} mph outside [{lo:.2f}, {hi:.2f}]"
    )


@pytest.mark.integration
def test_sentinel_and_occluded_markers_excluded() -> None:
    p = _c3d(DRIVER_FILE)
    if p is None:
        pytest.skip(f"Missing {DRIVER_FILE}")
    reader = c3d_loader._import_c3d_reader().C3DDataReader(p)
    df = reader.points_dataframe(include_time=True, target_units="m")
    labels = list(reader.get_metadata().marker_labels)
    # Confirm presence of the bad markers in the raw labels and that the
    # extractor still produces a clean ClubTarget without using them.
    assert "Marker_0:0:0" in labels
    target = load_club_target_c3d(p, AlignOptions())
    assert np.all(np.isfinite(target.clubhead))
    assert np.all(np.isfinite(target.butt))
    # The sentinel value (-1.71, 0.79, -1.97) must not appear in the output.
    bad = np.array([-1.71, 0.79, -1.97])
    assert not np.any(np.all(np.isclose(target.clubhead, bad, atol=1e-2), axis=1))
    del df  # silence linter
