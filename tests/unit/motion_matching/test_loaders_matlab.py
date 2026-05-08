"""Tests for the ``.mat`` club-target loader (issue #4477).

Integration-style tests rely on the canonical ``.mat`` fixtures under
``src/engines/physics_engines/pinocchio/data/<dataset>/``. Pure-unit tests
(rotation rejection, missing path) run unconditionally with synthetic data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat
from src.shared.python.motion_matching import (
    AlignOptions,
    ClubTarget,
    load_club_target,
    load_club_target_excel,
    load_club_target_mat,
)
from src.shared.python.motion_matching.loaders.matlab_dataset import (
    _stamped_impact_index,
)

from ._fixtures import repo_root

_MAT_RELATIVE_DIR = "src/engines/physics_engines/pinocchio/data/rob_neal"
_CANONICAL_TRIALS = ("TW_ProV1", "TW_wiffle", "GW_ProV1", "GW_wiffle")
_EXCEL_RELATIVE = (
    "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/"
    "golf_gui/Motion Capture Plotter/Wiffle_ProV1_club_3D_data.xlsx"
)
# All four canonical files contain driver-class swings (verified by max-speed
# probe). Empirical |v_clubhead|@impact range is roughly 50.7-51.8 m/s. We use
# a generous physical-plausibility band that clearly distinguishes a swing
# from noise but does not over-fit the snapshot values.
_DRIVER_SPEED_BAND_MPS = (35.0, 60.0)


def _mat_path(name: str) -> Path | None:
    p = repo_root() / _MAT_RELATIVE_DIR / f"{name}.mat"
    return p if p.is_file() else None


def _excel_path() -> Path | None:
    p = repo_root() / _EXCEL_RELATIVE
    return p if p.is_file() else None


def _clubhead_speed_at(target: ClubTarget) -> float:
    """Central-difference clubhead speed at ``target.impact_idx`` (1-based)."""
    i = int(target.impact_idx) - 1
    n = target.clubhead.shape[0]
    if i <= 0 or i >= n - 1:
        raise AssertionError(
            f"impact_idx={target.impact_idx} too close to edge of array of len {n}"
        )
    dt = float(target.time[i + 1] - target.time[i - 1])
    v = (target.clubhead[i + 1] - target.clubhead[i - 1]) / dt
    return float(np.linalg.norm(v))


def _make_synthetic_mat(
    path: Path,
    *,
    bad_rotation: bool = False,
    n: int = 50,
) -> None:
    """Write a minimal synthetic .mat file matching the documented schema.

    When ``bad_rotation`` is True, replace the first rotation with a reflection
    (``det = -1``) so the loader's rotation-validity check rejects it.
    """
    fs = 240.0
    time = (np.arange(n) - n // 2) / fs
    # Linearly moving clubhead and butt; both safely below 5 m position cap.
    butt = np.column_stack(
        [
            0.1 * time,
            np.zeros(n),
            np.zeros(n),
        ]
    )
    clubhead = butt + np.array([0.0, 0.0, 1.0])
    rot = np.tile(np.eye(3, dtype=np.float64), (n, 1, 1))
    if bad_rotation:
        # Reflection: flip one column => det = -1.
        rot[0] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    dircos_flat = rot.reshape(n, 9)
    data = {
        "time": time,
        "midhands_xyz": butt,
        "midhands_dircos": dircos_flat,
        "clubface_xyz": clubhead,
        "clubface_dircos": dircos_flat,
    }
    params = {
        "impact_frame": int(n // 2 + 1),
        "Impact": int(n // 2 + 1),
        "Address": 1,
        "TopOfBackswing": 1,
        "Finish": int(n),
        "swing_start": 1,
        "backswing_start": 1,
    }
    savemat(str(path), {"data": data, "params": params})


# ---------------------------------------------------------------------------
# Integration tests against canonical fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trial", _CANONICAL_TRIALS)
def test_canonical_mat_loads_to_clubtarget(trial: str) -> None:
    p = _mat_path(trial)
    if p is None:
        pytest.skip(f"{trial}.mat not present")
    target = load_club_target_mat(p, AlignOptions())
    assert isinstance(target, ClubTarget)
    assert target.time.shape[0] == target.butt.shape[0]
    assert target.time.shape[0] == target.clubhead.shape[0]
    assert target.time.shape[0] == target.club_quat.shape[0]
    qnorms = np.linalg.norm(target.club_quat, axis=1)
    assert np.all(np.abs(qnorms - 1.0) < 1e-6)
    assert 1 <= target.impact_idx <= target.time.shape[0]


@pytest.mark.parametrize("trial", _CANONICAL_TRIALS)
def test_canonical_mat_clubhead_speed_at_impact(trial: str) -> None:
    """Impact-time clubhead speed must fall in the driver-class plausibility band.

    All four canonical fixtures are driver-class swings (verified via max-speed
    probe). The issue text speculated GW = iron, but the recorded data shows
    GW also at driver speeds, so we use a single physical-plausibility band.
    """
    p = _mat_path(trial)
    if p is None:
        pytest.skip(f"{trial}.mat not present")
    target = load_club_target_mat(p, AlignOptions())
    speed = _clubhead_speed_at(target)
    lo, hi = _DRIVER_SPEED_BAND_MPS
    assert lo <= speed <= hi, (
        f"{trial}: clubhead speed at impact {speed:.2f} m/s "
        f"outside expected band [{lo}, {hi}]"
    )


def test_dispatcher_routes_mat_to_loader() -> None:
    p = _mat_path("TW_ProV1")
    if p is None:
        pytest.skip("TW_ProV1.mat not present")
    target = load_club_target(p)
    assert isinstance(target, ClubTarget)
    assert target.source.format == "mat_dataset"


def test_stamped_vs_heuristic_impact_within_tolerance() -> None:
    """Stamped (.mat) and heuristic (.xlsx) impact_idx must agree to within 2 samples."""
    mat_p = _mat_path("TW_ProV1")
    xlsx_p = _excel_path()
    if mat_p is None or xlsx_p is None:
        pytest.skip("Both .mat and .xlsx fixtures are required for parity check")
    opts = AlignOptions()
    mat_target = load_club_target_mat(mat_p, opts)
    xlsx_target = load_club_target_excel(xlsx_p, "TW_ProV1", opts)
    diff = abs(int(mat_target.impact_idx) - int(xlsx_target.impact_idx))
    assert diff <= 2, (
        f"Stamped vs heuristic impact_idx differ by {diff} samples "
        f"(mat={mat_target.impact_idx}, xlsx={xlsx_target.impact_idx})"
    )


def test_provenance_records_format_and_subject(tmp_path: Path) -> None:
    p = _mat_path("TW_ProV1")
    if p is None:
        pytest.skip("TW_ProV1.mat not present")
    target = load_club_target_mat(p, AlignOptions())
    assert target.source.format == "mat_dataset"
    assert target.source.filename == "TW_ProV1.mat"
    assert target.source.trial_id == "TW_ProV1"
    # subject_id is the leading underscore-segment of the stem; for the canonical
    # fixtures that is the two-letter recording-session prefix. We only assert the
    # invariant (non-empty, no path separators) so the loader stays generic.
    assert target.source.subject_id
    assert "/" not in target.source.subject_id
    assert "\\" not in target.source.subject_id
    assert len(target.source.sha256) == 64


# ---------------------------------------------------------------------------
# Pure unit tests with synthetic data
# ---------------------------------------------------------------------------


def test_reflect_rotation_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad_rotation.mat"
    _make_synthetic_mat(bad, bad_rotation=True)
    with pytest.raises(ValueError, match="rotation"):
        load_club_target_mat(bad, AlignOptions())


def test_synthetic_mat_loads_to_clubtarget(tmp_path: Path) -> None:
    good = tmp_path / "good.mat"
    _make_synthetic_mat(good)
    target = load_club_target_mat(good, AlignOptions())
    assert isinstance(target, ClubTarget)


def test_missing_path_raises() -> None:
    from src.shared.python.core.contracts import PreconditionError

    with pytest.raises((FileNotFoundError, ValueError, PreconditionError)):
        load_club_target_mat("does/not/exist.mat", AlignOptions())


def test_stamped_impact_uses_zero_crossing() -> None:
    """When the time vector spans t=0, the stamped impact is the t=0 row."""
    time = np.linspace(-0.5, 0.5, 121)

    class _P:
        Impact = 1  # would be wrong as a row index; should be ignored

    idx = _stamped_impact_index(time, _P())
    assert idx == int(np.argmin(np.abs(time)))
    assert abs(float(time[idx])) < 1e-9


def test_stamped_impact_falls_back_to_params(tmp_path: Path) -> None:
    """When time does not span t=0, fall back to params.Impact (1-based)."""
    time = np.linspace(0.05, 0.5, 50)

    class _P:
        Impact = 10

    idx = _stamped_impact_index(time, _P())
    assert idx == 9  # 10 (1-based) - 1


def test_stamped_impact_rejects_out_of_range_params() -> None:
    time = np.linspace(0.05, 0.5, 50)

    class _P:
        Impact = 9999

    with pytest.raises(ValueError, match="out of range"):
        _stamped_impact_index(time, _P())
