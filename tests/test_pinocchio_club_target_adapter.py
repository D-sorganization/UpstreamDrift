"""Tests for the Pinocchio club-swing-dataset -> ``ClubTarget`` adapter.

Covers:

* schema validation through ``ClubTarget.__post_init__``,
* round-trip on a synthetic fixture (always runnable),
* round-trip on each real swing-dataset trial under
  ``src/engines/physics_engines/pinocchio/data/club_swing_dataset/`` when present
  (skipped otherwise so this file is portable),
* edge cases: NaN frames, missing partner file, unknown path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
from src.engines.physics_engines.pinocchio.python.motion_matching.club_target_adapter import (  # noqa: E501
    ClubTarget,
    SourceProvenance,
    load_robneal_target,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _identity_dircos(n: int) -> np.ndarray:
    """Return ``(N, 9)`` row-major identity-rotation direction-cosine table."""
    row = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return np.tile(row, (n, 1))


def _write_synthetic_pair(
    tmp_path: Path,
    *,
    name: str = "SYN_test",
    n: int = 50,
    impact_one_based: int = 25,
    inject_nan_frames: tuple[int, ...] = (),
    monotonic_time: bool = True,
) -> Path:
    """Write a synthetic raw + resampled .mat pair under ``tmp_path``.

    Returns the *raw* file path. The structure mirrors the club-swing-dataset
    layout observed at
    ``src/engines/physics_engines/pinocchio/data/club_swing_dataset/`` (see
    issue #4127):

    * raw file: ``data.{time, midhands_xyz, midhands_dircos, clubface_xyz,
      clubface_dircos}`` and ``params.{Impact, impact_frame, ...}``.
    * resampled: ``Time``, ``MH``, ``MH_R`` (with ``MH_R`` shaped ``(3, 3, N)``).
    """
    if monotonic_time:
        time = np.linspace(0.0, 1.0, n, dtype=np.float64)
    else:
        time = np.linspace(0.0, 1.0, n, dtype=np.float64)
        time[5] = time[4]  # break monotonicity

    butt_xyz = np.column_stack(
        [
            0.4 * np.cos(2 * np.pi * time),
            0.4 * np.sin(2 * np.pi * time),
            np.full_like(time, 1.0),
        ]
    ).astype(np.float64)
    head_xyz = butt_xyz + np.array([0.0, 0.0, -1.0])
    butt_dircos = _identity_dircos(n)
    head_dircos = _identity_dircos(n)

    if inject_nan_frames:
        butt_xyz = butt_xyz.copy()
        for idx in inject_nan_frames:
            butt_xyz[idx, 0] = np.nan

    raw_data = {
        "time": time,
        "midhands_xyz": butt_xyz,
        "midhands_dircos": butt_dircos,
        "clubface_xyz": head_xyz,
        "clubface_dircos": head_dircos,
    }
    raw_params = {
        "Impact": impact_one_based,
        "impact_frame": impact_one_based,
        "Address": 1,
        "TopOfBackswing": max(1, impact_one_based // 2),
        "Finish": n,
        "backswing_start": 1,
        "swing_start": 1,
    }
    raw_path = tmp_path / f"{name}.mat"
    sio.savemat(
        str(raw_path), {"data": raw_data, "params": raw_params}, oned_as="column"
    )

    # Resampled file: identical time/positions/identity rotations stack.
    mh_r_stack = np.tile(np.eye(3, dtype=np.float64)[:, :, None], (1, 1, n))
    resampled = {
        "Time": time,
        "MH": butt_xyz,
        "MH_R": mh_r_stack,
    }
    resampled_path = tmp_path / f"{name}_targetKinematics.mat"
    sio.savemat(str(resampled_path), resampled, oned_as="column")
    return raw_path


# ---------------------------------------------------------------------------
# Synthetic-fixture tests (always runnable)
# ---------------------------------------------------------------------------


def test_load_synthetic_round_trip(tmp_path: Path) -> None:
    """Synthetic fixture loads and produces a valid ``ClubTarget``."""
    raw = _write_synthetic_pair(tmp_path, n=100, impact_one_based=42)
    target = load_robneal_target(raw)

    assert isinstance(target, ClubTarget)
    assert isinstance(target.source, SourceProvenance)
    assert target.source.format == "club_swing_mat"
    assert target.source.filename == raw.name
    assert target.source.subject_id == "SYN"
    assert target.time.shape == (100,)
    assert target.butt.shape == (100, 3)
    assert target.clubhead.shape == (100, 3)
    assert target.club_quat.shape == (100, 4)
    # Time is rebased to start at 0.
    assert target.time[0] == pytest.approx(0.0)
    # Identity rotations -> [1, 0, 0, 0] quaternions.
    np.testing.assert_allclose(target.club_quat[0], [1.0, 0.0, 0.0, 0.0])
    qnorms = np.linalg.norm(target.club_quat, axis=1)
    np.testing.assert_allclose(qnorms, 1.0, atol=1.0e-9)
    assert target.impact_idx == 42  # 1-based, matches raw params.Impact


def test_load_accepts_resampled_path(tmp_path: Path) -> None:
    """Passing the ``_targetKinematics.mat`` path also resolves the pair."""
    raw = _write_synthetic_pair(tmp_path, n=20, impact_one_based=10)
    resampled = raw.with_name(raw.stem + "_targetKinematics.mat")
    target_from_raw = load_robneal_target(raw)
    target_from_resampled = load_robneal_target(resampled)
    np.testing.assert_array_equal(target_from_raw.time, target_from_resampled.time)
    np.testing.assert_array_equal(target_from_raw.butt, target_from_resampled.butt)
    assert target_from_raw.impact_idx == target_from_resampled.impact_idx
    assert target_from_raw.source.filename == target_from_resampled.source.filename


def test_drops_nan_frames(tmp_path: Path) -> None:
    """Frames with NaNs are dropped; impact_idx is remapped to a survivor."""
    raw = _write_synthetic_pair(
        tmp_path,
        n=30,
        impact_one_based=15,
        inject_nan_frames=(2, 7),
    )
    target = load_robneal_target(raw)
    assert target.time.shape == (28,)
    assert np.all(np.isfinite(target.butt))
    assert np.all(np.isfinite(target.clubhead))
    # Impact (raw 1-based 15 -> 0-based 14) survives the drop, but the
    # 0-based index in the kept array shifts by 2 (frames 2 and 7 dropped).
    assert target.impact_idx == 15 - 2  # 13 (1-based)


def test_dropped_impact_frame_uses_next_survivor(tmp_path: Path) -> None:
    """If the impact frame itself is NaN we fall through to the next frame."""
    raw = _write_synthetic_pair(
        tmp_path,
        n=20,
        impact_one_based=10,
        inject_nan_frames=(9,),  # 0-based 9 == 1-based 10 (the impact)
    )
    target = load_robneal_target(raw)
    # Original impact frame (1-based 10, 0-based 9) was dropped. The next
    # surviving raw frame is 0-based 10 -> in the kept array, that's 0-based 9
    # -> 1-based 10.
    assert target.impact_idx == 10
    assert target.time.shape == (19,)


def test_missing_partner_raises(tmp_path: Path) -> None:
    """Deleting either half of the pair triggers a ``FileNotFoundError``."""
    raw = _write_synthetic_pair(tmp_path, n=10, impact_one_based=5)
    raw.with_name(raw.stem + "_targetKinematics.mat").unlink()
    with pytest.raises(FileNotFoundError, match="Resampled .mat partner"):
        load_robneal_target(raw)


def test_pinocchio_club_target_adapter_missing_file_raises(tmp_path: Path) -> None:
    """A non-existent path is rejected up front."""
    with pytest.raises(FileNotFoundError):
        load_robneal_target(tmp_path / "does_not_exist.mat")


def test_non_mat_extension_rejected(tmp_path: Path) -> None:
    """Paths without the ``.mat`` extension are rejected."""
    bogus = tmp_path / "trial.txt"
    bogus.write_text("hello")
    with pytest.raises(ValueError, match="Expected a .mat file"):
        load_robneal_target(bogus)


def test_validation_rejects_out_of_range_impact(tmp_path: Path) -> None:
    """Out-of-range ``params.Impact`` raises ``ValueError``."""
    # Hand-craft a bad raw with Impact > N.
    bad_raw = tmp_path / "BAD_test.mat"
    sio.savemat(
        str(bad_raw),
        {
            "data": {
                "time": np.linspace(0.0, 1.0, 5),
                "midhands_xyz": np.zeros((5, 3)),
                "midhands_dircos": _identity_dircos(5),
                "clubface_xyz": np.zeros((5, 3)) + np.array([0, 0, -1.0]),
                "clubface_dircos": _identity_dircos(5),
            },
            "params": {"Impact": 99, "impact_frame": 99},
        },
        oned_as="column",
    )
    # Resampled partner with matching N.
    sio.savemat(
        str(tmp_path / "BAD_test_targetKinematics.mat"),
        {
            "Time": np.linspace(0.0, 1.0, 5),
            "MH": np.zeros((5, 3)),
            "MH_R": np.tile(np.eye(3)[:, :, None], (1, 1, 5)),
        },
        oned_as="column",
    )
    with pytest.raises(ValueError, match="impact index"):
        load_robneal_target(bad_raw)


def test_quaternions_are_unit_norm_for_random_rotations(
    tmp_path: Path,
) -> None:
    """Non-identity rotations still yield unit-norm quaternions."""
    n = 40
    time = np.linspace(0.0, 1.0, n)
    butt_xyz = np.zeros((n, 3))
    butt_xyz[:, 2] = 1.0
    head_xyz = butt_xyz - np.array([0.0, 0.0, 1.0])
    # Build a smoothly-rotating clubface frame (yaw about +Z).
    head_dircos = np.zeros((n, 9))
    for i in range(n):
        a = 2.0 * np.pi * i / (n - 1) * 0.25
        ca, sa = np.cos(a), np.sin(a)
        # Row order [Xx, Xy, Xz, Yx, Yy, Yz, Zx, Zy, Zz].
        head_dircos[i] = [ca, sa, 0.0, -sa, ca, 0.0, 0.0, 0.0, 1.0]
    butt_dircos = _identity_dircos(n)

    raw = tmp_path / "ROT_trial.mat"
    sio.savemat(
        str(raw),
        {
            "data": {
                "time": time,
                "midhands_xyz": butt_xyz,
                "midhands_dircos": butt_dircos,
                "clubface_xyz": head_xyz,
                "clubface_dircos": head_dircos,
            },
            "params": {"Impact": n // 2, "impact_frame": n // 2},
        },
        oned_as="column",
    )
    sio.savemat(
        str(tmp_path / "ROT_trial_targetKinematics.mat"),
        {
            "Time": time,
            "MH": butt_xyz,
            "MH_R": np.tile(np.eye(3)[:, :, None], (1, 1, n)),
        },
        oned_as="column",
    )

    target = load_robneal_target(raw)
    qnorms = np.linalg.norm(target.club_quat, axis=1)
    np.testing.assert_allclose(qnorms, 1.0, atol=1.0e-9)
    # The first quaternion is identity rotation.
    np.testing.assert_allclose(target.club_quat[0], [1, 0, 0, 0], atol=1.0e-9)
    # Quaternions should not all be identical (the clubface is rotating).
    assert np.linalg.norm(target.club_quat[-1] - target.club_quat[0]) > 1e-6


# ---------------------------------------------------------------------------
# Real-fixture tests (skipped when data/ is unavailable)
# ---------------------------------------------------------------------------


def _repo_root_from_here() -> Path:
    """Walk up to find the repo root (directory with ``pyproject.toml``)."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("could not locate repo root from test file")


_SWING_DATASET_DIR = (
    _repo_root_from_here()
    / "src"
    / "engines"
    / "physics_engines"
    / "pinocchio"
    / "data"
    / "club_swing_dataset"
)
_REAL_TRIALS = ("TW_ProV1", "TW_wiffle", "GW_ProV1", "GW_wiffle")
