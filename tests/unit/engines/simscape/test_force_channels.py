"""Tests for the Simscape logged-force loader (#11303, FTO-18).

No MATLAB. Synthetic mini CSVs are created by the tests (they are not
measured data); the committed trial CSV is used for real-data checks.
"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path

import numpy as np
import pytest

from src.engines.simscape.adapter import SimscapeAdapter
from src.engines.simscape.force_channels import (
    SIMSCAPE_FORCE_CHANNELS,
    SIMSCAPE_JOINTS,
    load_simscape_force_series,
)
from src.shared.python.force_overlay import ForceTorqueSeries, WrenchKind

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[4]
TRIAL = (
    REPO
    / "src/engines/Simscape_Multibody_Models/3D_Golf_Model"
    / "golf_swing_dataset_20250907_bk/trial_001_20251117_114559.csv"
)

RZ90 = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]


def _write(tmp_path: Path, cols: dict[str, list[float]]) -> Path:
    path = tmp_path / "synthetic_trial.csv"
    names = list(cols)
    n = len(cols[names[0]])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(names)
        for i in range(n):
            w.writerow([repr(cols[k][i]) for k in names])
    return path


def _vec(prefix: str, rows: list[tuple[float, float, float]]) -> dict:
    return {f"{prefix}{k}": [r[i] for r in rows] for i, k in enumerate("123")}


def _rot(prefix: str, mats: list[list[list[float]]]) -> dict:
    return {
        f"{prefix}_I{i + 1}{j + 1}": [m[i][j] for m in mats]
        for i in range(3)
        for j in range(3)
    }


def _world_cols() -> dict:
    cols: dict = {"time": [0.0, 0.01]}
    cols |= _vec("HipLogs_BaseonHipForceGlobal_", [(1, 2, 3), (4, 5, 6)])
    cols |= _vec("HipLogs_BaseonHipTorqueGlobal_", [(7, 8, 9), (1, 1, 1)])
    cols |= _vec("HipLogs_HipGlobalPosition_dim", [(0, 0, 1), (0, 0.1, 1)])
    cols |= _vec("CalculatedSignalsLogs_TotalHandForceGlobal_", [(10, 0, 0)] * 2)
    cols |= _vec("CalculatedSignalsLogs_TotalHandTorqueGlobal_", [(0, 20, 0)] * 2)
    cols |= _vec("MidpointCalcsLogs_MPGlobalPosition_", [(0.5, 0.5, 1.0)] * 2)
    cols |= _vec("MomentandCoupleLogs_LHMOFonClubGlobal_", [(0, 0, 3)] * 2)
    cols |= _vec("LWLogs_LHGlobalPosition_", [(0.4, 0.5, 1.0)] * 2)
    return cols


def _joint_cols(rotation: bool = True, mats=None) -> dict:
    cols: dict = {"time": [0.0, 0.01]}
    cols |= _vec("LSLogs_ConstraintForceLocal_", [(1.0, 2.0, 3.0)] * 2)
    cols |= _vec("LSLogs_ConstraintTorqueLocal_", [(0.0, 1.0, 0.0)] * 2)
    cols |= _vec("LSLogs_GlobalPosition_", [(0.1, 0.2, 1.3)] * 2)
    if rotation:
        cols |= _rot("LSLogs_Rotation_Transform", mats or [RZ90, RZ90])
    return cols


def test_world_channels_exact(tmp_path: Path) -> None:
    series, missing = load_simscape_force_series(_write(tmp_path, _world_cols()))
    assert isinstance(series, ForceTorqueSeries)
    assert series.times_s == (0.0, 0.01)
    by = {w.label: w for w in series[1].wrenches}
    hip = by["external:base_on_hip"]
    assert hip.kind is WrenchKind.EXTERNAL and hip.body == "pelvis"
    assert hip.force_n == (4.0, 5.0, 6.0) and hip.torque_nm == (1.0, 1.0, 1.0)
    assert hip.point_m == (0.0, 0.1, 1.0)
    hand = by["grip:total_hand"]
    assert hand.kind is WrenchKind.GRIP and hand.body == "club"
    assert hand.force_n == (10.0, 0.0, 0.0) and hand.torque_nm == (0.0, 20.0, 0.0)
    assert hand.point_m == (0.5, 0.5, 1.0)
    lh = by["grip:lh_mof"]
    assert lh.force_n is None and lh.torque_nm == (0.0, 0.0, 3.0)
    assert "grip:rh_mof:torque" in missing
    assert "grip:total_hand:force" not in missing


def test_local_joint_rotated_by_logged_r(tmp_path: Path) -> None:
    series, _ = load_simscape_force_series(_write(tmp_path, _joint_cols()))
    w = {x.label: x for x in series[0].wrenches}["joint_reaction:LS"]
    assert w.kind is WrenchKind.JOINT_REACTION and w.body == "LS"
    expected_f = np.array(RZ90) @ np.array([1.0, 2.0, 3.0])
    expected_t = np.array(RZ90) @ np.array([0.0, 1.0, 0.0])
    assert w.force_n == pytest.approx(tuple(expected_f))
    assert w.force_n == pytest.approx((-2.0, 1.0, 3.0))
    assert w.torque_nm == pytest.approx(tuple(expected_t))
    assert w.point_m == (0.1, 0.2, 1.3)


def test_missing_rotation_is_unavailable_not_unrotated(tmp_path: Path) -> None:
    series, missing = load_simscape_force_series(
        _write(tmp_path, _joint_cols(rotation=False))
    )
    assert all(x.label != "joint_reaction:LS" for x in series[0].wrenches)
    assert "joint_reaction:LS:force" in missing
    assert "joint_reaction:LS:torque" in missing


def test_non_orthonormal_r_names_joint_and_row(tmp_path: Path) -> None:
    bad = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    path = _write(tmp_path, _joint_cols(mats=[RZ90, bad]))
    with pytest.raises(ValueError, match=r"joint LS.*row 1"):
        load_simscape_force_series(path)


def test_rotation_tol_must_be_positive(tmp_path: Path) -> None:
    path = _write(tmp_path, _world_cols())
    with pytest.raises(ValueError, match="rotation_tol"):
        load_simscape_force_series(path, rotation_tol=0.0)
    with pytest.raises(TypeError):
        load_simscape_force_series(123)  # type: ignore[arg-type]


def test_non_finite_value_rejected(tmp_path: Path) -> None:
    cols = _world_cols()
    cols["HipLogs_BaseonHipForceGlobal_1"][0] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        load_simscape_force_series(_write(tmp_path, cols))


def test_channel_table_is_consistent() -> None:
    labels = [s.label for s in SIMSCAPE_FORCE_CHANNELS]
    assert len(labels) == len(set(labels))
    for joint in SIMSCAPE_JOINTS:
        assert f"joint_reaction:{joint}" in labels
    mof = [s for s in SIMSCAPE_FORCE_CHANNELS if s.label.endswith("_mof")]
    assert mof and all(s.force_cols is None for s in mof)
    # BaseonHipForceHipBase is a different frame and must not be used.
    cols = [c for s in SIMSCAPE_FORCE_CHANNELS for c in (s.force_cols or ())]
    assert not any("HipBase" in c for c in cols)


def test_adapter_hook_delegates(tmp_path: Path) -> None:
    path = _write(tmp_path, _world_cols())
    series, missing = SimscapeAdapter().load_force_series(path)
    assert len(series) == 2 and "grip:rh_mof:torque" in missing


# --- real committed trial (software correctness only) -----------------------

needs_trial = pytest.mark.skipif(not TRIAL.exists(), reason="trial CSV absent")


def _real_columns() -> dict[str, np.ndarray]:
    with TRIAL.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


@needs_trial
def test_real_trial_loads_all_joint_reactions() -> None:
    # Logged R is orthonormal only to ~6e-3 (see test below); the default
    # 1e-6 is deliberately NOT changed, the real file needs an explicit bound.
    series, missing = load_simscape_force_series(TRIAL, rotation_tol=1e-2)
    times = series.times_s
    assert len(times) == 31 and all(
        b > a for a, b in zip(times, times[1:], strict=False)
    )
    for joint in SIMSCAPE_JOINTS:
        w = {x.label: x for x in series[5].wrenches}[f"joint_reaction:{joint}"]
        assert w.force_n is not None and w.torque_nm is not None
    assert not [m for m in missing if m.startswith("joint_reaction:")]
    buf = io.BytesIO()
    series.to_npz(buf)
    buf.seek(0)
    assert ForceTorqueSeries.from_npz(buf).to_dict() == series.to_dict()


@needs_trial
def test_real_trial_default_tolerance_rejects_rounded_rotations() -> None:
    with pytest.raises(ValueError, match="not orthonormal"):
        load_simscape_force_series(TRIAL)


@needs_trial
def test_real_rotations_orthonormal_within_logging_precision() -> None:
    """No joint has a logged global counterpart, so assert orthonormality.

    The only local/global pairs (MP couple/hand) use the MP matrix, which maps
    world = R^T @ local, so they cannot confirm the joint R @ v convention.
    """
    d = _real_columns()
    for joint in SIMSCAPE_JOINTS:
        p = f"{joint}Logs_Rotation_Transform"
        r = np.stack(
            [np.stack([d[f"{p}_I{i}{j}"] for j in (1, 2, 3)], 1) for i in (1, 2, 3)],
            1,
        )
        err = np.abs(np.einsum("tji,tjk->tik", r, r) - np.eye(3)).max()
        assert err < 1e-2, joint
        assert np.abs(np.linalg.det(r) - 1).max() < 1e-2, joint
