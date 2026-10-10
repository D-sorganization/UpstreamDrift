"""#12039: the Drake and MuJoCo marker IK solve the same problem.

Both plants have identical forward kinematics, so for the same inputs every
residual row and its Jacobian must agree, and ``solve_pose`` from the same
start must reach the same fit. The dual-grip weld rotation row was the gap:
Drake used the small-angle ``vee(R_a R_b^T - R_b R_a^T) / 2`` with the
Jacobian ``J_a - J_b``, which is exact only at zero rotation error, while
MuJoCo uses the principal log and its exact derivative.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.contact_law import GroundPlane

pytestmark = pytest.mark.unit
pytest.importorskip("mujoco")
pytest.importorskip("pydrake")

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
GROUND = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)
#: Issue #12039 acceptance: same start, same fixture frame, same RMS.
SOLVE_RMS_PARITY_M = 0.5e-3


@pytest.fixture(scope="module")
def kins() -> dict:
    from src.shared.python.motion_matching.pipeline.plant import get_plant

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    attachments = {
        label: (a["body"], tuple(a["offset_m"]))
        for label, a in spec["marker_attachments"].items()
        if a["offset_m"] is not None
    }
    out = {e: get_plant(e, spec).create_ik(attachments) for e in ("mujoco", "drake")}
    assert list(out["mujoco"].coordinate_order) == list(out["drake"].coordinate_order)
    return out


def _random_pose(kin, seed: int, spread: float = 0.25) -> np.ndarray:
    q = np.zeros(len(kin.coordinate_order))
    q[6:] = np.random.default_rng(seed).normal(0.0, spread, len(q) - 6)
    return q


def _closure_rows(kin, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kin._set(q)
    rows: list[np.ndarray] = []
    jacs: list[np.ndarray] = []
    kin._append_closure(rows, jacs, 1.0, 1.0)
    return np.concatenate(rows), np.vstack(jacs)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_closure_rows_and_jacobians_match_mujoco(kins: dict, seed: int) -> None:
    q = _random_pose(kins["mujoco"], seed)
    r_mj, j_mj = _closure_rows(kins["mujoco"], q)
    r_dk, j_dk = _closure_rows(kins["drake"], q)
    assert np.degrees(np.linalg.norm(r_mj[3:])) > 10.0  # a real weld mismatch
    np.testing.assert_allclose(r_dk, r_mj, atol=1e-9)
    np.testing.assert_allclose(j_dk, j_mj, atol=1e-7)


def test_drake_closure_jacobian_matches_finite_differences(kins: dict) -> None:
    kin = kins["drake"]
    q = _random_pose(kin, 4)
    _, jac = _closure_rows(kin, q)
    h = 1e-6
    numeric = np.zeros_like(jac)
    for i in range(len(q)):
        step = np.zeros(len(q))
        step[i] = h
        numeric[:, i] = (
            _closure_rows(kin, q + step)[0] - _closure_rows(kin, q - step)[0]
        ) / (2 * h)
    np.testing.assert_allclose(jac, numeric, atol=1e-6)


def test_closure_error_reports_the_principal_angle(kins: dict) -> None:
    q = _random_pose(kins["mujoco"], 5)
    pos_mj, rot_mj = kins["mujoco"].closure_error(q)
    pos_dk, rot_dk = kins["drake"].closure_error(q)
    assert pos_dk == pytest.approx(pos_mj, abs=1e-9)
    assert rot_dk == pytest.approx(rot_mj, abs=1e-9)


def test_solve_pose_from_the_same_start_reaches_the_same_fit(kins: dict) -> None:
    """Acceptance (#12039): same start on a fixture frame, RMS within 0.5 mm."""
    truth = _random_pose(kins["mujoco"], 6, spread=0.15)
    targets = kins["mujoco"].marker_positions(truth)
    valid = np.ones(len(targets), dtype=bool)
    start = truth.copy()
    start[6:] += np.random.default_rng(7).normal(0.0, 0.1, len(start) - 6)
    fits = {
        e: k.solve_pose(targets, valid, start, ground=GROUND, iterations=100)
        for e, k in kins.items()
    }
    gap = abs(fits["drake"].marker_rms_m - fits["mujoco"].marker_rms_m)
    assert gap <= SOLVE_RMS_PARITY_M, {e: f.marker_rms_m for e, f in fits.items()}
