"""Tests for 3DOF drive-mode switch and paired sensitivity receipts (SAP-4b, #13886)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.loop_nullspace import impedance_spec as isp
from src.shared.python.pendulum_simulator import sensitivity as sens
from src.shared.python.pendulum_simulator.drive_modes import DriveMode, drive_torque

pytestmark = pytest.mark.unit
M = DriveMode
RECEIPT_PATH = (
    Path(__file__).parents[4] / "docs/validation/drive_mode_sensitivity_receipt.json"
)


def _imp(
    t_end: float = 0.2, k_val: float = 25.0, d_val: float = 1.5
) -> isp.JointImpedance:
    return isp.JointImpedance(
        (isp.ImpedanceWindow(0.0, t_end, np.full(3, k_val), np.full(3, d_val)),)
    )


def test_impedance_zero_equals_torque_mode() -> None:
    nom = sens.rollout_nominal_trajectory(t_end=0.05, dt=0.002)
    z_imp = isp.JointImpedance.zero(3)
    s_c, p_c, pw_c, _ = sens.integrate_drive_mode_stm(
        M.TORQUE_CANONICAL, nom, window_s=0.03
    )
    s_i, p_i, pw_i, _ = sens.integrate_drive_mode_stm(
        M.JOINT_IMPEDANCE, nom, z_imp, window_s=0.03
    )
    assert np.array_equal(s_i, s_c)
    assert np.array_equal(p_i, p_c) and np.array_equal(pw_i, pw_c)


def test_nominal_unchanged_under_impedance() -> None:
    nom, imp = (
        sens.rollout_nominal_trajectory(t_end=0.1, dt=0.002),
        _imp(0.1, 40.0, 2.0),
    )
    s_can, _, _, _ = sens.integrate_drive_mode_stm(
        M.TORQUE_CANONICAL, nom, window_s=0.05
    )
    s_imp, _, _, _ = sens.integrate_drive_mode_stm(
        M.JOINT_IMPEDANCE, nom, imp, window_s=0.05
    )
    s_und, _, _, _ = sens.integrate_drive_mode_stm(
        M.TORQUE_UNDAMPED, nom, window_s=0.05
    )
    assert float(np.max(np.abs(s_imp[:, :3] - s_can[:, :3]))) < 1e-12
    assert float(np.max(np.abs(s_und[:, :3] - s_can[:, :3]))) < 1e-12


@pytest.mark.parametrize(
    "mode", [M.TORQUE_UNDAMPED, M.TORQUE_CANONICAL, M.JOINT_IMPEDANCE]
)
def test_stm_matches_finite_difference(mode: DriveMode) -> None:
    dt, t_end = 0.002, 0.04
    nom = sens.rollout_nominal_trajectory(t_end=t_end, dt=dt)
    imp = _imp(t_end, 20.0, 1.2) if mode is M.JOINT_IMPEDANCE else None
    _, phi_var, _, _ = sens.integrate_drive_mode_stm(mode, nom, imp, window_s=0.02)

    def _step(x: np.ndarray, t: float, step_dt: float, ctrl: int) -> np.ndarray:
        return sens.step_drive_mode_rk4(mode, x, ctrl, step_dt, nom, imp)

    x0 = np.concatenate([nom.q[0], nom.v[0]])
    _, tr_fd = sens.propagate_state_transition_matrix(
        _step, x0, list(range(len(nom.times) - 1)), nom.times[:-1], dt, fd_step=1e-6
    )
    assert float(np.linalg.norm(phi_var - tr_fd[-1]) / np.linalg.norm(tr_fd[-1])) < 1e-5


def test_sigma_monotone_in_damping_toy() -> None:
    ds = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    sigmas = [sens.toy_1dof_sigma_max(d, 12.0, 1.0, 0.5) for d in ds]
    for prev_s, next_s in zip(sigmas, sigmas[1:], strict=False):
        assert next_s < prev_s


def test_receipt_fields_present() -> None:
    nom, imp = (
        sens.rollout_nominal_trajectory(t_end=0.2, dt=0.002),
        _imp(0.2, 30.0, 1.5),
    )
    _, _, _, rec = sens.integrate_drive_mode_stm(
        M.JOINT_IMPEDANCE, nom, imp, window_s=0.15
    )
    data = rec.to_dict()
    assert data["drive_mode"] == "joint_impedance"
    assert isp.HYPOTHESIS_ID in data["impedance_hypotheses"]
    assert data["validation_status"] == "UNVALIDATED"
    assert math.isfinite(data["sigma_max_window"]) and data["sigma_max_window"] >= 0.0
    assert math.isfinite(data["ftle_window"]) and data["det_phi_window"] > 0.0
    s_vals = data["singular_values_window"]
    assert len(s_vals) == 6 and all(math.isfinite(s) and s >= 0.0 for s in s_vals)
    assert list(s_vals) == sorted(s_vals, reverse=True)
    for key in ("grad_v_tip", "in_plane_path_lean_proxy"):
        assert math.isfinite(rec.gain_along(key)) and rec.gain_along(key) > 0.0
    assert rec.gain_along(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])) > 0.0

    with pytest.raises(ValueError, match="unknown drive mode"):
        drive_torque("invalid_mode", 0.0, np.zeros(3), np.zeros(3), nom)
    with pytest.raises(ValueError, match="window"):
        bad = isp.JointImpedance(
            (isp.ImpedanceWindow(0.0, 0.5, np.ones(3), np.ones(3)),)
        )
        sens.integrate_drive_mode_stm(M.JOINT_IMPEDANCE, nom, bad)
    with pytest.raises(ValueError, match="stiffness"):
        isp.ImpedanceWindow(0.0, 0.1, np.array([-1.0, 1.0, 1.0]), np.zeros(3))

    assert RECEIPT_PATH.is_file()
    doc = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert (
        doc["validation_status"] == "UNVALIDATED" and len(doc["impedance_k_sweep"]) >= 4
    )
    assert isp.HYPOTHESIS_ID in doc["impedance_hypotheses"]
    assert {m["drive_mode"] for m in doc["modes"]} == {m.value for m in DriveMode}
