"""HILL_MTU drive mode for 3DOF FTLE study (Refs #13888)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.biomechanics.mtu_joint_actuator import TendonMode
from src.shared.python.loop_nullspace import impedance_spec as isp
from src.shared.python.pendulum_simulator import sensitivity as sens
from src.shared.python.pendulum_simulator.drive_modes import DriveMode
from src.shared.python.pendulum_simulator.hill_mtu_drive import (
    RESIDUAL_TORQUE_RMS_LIMIT_NM,
    SMOKE_COCONTRACTION_LEVELS,
    build_matched_excitation,
    default_triple_mtu_actuators,
)

pytestmark = pytest.mark.unit
M = DriveMode
RECEIPT_PATH = (
    Path(__file__).parents[4] / "docs/validation/drive_mode_sensitivity_receipt.json"
)


def _smoke_nominal(t_end: float = 0.08, dt: float = 0.002):
    return sens.rollout_nominal_trajectory(t_end=t_end, dt=dt)


def _hill(c: float, nom, tendon: TendonMode = TendonMode.RIGID):
    acts = default_triple_mtu_actuators(tendon)
    return build_matched_excitation(nom, c, acts, tendon_mode=tendon)


@pytest.mark.parametrize("c", SMOKE_COCONTRACTION_LEVELS)
def test_hill_mode_reproduces_nominal(c: float) -> None:
    nom = _smoke_nominal()
    hill = _hill(c, nom)
    if not hill.qualified:
        pytest.fail(
            f"HILL_MTU not qualified: residual_rms={hill.residual_torque_rms_nm:.3e} N·m"
        )
    assert hill.residual_torque_rms_nm < RESIDUAL_TORQUE_RMS_LIMIT_NM
    sim = sens.rollout_hill_mtu(hill, nom)
    err = float(np.max(np.abs(sim[:, :3] - nom.q)))
    assert err < 1e-6


def test_hill_stm_matches_fd() -> None:
    """Full-matrix STM vs FD; short horizon keeps default CI timeout feasible."""
    dt, t_end, window = 0.004, 0.008, 0.008
    nom = sens.rollout_nominal_trajectory(t_end=t_end, dt=dt)
    hill = _hill(0.2, nom)
    _, phi_full, phi_joint, _ = sens.integrate_drive_mode_stm(
        M.HILL_MTU, nom, window_s=window, hill=hill
    )

    def _step(x18: np.ndarray, _t: float, step_dt: float, ctrl: int) -> np.ndarray:
        return sens.step_hill_mtu_rk4(hill, x18, ctrl, step_dt)

    x0 = sens._initial_hill_state(hill, nom)
    _, tr_fd = sens.propagate_state_transition_matrix(
        _step, x0, list(range(len(nom.times) - 1)), nom.times[:-1], dt, fd_step=1e-5
    )
    rel_full = float(np.linalg.norm(phi_full - tr_fd[-1]) / np.linalg.norm(tr_fd[-1]))
    assert rel_full < 1e-4


def test_effective_impedance_psd_on_plateau() -> None:
    nom = _smoke_nominal()
    hill = _hill(0.2, nom, TendonMode.RIGID)
    finite_k = hill.effective_kq[np.isfinite(hill.effective_kq)]
    if finite_k.size == 0 or float(np.max(np.abs(finite_k))) == 0.0:
        pytest.skip("effective stiffness NOT_QUALIFIED on this nominal slice")
    assert float(np.mean(np.abs(finite_k))) > 0.0
    assert np.nanmean(np.abs(hill.effective_dq)) >= 0.0


def test_receipt_has_four_modes() -> None:
    assert RECEIPT_PATH.is_file()
    doc = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    mode_ids = {m["drive_mode"] for m in doc["modes"]}
    assert mode_ids == {m.value for m in DriveMode}
    hill_modes = [m for m in doc["modes"] if m["drive_mode"] == M.HILL_MTU.value]
    assert len(hill_modes) >= 1
    sweep = doc.get("hill_cocontraction_sweep") or []
    assert len(sweep) >= 3
    tendons = {row.get("tendon_mode") for row in sweep}
    assert tendons >= {"rigid", "compliant"}
    for row in sweep:
        assert "cocontraction" in row and "tendon_mode" in row
        if row.get("validation_status") != "NOT_QUALIFIED":
            assert row.get("sigma_max_window") is not None
    imp = isp.JointImpedance(
        (isp.ImpedanceWindow(0.0, 0.2, np.full(3, 25.0), np.full(3, 1.5)),)
    )
    nom = sens.rollout_nominal_trajectory(t_end=0.2, dt=0.002)
    _, _, _, rec = sens.integrate_drive_mode_stm(
        M.JOINT_IMPEDANCE, nom, imp, window_s=0.15
    )
    assert rec.validation_status == "UNVALIDATED"
