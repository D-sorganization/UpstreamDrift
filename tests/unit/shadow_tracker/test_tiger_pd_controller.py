"""Unit tests for Track D bounded-PD free-root MuJoCo controller (#15837)."""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np
import pytest

from src.shared.python.shadow_tracker import tiger_pd_controller as pdc

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]


def _assert_unverified(obj: Any) -> None:
    """Assert unverified timebase and image_observations_only flags."""
    g = obj.__getitem__ if isinstance(obj, dict) else lambda k: getattr(obj, k)
    assert g("timebase") == "unverified" and g("physical_time_verified") is False
    assert g("qualification") == "image_observations_only"


def _swing_targets(m: mujoco.MjModel, n: int = 21) -> np.ndarray:
    """Build smooth Track B-style swing targets with closed bilateral grip."""
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    qp, df, (lid, tid, _) = pdc._model_indices(m)
    tgt, prev, jac = np.zeros((n, m.nu)), np.zeros(m.nu), np.zeros((3, m.nv))
    for k in range(n):
        u = k / max(1, n - 1)
        b, w = math.sin(math.pi * u) ** 2, math.sin(2.0 * math.pi * u)
        q = prev.copy()
        q[:6] = np.array([0.35 * w, 0.05, 0.16 * w, 0.04, -0.05, 0.18 * w]) * b
        for _ in range(25):
            d.qpos[qp] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err = d.site_xpos[lid] - d.site_xpos[tid]
            if float(np.linalg.norm(err)) < 1e-6:
                break
            mujoco.mj_jacSite(m, d, jac, None, tid)
            jt = jac[:, df[6:9]]
            q[6:9] += np.linalg.solve(jt.T @ jt + 1e-4 * np.eye(3), jt.T @ err)
        prev = tgt[k] = q
    return tgt


def test_confidence_gain_schedule() -> None:
    """ConfidenceGainSchedule maps c=0->min, c=1->max monotonically."""
    s = pdc.ConfidenceGainSchedule(kp_min=20.0, kp_max=160.0, kd_min=2.0, kd_max=16.0)
    assert s.gains(0.0) == pytest.approx((20.0, 2.0))
    assert s.gains(1.0) == pytest.approx((160.0, 16.0))
    pts = [s.gains(c) for c in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(
        a < b and c < d for (a, c), (b, d) in zip(pts[:-1], pts[1:], strict=True)
    )
    for bad in (-0.01, 1.01, float("nan")):
        with pytest.raises(ValueError):
            s.gains(bad)
    m = mujoco.MjModel.from_xml_string(pdc.build_free_root_test_mjcf())
    ctrl = pdc.BoundedPdController(pdc.PdControllerConfig(s, tau_limit_nm=150.0))
    tgt, cs = _swing_targets(m, 11), np.linspace(0.0, 1.0, 11)
    tel = pdc.run_pd_rollout(m, tgt, cs, duration_s=0.1, controller=ctrl)
    assert len(tel) > 5 and (tel[0].kp, tel[0].kd) == pytest.approx((20.0, 2.0))
    assert (tel[-1].kp, tel[-1].kd) == pytest.approx((160.0, 16.0))
    assert [x.kp for x in tel] == sorted(x.kp for x in tel)


def test_actuator_saturation_clamping() -> None:
    """Large tracking error clamps torques within [-tau_limit, +tau_limit]."""
    m = mujoco.MjModel.from_xml_string(pdc.build_free_root_test_mjcf())
    lim, s = 4.5, pdc.ConfidenceGainSchedule(kp_min=80.0, kp_max=160.0)
    ctrl = pdc.BoundedPdController(pdc.PdControllerConfig(s, tau_limit_nm=lim))
    tgt, cs = np.full((5, m.nu), 1.5), np.ones(5)
    tel = pdc.run_pd_rollout(m, tgt, cs, duration_s=0.04, controller=ctrl)
    assert tel and max(st.peak_torque_nm for st in tel) == pytest.approx(lim)
    assert all(abs(t) <= lim + 1e-12 for st in tel for t in st.applied_torques_nm)
    assert any(st.saturation_fraction > 0.0 for st in tel)


def test_excessive_grip_drift_downgrades_claim() -> None:
    """Excessive grip drift downgrades claim_label and names phases."""
    m = mujoco.MjModel.from_xml_string(pdc.build_free_root_test_mjcf())
    tgt = _swing_targets(m, 21)
    tgt[13:, 6], tgt[13:, 7] = tgt[13:, 6] + 0.9, tgt[13:, 7] - 0.7
    cfg = pdc.PdControllerConfig(tau_limit_nm=180.0, max_grip_drift_m=0.02)
    ctrl = pdc.BoundedPdController(cfg)
    rep = pdc.run_duration_sweep(m, tgt, (1.05,), controller=ctrl)
    assert isinstance(rep, pdc.TrackDDurationSweepReport)
    assert rep.claim_label == "below_gate"
    _assert_unverified(rep)
    failed = rep.results[0].failing_phases
    assert ("impact" in failed or "finish" in failed) and "address" not in failed


def test_planted_foot_slip_downgrades_claim() -> None:
    """Planted-foot slip on the free-root base downgrades claim_label."""
    m = mujoco.MjModel.from_xml_string(pdc.build_free_root_test_mjcf())
    m.geom_friction[:, 0] = 0.005
    cfg = pdc.PdControllerConfig(tau_limit_nm=180.0, max_foot_slip_m=0.002)
    v0 = np.eye(1, m.nv, dtype=np.float64).ravel() * 0.35
    ctrl, tgt = pdc.BoundedPdController(cfg), _swing_targets(m, 21)
    rep = pdc.run_duration_sweep(m, tgt, (0.85,), controller=ctrl, initial_qvel=v0)
    assert rep.claim_label == "below_gate" and len(rep.failing_phases) > 0
    _assert_unverified(rep)
    assert rep.results[0].max_foot_slip_m > 0.002


def test_duration_sweep_scores_five_phases() -> None:
    """Duration sweep runs closed-loop MuJoCo and scores all 5 phases."""
    m = mujoco.MjModel.from_xml_string(pdc.build_free_root_test_mjcf())
    tgt, durs = _swing_targets(m, 21), (0.85, 1.05, 1.25)
    s = pdc.ConfidenceGainSchedule(kp_min=25.0, kp_max=115.0, kd_min=2.0, kd_max=8.5)
    cfg = pdc.PdControllerConfig(s, 180.0, 0.20, 0.03, 0.025, 0.50)
    cs, ctrl = np.full(len(tgt), 0.9), pdc.BoundedPdController(cfg)
    rep = pdc.run_duration_sweep(m, tgt, durs, confidences=cs, controller=ctrl)
    assert rep.schema_version == "markerless-scorecard/1.0.0"
    assert rep.phase_names == pdc.HELD_OUT_PHASE_NAMES and len(rep.results) == 3
    assert rep.claim_label == "dynamic_feasible_relative"
    assert rep.best_duration_s in durs
    for res in rep.results:
        assert isinstance(res, pdc.DurationSweepResult)
        assert set(res.phase_metrics) == set(pdc.HELD_OUT_PHASE_NAMES)
        _assert_unverified(res)
    p = rep.to_dict()
    assert p["schema_version"] == rep.schema_version
    assert p["claim_label"] == "dynamic_feasible_relative"
    _assert_unverified(p)
