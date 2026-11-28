"""TDD unit tests for canonical-v2 inverse dynamics & GRF/CoP gate (WP4, #15787)."""

from __future__ import annotations

from collections.abc import Sequence
import math

import pytest

from src.shared.python.biomechanics.cop_grf_replay_harness import (
    ContactPointRecord as CPR,
)
from src.shared.python.motion_matching.club_only.canonical_v2_dynamics import (
    DynamicsFeasibilityThresholds as Thr,
    FootContactWrench as FCW,
    InverseDynamicsFrame as IDF,
    RepairConfig as RCfg,
    build_irreparable_dynamics_fixture as build_irrep,
    evaluate_inverse_dynamics_gate as eval_gate,
    repair_dynamically_infeasible_trajectory as repair_traj,
)

pytestmark = pytest.mark.unit
_PL = ((0.0, -0.05), (0.24, -0.05), (0.24, 0.05), (0.0, 0.05))
_PR = ((0.0, 0.25), (0.24, 0.25), (0.24, 0.35), (0.0, 0.35))


def _f(idx: int, **kw: object) -> IDF:
    """Build a nominal feasible frame with optional fault overrides."""
    tmap = kw.get("tau") or {"lumbar_L5_S1_flexion": 45.0, "hip_l_extension": 85.0}
    lf = kw.get("lf", (20.0, 5.0, 350.0))
    lm = kw.get("lm", (0.0, 0.0, 0.0))
    rf, rm = kw.get("rf", (1.0, -1.0, 2.0)), kw.get("rm", (0.5, -0.5, 0.5))
    cl = CPR(f"l_{idx}", "foot_l", "left", (0.12, 0.0, 0.0), lf, lm)  # type: ignore[arg-type]
    cr = CPR(f"r_{idx}", "foot_r", "right", (0.12, 0.3, 0.0), (-10.0, 5.0, 350.0))
    ws = (FCW("left", (cl,), _PL), FCW("right", (cr,), _PR))
    q = kw.get("q", (0.0, 0.0, 0.0, 0.0))
    trk, grp = kw.get("trk", 0.01), kw.get("grp", 0.001)
    coupled = kw.get("coupled", True)
    return IDF(idx, 0.01 * idx, q, tmap, rf, rm, ws, trk, grp, coupled)  # type: ignore[arg-type]


def test_feasible_and_breaches() -> None:
    """Verify valid trajectory passes and torque/ADR-0059/GRF/CoP/root breaches fail."""
    ok = eval_gate([_f(i) for i in range(4)], Thr())
    assert ok.passed and ok.failure_reasons == () and ok.cop_excursion_count == 0
    assert ok.min_normal_force_n == pytest.approx(350.0)

    t1 = Thr(tau_max_nm=200.0, trunk_coupled_tau_max_nm=150.0)
    rl = eval_gate([_f(0), _f(1, tau={"hip_l_extension": 240.0})], t1)
    assert not rl.passed and "torque_ceiling_exceeded" in rl.failure_reasons
    assert rl.failing_frame_indices == (1,)
    assert rl.peak_torque_nm == pytest.approx(240.0)
    rt = eval_gate([_f(0, tau={"thoracic_T12_L1_axial": -185.0})], t1)
    assert not rt.passed and "trunk_torque_ceiling_exceeded" in rt.failure_reasons
    assert rt.peak_trunk_torque_nm == pytest.approx(185.0)
    ru = eval_gate([_f(0, coupled=False)], t1)
    assert not ru.passed and "adr0059_uncoupled_trunk" in ru.failure_reasons

    t2 = Thr(friction_mu=0.6, root_force_tol_n=10.0, root_moment_tol_nm=8.0)
    fz = eval_gate([_f(0, lf=(0.0, 0.0, -25.0))], t2)
    assert not fz.passed and "unilateral_normal_force_breach" in fz.failure_reasons
    assert fz.min_normal_force_n == pytest.approx(-25.0)
    mu = eval_gate([_f(0, lf=(60.0, 60.0, 100.0))], t2)
    assert not mu.passed and "friction_cone_breach" in mu.failure_reasons
    assert mu.peak_friction_ratio == pytest.approx(math.hypot(60.0, 60.0) / 100.0)
    cop = eval_gate([_f(0, lf=(0.0, 0.0, 100.0), lm=(0.0, -30.0, 0.0))], t2)
    assert not cop.passed and "cop_outside_support_polygon" in cop.failure_reasons
    assert cop.cop_excursion_count == 1
    rf = eval_gate([_f(0, rf=(9.0, 9.0, 0.0))], t2)
    assert not rf.passed and "root_residual_force_exceeded" in rf.failure_reasons
    assert rf.peak_root_force_n == pytest.approx(math.hypot(9.0, 9.0))
    rm = eval_gate([_f(0, rm=(0.0, 12.0, 0.0))], t2)
    assert not rm.passed and "root_residual_moment_exceeded" in rm.failure_reasons
    assert rm.peak_root_moment_nm == pytest.approx(12.0)


def test_capped_repair_and_tolerance_preservation() -> None:
    """Verify capped repair fixes small breaches preserving tracking & closure."""
    thr = Thr(tau_max_nm=200.0, trunk_coupled_tau_max_nm=150.0, root_force_tol_n=10.0)
    cfg = RCfg(max_iters=12, max_delta_q_rad=0.04, step_size_rad=0.01)

    def _eval(cands: Sequence[IDF]) -> tuple[IDF, ...]:
        """Recompute dynamics for perturbed posture: q[0]=0.025 clears breaches."""

        def _upd(f: IDF) -> IDF:
            """Update frame 1 based on perturbation q[0]."""
            q0 = f.q_rad[0]
            dq, my = abs(q0 - 0.025), -22.0 * max(0.0, (0.025 - q0) / 0.025)
            tau = {"lumbar_L5_S1_flexion": 140.0 + 1200.0 * dq}
            rf, tk = (2.0 + 600.0 * dq, 0.0, 0.0), 0.015 + 0.4 * abs(q0)
            lf, lm = (10.0, 0.0, 200.0), (0.0, my, 0.0)
            return _f(1, q=f.q_rad, tau=tau, lf=lf, lm=lm, rf=rf, trk=tk)

        return tuple(_upd(f) if f.frame_index == 1 else f for f in cands)

    init = _eval(tuple(_f(i) for i in range(3)))
    assert not eval_gate(init, thr).passed
    rep = repair_traj(init, thr, cfg, _eval)
    assert rep.passed and rep.repair_attempted and rep.repair_succeeded
    assert rep.repair_iterations <= cfg.max_iters
    assert rep.max_applied_delta_q_rad <= cfg.max_delta_q_rad + 1e-12
    assert rep.peak_club_tracking_error_m <= cfg.max_club_tracking_error_m
    assert rep.peak_grip_closure_error_m <= cfg.max_grip_closure_error_m

    thr2 = Thr(trunk_coupled_tau_max_nm=150.0, max_club_tracking_error_m=0.02)
    cfg2 = RCfg(max_iters=8, max_delta_q_rad=0.04, max_club_tracking_error_m=0.02)

    def _conflict(cands: Sequence[IDF]) -> tuple[IDF, ...]:
        """Lowering torque necessarily breaks club tracking beyond 0.02 m."""

        def _c(f: IDF) -> IDF:
            """Compute conflicting frame."""
            t = {"lumbar_L5_S1_flexion": max(100.0, 175.0 - 3e3 * abs(f.q_rad[0]))}
            return _f(f.frame_index, q=f.q_rad, tau=t, trk=0.01 + 2.0 * abs(f.q_rad[0]))

        return tuple(_c(f) for f in cands)

    bad = repair_traj(_conflict([_f(0)]), thr2, cfg2, _conflict)
    assert not bad.passed and bad.repair_attempted and not bad.repair_succeeded


def test_irreparable_fixture_strictly_rejected_even_after_repair() -> None:
    """Verify grossly impossible demands remain strictly rejected after repair."""
    thr, cfg = Thr(), RCfg(max_iters=10, max_delta_q_rad=0.05, step_size_rad=0.01)
    fix = build_irrep()
    pre = eval_gate(fix, thr)
    assert not pre.passed and "trunk_torque_ceiling_exceeded" in pre.failure_reasons
    assert "root_residual_force_exceeded" in pre.failure_reasons

    def _bounded(cands: Sequence[IDF]) -> tuple[IDF, ...]:
        """Perturbations within cap alter demands slightly, staying irreparable."""

        def _sc(o: IDF, c: IDF) -> IDF:
            """Scale original impossible loads by perturbation cap."""
            dq = sum(abs(a - b) for a, b in zip(c.q_rad, o.q_rad, strict=True))
            s = max(0.85, 1.0 - 0.5 * dq)
            rf, rm = o.root_residual_force_n, o.root_residual_moment_nm
            taus = {k: v * s for k, v in o.joint_torques_nm.items()}
            rfs = (rf[0] * s, rf[1] * s, rf[2] * s)
            rms = (rm[0] * s, rm[1] * s, rm[2] * s)
            return IDF(
                c.frame_index, c.time_s, c.q_rad, taus, rfs, rms, o.foot_wrenches
            )

        return tuple(_sc(o, c) for o, c in zip(fix, cands, strict=True))

    rep = repair_traj(fix, thr, cfg, _bounded)
    assert not rep.passed and rep.repair_attempted and not rep.repair_succeeded
