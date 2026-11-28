"""Unit tests for canonical-v2 closed-chain constrained IK gate (Refs #15787)."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import src.shared.python.motion_matching.club_only.canonical_v2_ik as ik
import src.shared.python.motion_matching.club_only.hand_geometry as hg

pytestmark = pytest.mark.unit
Q0 = np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0])
_OFF = hg.ModelHandFrameOffsets(
    "s8", hg.GolferHandedness.RIGHT, "L", "R", 0.04, -0.04, "h"
)


def _rot(a: float) -> tuple[np.ndarray, np.ndarray]:
    """Return unit quaternion and SO(3) rotation matrix for angle about +Y."""
    c, s = math.cos(a), math.sin(a)
    q = np.array([math.cos(0.5 * a), 0.0, math.sin(0.5 * a), 0.0])
    return q, np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class _Kin:
    """Analytic 8-DOF synthetic closed-chain golfer + club kinematics fixture."""

    def __init__(self, **b: float) -> None:
        """Store optional defect biases (lb, tb, grb, hrb, hpb)."""
        self.b = b

    def forward_kinematics(self, q: np.ndarray) -> ik.ClosedChainFrameState:
        """Evaluate club grip/head, bilateral hand sites, and foot sites at q."""
        g, th, b = q[0:3].copy(), float(q[3]), self.b
        hq, r = _rot(th)
        lx = g + 0.04 * r[:, 2] + np.array([q[4] + b.get("lb", 0.0), 0.0, 0.0])
        tx = g - 0.04 * r[:, 2] + np.array([0.0, q[5] + b.get("tb", 0.0), 0.0])
        hx = g + r @ np.array([b.get("hpb", 0.0), 0.0, -1.0])
        s0, s1 = np.array([q[7], 0.3, 0.0]), np.array([0.0, -0.3, float(q[6])])
        gq = _rot(th + b.get("grb", 0.0))[0]
        h_q = _rot(th + b.get("hrb", 0.0))[0]
        return ik.ClosedChainFrameState(
            g, gq, hx, h_q, lx, hq, tx, hq, {"lead_foot": s0, "trail_heel": s1}
        )


def _run(
    n: int, dt: float, kin: _Kin, q0: np.ndarray = Q0, x_end: float = 0.04, **kw: Any
) -> ik.CanonicalV2IkResult:
    """Build targets + config and run closed-chain IK solver."""
    th = 0.25 * (s := 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, n))))
    gx = np.column_stack([x_end * s, np.zeros(n), np.full(n, 0.90)])
    gq = np.vstack([_rot(float(a))[0] for a in th])
    hx = gx - np.array([_rot(float(a))[1][:, 2] for a in th])
    lo, hi = kw.pop("lo", np.full(8, -2.0)), kw.pop("hi", np.full(8, 2.0))
    qd, qdd = kw.pop("qd", np.full(8, 50.0)), kw.pop("qdd", np.full(8, 5000.0))
    lims = ik.JointKinematicLimits(lo, hi, qd, qdd)
    tols = kw.pop("tols", None) or ik.GripHeadTolerances()
    cfg = ik.ClosedChainIkConfig(tols, lims, _OFF, **kw)
    tgt = ik.ClubTargetPoses(gx, gq, hx, gq.copy())
    return ik.solve_canonical_v2_closed_chain_ik(cfg, tgt, q0, dt, kin)


def _fp(k: str, s: int, e: int, sl: float, lf: float) -> object:
    """Construct a FootContactPhaseSpec for lead_foot (l) or trail_heel (t)."""
    nm, y = ("lead_foot", 0.3) if k == "l" else ("trail_heel", -0.3)
    return ik.FootContactPhaseSpec(nm, s, e, np.array([0.0, y, 0.0]), sl, lf)


def test_clean_closed_chain_trajectory_passes_as_kinematic_preview() -> None:
    """Clean synthetic closed-chain swing solves cleanly and stays preview-only."""
    p = (_fp("l", 0, 9, 5e-3, 5e-3), _fp("t", 0, 5, 5e-3, 5e-3))
    ph = (*p, _fp("t", 5, 9, 0.015, 0.08))
    q0 = np.tile(Q0, (9, 1))
    q0[:, 4], q0[:, 5], q0[5:, 6] = 0.01, -0.01, 0.03
    res = _run(9, 0.02, _Kin(), q0, foot_phases=ph, tour_prior_q=Q0)
    assert res.passed and res.kinematic_preview_only and res.failure_reasons == ()
    assert res.zero_kinematic_limit_violations and res.zero_tolerance_violations
    assert res.per_frame_residuals["grip_translation_m"].shape == (9,)
    assert res.per_frame_residuals["bilateral_closure_pos_m"].max() < 0.005
    with pytest.raises(ValueError, match="kinematic_preview_only"):
        replace(res, kinematic_preview_only=False)


def test_joint_position_velocity_acceleration_violations_fail_closed() -> None:
    """Position, velocity, and acceleration limit breaches each fail closed."""
    hi, qd, qdd = np.full(8, 2.0), np.full(8, 50.0), np.full(8, 5000.0)
    hi[0], qd[0], qdd[0] = 0.05, 0.2, 1.0
    rp = _run(7, 0.02, _Kin(), x_end=0.12, hi=hi)
    assert not rp.passed and rp.violation_counts["position"] > 0
    assert not rp.zero_kinematic_limit_violations
    assert _run(7, 0.01, _Kin(), x_end=0.12, qd=qd).violation_counts["velocity"] > 0
    ra = _run(7, 0.01, _Kin(), x_end=0.12, qdd=qdd)
    assert not ra.passed and ra.violation_counts["acceleration"] > 0


@pytest.mark.parametrize("kw", [{"lb": 0.02}, {"tb": 0.02}])
def test_bilateral_grip_closure_fails_closed(kw: dict[str, float]) -> None:
    """Excessive lead or trail hand closure error fails closed with counts."""
    lo, hi = np.full(8, -2.0), np.full(8, 2.0)
    lo[4:6], hi[4:6] = 0.0, 0.0
    res = _run(5, 0.02, _Kin(**kw), lo=lo, hi=hi)
    assert not res.passed and res.violation_counts["bilateral_closure"] == 5
    assert not res.zero_tolerance_violations
    assert np.all(res.per_frame_residuals["bilateral_closure_pos_m"] >= 0.019)


def test_separate_grip_and_head_tolerances_fail_independently() -> None:
    """Head passing while grip fails (and vice versa) fails closed with split residuals."""
    tols = ik.GripHeadTolerances(0.005, 0.02, 0.010, 0.02)
    c1, c2 = np.full(6, 200.0), np.r_[10.0, 10.0, 10.0, 1.0, 1.0, 1.0]
    c3 = np.r_[200.0, 200.0, 200.0, 1.0, 1.0, 1.0]
    k1 = {"tols": tols, "grip_cov_inv_sqrt": c1, "head_cov_inv_sqrt": c2}
    vh = _run(5, 0.02, _Kin(hrb=0.1, hpb=0.03), **k1).violation_counts
    assert vh["head_rotation"] == 5 and vh["head_translation"] == 5
    assert vh["grip_rotation"] == 0 and vh["grip_translation"] == 0
    k2 = {"tols": tols, "grip_cov_inv_sqrt": c3, "head_cov_inv_sqrt": c1}
    vg = _run(5, 0.02, _Kin(grb=0.1), **k2).violation_counts
    assert vg["grip_rotation"] == 5 and vg["head_rotation"] == 0


def test_phase_specific_foot_contact_slip_and_heel_rise_gate() -> None:
    """Planted foot slip fails closed; legal supported trail-heel rise passes."""
    lo, hi = np.full(8, -2.0), np.full(8, 2.0)
    lo[7], hi[7] = 0.04, 0.04
    rs = _run(6, 0.02, _Kin(), lo=lo, hi=hi, foot_phases=(_fp("l", 0, 6, 5e-3, 5e-3),))
    assert not rs.passed and rs.violation_counts["foot_contact"] == 6
    assert np.all(rs.per_frame_residuals["foot_slip_m"] >= 0.039)
    q_h = Q0 + 0.04 * np.eye(8)[6]
    rh = _run(6, 0.02, _Kin(), q_h, foot_phases=(_fp("t", 0, 6, 0.01, 0.08),))
    assert rh.passed and rh.violation_counts["foot_contact"] == 0
