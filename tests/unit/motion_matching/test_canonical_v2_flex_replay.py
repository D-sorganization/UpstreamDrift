"""Unit tests for canonical-v2 flexible-shaft control replay (#15787 WP5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.club_only.canonical_v2_replay import (
    BallType,
    CanonicalV2ReplayResult,
    FlexShaftReplayRequest,
    ImpactImpulseEstimate,
    ReplayBadge,
    ReplayDriftBudget,
    estimate_bounded_impact_impulse,
    run_canonical_v2_flex_replay,
)
from src.shared.python.motion_matching.club_shaft_chain import (
    flex_frequency_hz,
    flex_ratings,
)

pytestmark = pytest.mark.unit


def _req(
    ball: BallType = BallType.PROV1,
    flex: str = "Stiff",
    budget: ReplayDriftBudget | None = None,
    verified: bool = True,
    disturbance: float = 0.0,
    peak_omega: float = 38.0,
    zero_ff: bool = False,
) -> FlexShaftReplayRequest:
    """Construct a 240 Hz FlexShaftReplayRequest fixture for unit tests."""
    n, dt, idx = 121, 1.0 / 240.0, 60
    t = np.arange(n, dtype=np.float64) * dt
    q, v = np.zeros((n, 2)), np.zeros((n, 2))
    dv = (0.25 if ball is BallType.WIFFLE else 3.20) / (0.20 * 1.156)
    v[0] = [15.0, 2.0]
    for i in range(1, n):
        acc = peak_omega * 1.8 * np.sin(2.0 * np.pi * float(t[i - 1]) / 0.5)
        vn = v[i - 1] + np.array([acc, 0.2 * acc]) * dt
        if i - 1 == idx:
            vn = vn - dv * (v[i - 1] / max(float(np.linalg.norm(v[i - 1])), 1e-9))
        v[i], q[i] = vn, q[i - 1] + 0.5 * (v[i - 1] + vn) * dt
    dist = np.zeros_like(q) if disturbance != 0.0 else None
    if dist is not None:
        dist[61:, 0] = disturbance
    ff = np.zeros_like(q) if zero_ff else None
    return FlexShaftReplayRequest(
        "c", ball, t, q, v, flex, budget, float(t[idx]), ff, verified, dist
    )


def test_identical_pre_impact_gating_for_wiffle_and_prov1() -> None:
    """Pre-impact gating criteria and drift/torque checks match across ball types."""
    b = ReplayDriftBudget()
    rw = run_canonical_v2_flex_replay(_req(BallType.WIFFLE, budget=b))
    rp = run_canonical_v2_flex_replay(_req(BallType.PROV1, budget=b))
    assert rw.pre_impact_passed and rp.pre_impact_passed
    assert rw.pre_impact_max_pos_drift_m == pytest.approx(rp.pre_impact_max_pos_drift_m)
    assert rw.pre_impact_max_rot_drift_rad == pytest.approx(
        rp.pre_impact_max_rot_drift_rad
    )


def test_bounded_impact_impulse_wiffle_vs_prov1_240hz_interval() -> None:
    """Wiffle yields low impulse vs ProV1 higher impulse with 240 Hz interval."""
    w = estimate_bounded_impact_impulse(BallType.WIFFLE, 45.0, 43.8)
    p = estimate_bounded_impact_impulse(BallType.PROV1, 45.0, 29.0)
    assert isinstance(w, ImpactImpulseEstimate)
    assert w.under_resolved_by_240hz and p.under_resolved_by_240hz
    assert w.nominal_impulse_ns < 1.0 and p.nominal_impulse_ns > 2.0
    w0, w1 = w.compatible_impulse_interval_ns
    p0, p1 = p.compatible_impulse_interval_ns
    assert 0.0 <= w0 < w.nominal_impulse_ns < w1 < p0 < p.nominal_impulse_ns < p1


def test_versioned_budget_emits_full_swing_replayed_badge() -> None:
    """Passing pre- and post-impact within versioned budget emits full-swing-replayed."""
    b = ReplayDriftBudget(budget_version="canonical-v2-flex/2.1.0")
    assert b.budget_version == "canonical-v2-flex/2.1.0"
    assert len(b.budget_sha256) == 64
    for ball in (BallType.WIFFLE, BallType.PROV1):
        r = run_canonical_v2_flex_replay(_req(ball, budget=b))
        assert isinstance(r, CanonicalV2ReplayResult)
        assert r.budget.budget_sha256 == b.budget_sha256
        assert r.badge is ReplayBadge.FULL_SWING_REPLAYED
        assert r.badge.value == "full-swing-replayed"
        assert r.pre_impact_passed and r.post_impact_passed


def test_post_impact_drift_or_unverified_emits_pre_impact_badge() -> None:
    """Pre-impact pass + post-impact failure or unverified impact emits pre-impact."""
    rd = run_canonical_v2_flex_replay(_req(disturbance=180.0))
    assert rd.pre_impact_passed and not rd.post_impact_passed
    assert rd.badge is ReplayBadge.PRE_IMPACT_REPLAYED
    assert rd.badge.value == "pre-impact-replayed"
    ru = run_canonical_v2_flex_replay(_req(verified=False))
    assert ru.pre_impact_passed and not ru.impact_verified
    assert ru.badge is ReplayBadge.PRE_IMPACT_REPLAYED


def test_pre_impact_drift_or_torque_ceiling_emits_kinematic_only() -> None:
    """Pre-impact torque ceiling or drift failure emits kinematic-only."""
    rt = run_canonical_v2_flex_replay(_req(budget=ReplayDriftBudget(max_torque_nm=2.0)))
    assert not rt.torque_within_budget and not rt.pre_impact_passed
    assert rt.badge is ReplayBadge.KINEMATIC_ONLY
    assert rt.badge.value == "kinematic-only"
    b_tight = ReplayDriftBudget(pre_impact_pos_tol_m=1e-4)
    rd = run_canonical_v2_flex_replay(
        _req(BallType.WIFFLE, budget=b_tight, peak_omega=55.0, zero_ff=True)
    )
    assert rd.torque_within_budget and not rd.pre_impact_passed
    assert rd.badge is ReplayBadge.KINEMATIC_ONLY


def test_flexible_shaft_validation_deflection_and_recovery_logged() -> None:
    """Flex rating is validated via club_shaft_chain and deflection/recovery tracked."""
    assert "Stiff" in flex_ratings()
    with pytest.raises(ValueError, match="flex_rating"):
        _req(flex="NoodleFlex")
    soft = run_canonical_v2_flex_replay(_req(flex="Ladies"))
    stiff = run_canonical_v2_flex_replay(_req(flex="X-Stiff"))
    assert soft.flex_frequency_hz == pytest.approx(flex_frequency_hz("Ladies"))
    assert stiff.flex_frequency_hz == pytest.approx(flex_frequency_hz("X-Stiff"))
    assert soft.peak_tip_deflection_m > stiff.peak_tip_deflection_m
    assert 0.0 < stiff.post_impact_recovery_fraction <= 1.0
