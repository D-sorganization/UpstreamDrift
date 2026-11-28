"""Unit tests for WP-4 sand-ball momentum diagnostics and 4-frame strip (#15805)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bunkershot3d.exceptions import BunkerShot3DValueError
from bunkershot3d.metrics.realism_diagnostics import (
    FourFrameVerificationStrip,
    MomentumCouplingReport,
    VerificationStripFrame,
    build_four_frame_verification_strip as build_strip,
    check_sand_ball_momentum_coupling as check_coupling,
    evaluate_bunker_realism_targets as eval_targets,
)
from bunkershot3d.solvers.envelope import EnvelopeStatus
from bunkershot3d.solvers.protocol import FidelityTier

pytestmark = pytest.mark.unit
_WP0_REL = "docs/bunkershot3d/baselines/wp0_f0_reference.json"
_WP0 = Path(__file__).resolve().parents[2] / _WP0_REL


def _frames(
    tier: str, verdict: str, elev: float = 12.0, azim: float = 180.0
) -> list[VerificationStripFrame]:
    """Build a valid 4-phase frame sequence with consistent camera and tier."""
    s = (("address", 0.0), ("sand_entry", 0.02), ("release", 0.05), ("landing", 1.25))
    return [VerificationStripFrame(p, t, elev, azim, tier, verdict) for p, t in s]


def test_momentum_coupling_passes_and_integrates_sources() -> None:
    """Matching ball momentum gain passes for boundary, particle, and force trace."""
    rep = check_coupling((0.45, 0.30), (0.45, 0.30))
    assert isinstance(rep, MomentumCouplingReport) and rep.within_tolerance
    assert rep.imbalance_magnitude_Ns == pytest.approx(0.0)
    part = np.array([[0.15, 0.10], [0.30, 0.20]], dtype=float)
    assert check_coupling((0.45, 0.30), sand_particle_impulses=part).within_tolerance
    times = np.array([0.0, 0.01, 0.02], dtype=float)
    forces = np.array([[0.0, 0.0], [45.0, 30.0], [0.0, 0.0]], dtype=float)
    rt = check_coupling((0.45, 0.30), boundary_force_N=forces, time_s=times)
    assert rt.within_tolerance and rt.integrated_impulse_Ns == pytest.approx(
        (0.45, 0.3)
    )


def test_momentum_coupling_fails_closed_on_imbalance() -> None:
    """Momentum imbalance raises fail-closed and reports when fail_closed=False."""
    with pytest.raises(BunkerShot3DValueError, match="imbalance"):
        check_coupling((0.60, 0.30), (0.45, 0.30))
    diag = check_coupling((0.60, 0.30), (0.45, 0.30), fail_closed=False)
    assert not diag.within_tolerance and diag.imbalance_magnitude_Ns == pytest.approx(
        0.15
    )
    with pytest.raises(BunkerShot3DValueError, match="mismatch"):
        check_coupling((0.45, 0.30), (0.45, 0.30), sand_particle_impulses=[[0.1, 0.1]])
    with pytest.raises(BunkerShot3DValueError):
        check_coupling((np.nan, 0.30), (0.45, 0.30))


def test_evaluate_realism_targets_rejects_wp0_baseline_defect() -> None:
    """Committed WP-0 0.578 m carry fails realism while preserving verdict_status."""
    s = json.loads(_WP0.read_text(encoding="utf-8"))["reported_summary"]
    tier = str(s["fidelity_tier"])
    ev = eval_targets(
        35.0, 5500.0, s["carry_m"], s["verdict_status"], fidelity_tier=tier
    )
    assert s["carry_m"] == pytest.approx(0.578)
    assert not ev.passed and not ev.carry_ok and ev.launch_angle_ok and ev.backspin_ok
    assert ev.verdict_status is EnvelopeStatus.BEYOND_VALIDATION
    assert ev.fidelity_tier is FidelityTier.F0 and "carry" in ev.failures[0]


def test_evaluate_realism_targets_enforces_angle_spin_and_refusal() -> None:
    """Angle outside 30-45 deg, spin outside 4000-8000 rpm, or REFUSED fail."""
    cases: tuple[tuple[float, float, EnvelopeStatus | str], ...] = (
        (28.0, 6000.0, "beyond_validation"),
        (47.0, 6000.0, "beyond_validation"),
        (36.0, 3500.0, "beyond_validation"),
        (36.0, 8500.0, "beyond_validation"),
        (36.0, 6000.0, EnvelopeStatus.REFUSED),
    )
    for angle, spin, verdict in cases:
        assert not eval_targets(angle, spin, 12.0, verdict).passed


def test_evaluate_realism_targets_passes_realistic_shot_preserving_verdict() -> None:
    """Realistic targets pass while ADR-0032/0033 verdict remains explicit."""
    ev = eval_targets(36.0, 6000.0, 12.0, "beyond_validation", fidelity_tier="F1")
    assert ev.passed and ev.failures == () and ev.launch_angle_ok and ev.backspin_ok
    assert ev.carry_ok and ev.fidelity_tier is FidelityTier.F1
    assert ev.verdict_status is EnvelopeStatus.BEYOND_VALIDATION


def test_build_four_frame_strip_succeeds_and_stamps_metadata() -> None:
    """Matched 4-phase frames assemble canonically with tier/verdict metadata."""
    base = list(reversed(_frames("F0", "beyond_validation")))
    cand = _frames("F1", "beyond_validation", azim=-180.0)
    strip = build_strip(base, cand)
    assert isinstance(strip, FourFrameVerificationStrip)
    assert strip.phases == ("address", "sand_entry", "release", "landing")
    assert tuple(f.phase for f in strip.baseline_frames) == strip.phases
    assert tuple(f.phase for f in strip.candidate_frames) == strip.phases
    assert strip.baseline_tier is FidelityTier.F0
    assert strip.candidate_tier is FidelityTier.F1
    assert strip.overall_verdict_status is EnvelopeStatus.BEYOND_VALIDATION
    m = strip.metadata
    assert (m["baseline_tier"], m["candidate_tier"]) == ("F0", "F1")
    assert m["overall_verdict_status"] == "beyond_validation"


def test_build_four_frame_strip_fails_on_missing_phase_or_camera_mismatch() -> None:
    """Missing phases, camera mismatch, or bad timestamps fail closed."""
    base, cand = _frames("F0", "beyond_validation"), _frames("F1", "beyond_validation")
    with pytest.raises(BunkerShot3DValueError, match="phase"):
        build_strip(base[:-1], cand)
    with pytest.raises(BunkerShot3DValueError, match="elevation_deg"):
        build_strip(base, _frames("F1", "beyond_validation", elev=20.0))
    with pytest.raises(BunkerShot3DValueError, match="azimuth_deg"):
        build_strip(base, _frames("F1", "beyond_validation", azim=90.0))
    cls, bad = VerificationStripFrame, list(cand)
    bad[0] = cls("address", 2.0, 12.0, 180.0, "F1", "beyond_validation")
    with pytest.raises(BunkerShot3DValueError, match="monotonic"):
        build_strip(base, bad)
