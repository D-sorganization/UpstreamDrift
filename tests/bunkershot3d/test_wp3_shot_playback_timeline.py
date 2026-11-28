"""WP-3 tests for the shared read-only ShotPlaybackTimeline contract (Refs #15805)."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from bunkershot3d.fields import playback_timeline as pt
from bunkershot3d.fields.schema import ParticleKind
from bunkershot3d.solvers.envelope import EnvelopeStatus
from bunkershot3d.solvers.protocol import FidelityTier
from src.shared.python.core.physics_constants import GRAVITY_M_S2

pytestmark = pytest.mark.unit
_ID = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_F0, _F1 = FidelityTier.F0, FidelityTier.F1
_BV, _P, _B = EnvelopeStatus.BEYOND_VALIDATION, pt.ShotPhase, pt.BallState


def _snap(t: float, x: float = 0.0) -> pt.ParticleSnapshot:
    """Build a deterministic synchronized MPM particle snapshot."""
    p = np.array([[x, 0.0, -0.004], [x + 0.01, 0.0, 0.002]], dtype=np.float64)
    v = np.array([[1.0, 0.0, 0.5], [3.0, 0.0, 2.5]], dtype=np.float64)
    return pt.ParticleSnapshot(t, p, v, ParticleKind.MATERIAL_POINT)


def _fr(t: float, ph: pt.ShotPhase, b: pt.BallState, x: float, s: Any = None) -> Any:
    """Construct a compact TimelineFrame for test fixtures."""
    return pt.TimelineFrame(0, t, ph, b, pt.ClubheadPose((x, 0.0, 0.01), _ID), s)


def _segments(*, particles: bool = False) -> tuple[Any, Any, Any]:
    """Construct continuous address->sand->flight->rollout segments."""
    z0 = pt.BALL_DIAMETER_M / 2.0 - pt.DEFAULT_ADDRESS_BURIAL_DEPTH_M
    sn = [_snap(0.0), _snap(0.005, 0.01), _snap(0.01)] if particles else [None] * 3
    b1 = _B((0.01, 0.0, 0.02), (4.0, 0.0, 4.0), burial_depth_m=0.001)
    imp = [
        _fr(0.0, _P.ADDRESS, _B((0.0, 0.0, z0), burial_depth_m=0.005), -0.08, sn[0]),
        _fr(0.005, _P.SAND_IMPACT, b1, -0.01, sn[1]),
        _fr(0.010, _P.SAND_EXIT, _B((0.035, 0.0, 0.045), (6.0, 0.0, 6.0)), 0.06, sn[2]),
    ]
    t0, g = 0.010, float(GRAVITY_M_S2)
    tau = (6.0 + math.sqrt(36.0 + 2.0 * g * 0.045)) / g
    b_land = _B((0.035 + 6.0 * tau, 0.0, 0.0), (6.0, 0.0, 6.0 - g * tau))
    flt = [
        _fr(t0, _P.FLIGHT, imp[-1].ball, 0.06),
        _fr(t0 + tau, _P.FLIGHT, b_land, 0.3),
    ]
    lp = b_land.position_m
    rol = [
        _fr(t0 + tau, _P.GREEN_ROLLOUT, _B(lp, (2.0, 0.0, 0.0)), 0.3),
        _fr(t0 + tau + 0.5, _P.GREEN_ROLLOUT, _B((lp[0] + 0.5, 0.0, 0.0)), 0.3),
    ]
    return imp, flt, rol


def _build(imp: Any, flt: Any, rol: Any, tier: Any, status: Any) -> Any:
    """Build a timeline using shared test defaults."""
    return pt.build_shot_playback_timeline(
        impact_frames=imp,
        flight_frames=flt,
        rollout_frames=rol,
        fidelity_tier=tier,
        verdict_status=status,
        verdict_summary="ADR-0032/33",
    )


def test_address_visibility_and_continuous_phases() -> None:
    """Ball starts at 5 mm buried address depth and stays continuous through rollout."""
    assert math.isclose(pt.BALL_DIAMETER_M, 0.04267, abs_tol=1e-9)
    assert math.isclose(pt.DEFAULT_ADDRESS_BURIAL_DEPTH_M, 0.005, abs_tol=1e-9)
    tl = _build(*_segments(particles=False), _F0, _BV)
    f0 = tl.frame(0)
    assert f0.phase is _P.ADDRESS and f0.ball.is_visible_above_sand
    exp_z = pt.BALL_DIAMETER_M / 2.0 - 0.005
    assert math.isclose(f0.ball.position_m[2], exp_z, abs_tol=1e-9)
    assert {f.phase for f in tl.frames} == set(_P)
    assert not tl.ball_positions_m().flags.writeable
    with pytest.raises(FrozenInstanceError):
        f0.time_s = 99.0  # type: ignore[misc]


def test_handoff_continuity_rejects_one_mm_or_larger_jump() -> None:
    """Impact-to-flight handoff enforces < 1 mm continuity and fails closed on >= 1 mm."""
    assert math.isclose(pt.MAX_HANDOFF_JUMP_M, 1e-3, abs_tol=1e-12)
    imp, flt, rol = _segments(particles=False)
    p, v, t = flt[0].ball.position_m, flt[0].ball.velocity_m_s, flt[0].time_s
    ok_b, bad_b = _B((p[0] + 5e-4, 0.0, p[2]), v), _B((p[0] + 1e-3, 0.0, p[2]), v)
    assert pt.verify_handoff_continuity(imp[-1], _fr(t, _P.FLIGHT, ok_b, 0.06)) < 1e-3
    bad_f = [_fr(t, _P.FLIGHT, bad_b, 0.06), *flt[1:]]
    with pytest.raises(ValueError, match="jump"):
        pt.verify_handoff_continuity(imp[-1], bad_f[0])
    with pytest.raises(ValueError):
        _build(imp, bad_f, rol, _F0, _BV)


def test_tier_honesty_refuses_cosmetic_f0_spray() -> None:
    """F0 timelines must reject any attached MPM particle snapshots."""
    with pytest.raises(ValueError, match="F0"):
        _build(*_segments(particles=True), _F0, _BV)


def test_f1_requires_synchronized_particle_arrays() -> None:
    """F1 timelines require particle snapshots synchronized to frame timestamps."""
    with pytest.raises(ValueError, match="F1"):
        _build(*_segments(particles=False), _F1, _BV)
    imp, flt, rol = _segments(particles=True)
    with pytest.raises(ValueError, match="desynchronized"):
        _fr(0.0, _P.ADDRESS, imp[0].ball, -0.08, _snap(0.002, 0.0))
    s = _build(imp, flt, rol, _F1, _BV).frame(0).particles
    assert s is not None and not s.positions_m.flags.writeable
    assert not s.velocities_m_s.flags.writeable


def test_validity_verdict_and_deterministic_sha256_json_roundtrip() -> None:
    """Verdict stays explicitly visible and SHA-256 provenance survives JSON round-trip."""
    seg = _segments(particles=True)
    tl_a, tl_b = _build(*seg, "F1", "beyond_validation"), _build(*seg, _F1, _BV)
    assert tl_a.verdict_status is _BV and tl_a.source_digest == tl_b.source_digest
    assert len(tl_a.source_digest) == 64
    encoded = tl_a.to_json()
    payload = json.loads(encoded)
    assert payload["verdict_status"] == "beyond_validation"
    assert pt.ShotPlaybackTimeline.from_json(encoded) == tl_a
    payload["verdict_status"] = "within"
    with pytest.raises(ValueError, match="digest"):
        pt.ShotPlaybackTimeline.from_dict(payload)


def test_clubhead_pose_rejects_axis_inversion() -> None:
    """ClubheadPose refuses reflection / axis-inverted orientation matrices."""
    bad = ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="rotation"):
        pt.ClubheadPose((0.0, 0.0, 0.0), bad)
