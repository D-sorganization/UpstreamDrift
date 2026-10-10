"""GCV-20 (#11767): the shared ball-collision impulse (reused by GCV-13).

A ball at rest struck by a clubhead of effective mass ``m`` whose face
centre approaches along the face normal at ``v_n``: momentum conservation
along the normal and the coefficient of restitution ``e`` give the ball
``(1 + e) m v_n / (m + m_b)`` and the club an equal and opposite impulse.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.shared.python.impact_parameters import ball_impact as bi
from src.shared.python.model_appearance.ball import BALL_RADIUS_M

pytestmark = pytest.mark.unit

NORMAL = np.array([0.0, -1.0, 0.0])  # square face, target along native -Y


def _collide(**overrides: object) -> bi.BallCollision:
    face: dict[str, object] = {
        "face_normal": NORMAL,
        "face_velocity_mps": np.array([0.0, -45.0, 0.0]),
        "application_point_m": np.array([0.0, 0.0, 0.02]),
        "effective_mass_kg": 0.20,
    }
    ball_fields = {"ball_mass_kg": "mass_kg", "cor": "cor", "duration_s": "duration_s"}
    ball: dict[str, object] = {}
    timing: dict[str, object] = {"time_s": 1.3, "swing_span_s": (0.0, 1.8)}
    for key, value in overrides.items():
        if key in face:
            face[key] = value
        elif key in ball_fields:
            ball[ball_fields[key]] = value
        elif key == "ball_centre_m":
            ball["centre_m"] = value
        else:
            timing[key] = value
    return bi.collision_impulse(
        bi.FaceContact(**face),  # type: ignore[arg-type]
        ball=bi.BallSpec(**ball),  # type: ignore[arg-type]
        **timing,  # type: ignore[arg-type]
    )


def test_impulse_matches_momentum_and_restitution() -> None:
    m, mb, e, v = 0.20, bi.BALL_MASS_KG, bi.COR_LIMIT, 45.0
    hit = _collide()
    expected = (1.0 + e) * m * mb * v / (m + mb)
    np.testing.assert_allclose(hit.impulse_on_ball_n_s, expected * NORMAL)
    np.testing.assert_allclose(hit.impulse_on_club_n_s, -hit.impulse_on_ball_n_s)
    # Post-collision normal velocities: ball - club = e * approach speed.
    v_club = v - expected / m
    v_ball = expected / mb
    assert v_ball - v_club == pytest.approx(e * v)
    assert hit.club_normal_speed_change_mps == pytest.approx(expected / m)
    assert hit.ball_speed_mps == pytest.approx(v_ball)
    # Total normal momentum is conserved.
    assert m * v == pytest.approx(m * v_club + mb * v_ball)


def test_coordinator_momentum_check_driver_head_drop() -> None:
    """Driver head ~0.2 kg at 51.8 m/s: the club loses ~17.7 m/s."""
    hit = _collide(face_velocity_mps=np.array([0.0, -51.8, 0.0]))
    assert hit.club_normal_speed_change_mps == pytest.approx(17.7, abs=0.2)


def test_only_the_normal_component_is_exchanged() -> None:
    """Frictionless contact: the tangential face velocity carries no impulse."""
    loft = math.radians(30.0)
    normal = np.array([0.0, -math.cos(loft), math.sin(loft)])
    hit = _collide(face_normal=normal, face_velocity_mps=np.array([0.0, -40.0, 0.0]))
    v_n = 40.0 * math.cos(loft)
    assert hit.approach_speed_mps == pytest.approx(v_n)
    unit = hit.impulse_on_ball_n_s / np.linalg.norm(hit.impulse_on_ball_n_s)
    np.testing.assert_allclose(unit, normal, atol=1e-12)


def test_force_is_the_impulse_spread_over_the_contact() -> None:
    hit = _collide(duration_s=5e-4)
    np.testing.assert_allclose(hit.mean_force_on_club_n * 5e-4, hit.impulse_on_club_n_s)


def test_contact_point_reuses_the_shared_ball() -> None:
    centre = np.array([0.1, -0.2, BALL_RADIUS_M])
    hit = _collide(ball_centre_m=centre)
    np.testing.assert_allclose(hit.contact_point_m, centre - BALL_RADIUS_M * NORMAL)
    assert hit.face_to_ball_gap_m == pytest.approx(
        float(np.linalg.norm(np.array([0.0, 0.0, 0.02]) - hit.contact_point_m))
    )
    assert _collide().contact_point_m is None


def test_record_round_trips_through_json() -> None:
    record = _collide(ball_centre_m=np.array([0.0, 0.0, BALL_RADIUS_M])).to_record()
    again = json.loads(json.dumps(record))
    assert again["cor"] == bi.COR_LIMIT
    assert again["ball_mass_kg"] == bi.BALL_MASS_KG
    assert len(again["impulse_on_club_n_s"]) == 3


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("effective_mass_kg", 0.0, "effective_mass_kg"),
        ("effective_mass_kg", float("nan"), "effective_mass_kg"),
        ("ball_mass_kg", -0.04, "ball_mass_kg"),
        ("cor", 0.0, "cor"),
        ("cor", 1.2, "cor"),
        ("duration_s", 0.0, "duration_s"),
        ("time_s", -0.1, "inside the swing"),
        ("time_s", 1.7999, "inside the swing"),
        ("face_normal", np.zeros(3), "face_normal"),
        ("face_velocity_mps", np.array([0.0, 45.0, 0.0]), "moving into the ball"),
        ("application_point_m", np.array([0.0, np.inf, 0.0]), "application_point"),
    ],
)
def test_contracts(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _collide(**{field: value})


def test_swing_span_must_increase() -> None:
    with pytest.raises(ValueError, match="swing_span_s"):
        _collide(swing_span_s=(1.8, 0.0))


def test_cor_of_one_is_allowed() -> None:
    assert _collide(cor=1.0).cor == 1.0
