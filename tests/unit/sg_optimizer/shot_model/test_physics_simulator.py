"""Impact-condition -> landing adapter over ``SwingBallFlightPipeline`` (#14567)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.sg_optimizer.shot_model.physics_simulator import (
    CLUB_LOFT_DEG,
    REPRESENTATIVE_CLUB,
    PipelineLandingSimulator,
    pipeline_club,
    swing_state_from_impact,
)
from src.shared.python.sg_optimizer.shot_model.skill_family import SKILL_NAMES

pytestmark = pytest.mark.unit

_ZERO = np.array([100.0, 0.0, 0.0, -2.0, 0.0, 0.0])


def test_every_skill_has_a_loft() -> None:
    assert set(CLUB_LOFT_DEG) == set(SKILL_NAMES)


def test_swing_state_follows_gcv15_signs() -> None:
    """+path in-to-out -> clubhead moves right (-y); +face open -> normal right."""
    base = swing_state_from_impact(_ZERO, "driver")
    path = swing_state_from_impact(_ZERO + [0, 0, 3.0, 0, 0, 0], "driver")
    face = swing_state_from_impact(_ZERO + [0, 3.0, 0, 0, 0, 0], "driver")
    aoa = swing_state_from_impact(_ZERO + [0, 0, 0, 4.0, 0, 0], "driver")
    assert base.clubhead_velocity[1] == pytest.approx(0.0, abs=1e-9)
    assert path.clubhead_velocity[1] < 0.0
    assert face.clubhead_orientation[1] < 0.0
    assert aoa.clubhead_velocity[2] > 0.0
    assert np.linalg.norm(base.clubhead_velocity) == pytest.approx(100.0 * 0.44704)
    assert np.linalg.norm(base.clubhead_orientation) == pytest.approx(1.0)


def test_strike_offsets_map_to_metres() -> None:
    s = swing_state_from_impact(_ZERO + [0, 0, 0, 0, 10.0, -5.0], "driver")
    np.testing.assert_allclose(s.impact_offset, [0.010, -0.005])


def test_real_pipeline_driver_rest_exceeds_carry() -> None:
    from src.shared.python.physics.rust_kernel import is_rust_available
    from src.shared.python.sg_optimizer.shot_model.ground_runout import SlopeAwareGround

    if not is_rust_available():
        pytest.skip("upstream-physics Rust kernel required for flight pipeline")
    sim_carry = PipelineLandingSimulator()
    sim_land = PipelineLandingSimulator(landing_state=True)
    ground = SlopeAwareGround()
    impacts = np.array([_ZERO])
    carry = sim_carry(impacts, "driver", (0.0, 0.0))
    landing = sim_land(impacts, "driver", (0.0, 0.0))
    rest = ground(landing, "fairway", "flat", "driver")
    assert rest[0, 0] > carry[0, 0] + 0.5


def test_real_pipeline_driver_carry_is_plausible_and_face_curves_ball() -> None:
    from src.shared.python.physics.rust_kernel import is_rust_available

    if not is_rust_available():
        pytest.skip("upstream-physics Rust kernel required for flight pipeline")
    sim = PipelineLandingSimulator()
    out = sim(np.stack([_ZERO, _ZERO + [0, 4.0, 0, 0, 0, 0]]), "driver", (0.0, 0.0))
    assert 150.0 < out[0, 0] < 330.0  # yards; a 100 mph driver
    assert abs(out[0, 1]) < 8.0  # square face/path: nearly straight
    assert out[1, 1] < out[0, 1] - 3.0  # open face -> lands right (y is left)


def test_landing_state_mode_adds_velocity_and_matches_carry_columns() -> None:
    from src.shared.python.physics.rust_kernel import is_rust_available

    if not is_rust_available():
        pytest.skip("upstream-physics Rust kernel required for flight pipeline")
    rows = np.stack([_ZERO, _ZERO + [0, 4.0, 0, 0, 0, 0]])
    carry = PipelineLandingSimulator()(rows, "mid_iron", (0.0, 0.0))
    state = PipelineLandingSimulator(landing_state=True)(rows, "mid_iron", (0.0, 0.0))
    assert state.shape == (2, 6)
    assert np.all(state[:, 5] > 0.0)  # backspin at landing
    np.testing.assert_allclose(state[:, :2], carry)
    assert np.all(state[:, 4] < 0.0)  # descending at first ground contact


def _clubs_passed(monkeypatch, skill: str) -> list[object]:
    """Run one row of ``skill`` and return the ``club`` given to ``pipe.run``."""
    from src.shared.python.physics.swing_ball_flight_pipeline import (
        SwingBallFlightPipeline,
    )

    seen: list[object] = []

    def _run(self, swing, **kw):  # noqa: ANN001, ANN202 - spy on the real run
        seen.append(kw.get("club"))
        raise _Stop

    monkeypatch.setattr(SwingBallFlightPipeline, "run", _run)
    with pytest.raises(_Stop):
        PipelineLandingSimulator()(np.array([_ZERO]), skill, (0.0, 0.0))
    return seen


class _Stop(Exception):
    """Ends a spied pipeline run before flight."""


def test_wedge_skill_runs_the_pipeline_with_the_wedge_spin_cap(monkeypatch) -> None:
    """#15443: the skill's own club, not the driver fallback, sets the spin cap."""
    from src.shared.python.physics.ball_flight_validity import (
        MAX_DRIVER_LAUNCH_SPIN_RPM,
        launch_spin_envelope,
    )

    (club,) = _clubs_passed(monkeypatch, "wedge")
    assert club == REPRESENTATIVE_CLUB["wedge"]
    envelope = launch_spin_envelope(club)
    assert envelope.club_class == "wedge"
    assert envelope.max_rpm > MAX_DRIVER_LAUNCH_SPIN_RPM


@pytest.mark.parametrize(
    ("skill", "club_class"),
    [
        ("driver", "driver"),
        ("fairway_wood", "wood"),
        ("long_iron", "iron"),
        ("mid_iron", "iron"),
        ("short_iron", "iron"),
    ],
)
def test_full_swing_skills_pass_their_bag_club(monkeypatch, skill, club_class) -> None:
    from src.shared.python.physics.ball_flight_validity import club_class_of

    (club,) = _clubs_passed(monkeypatch, skill)
    assert club == REPRESENTATIVE_CLUB[skill]
    assert club_class_of(club) == club_class


def test_short_game_skill_without_a_club_class_warns_and_falls_back(
    monkeypatch, caplog
) -> None:
    """No bag club -> the pipeline's documented driver fallback, logged, not silent."""
    from src.shared.python.physics.ball_flight_validity import club_class_of

    pipeline_club.cache_clear()  # the warning is logged once per skill
    with caplog.at_level("WARNING"):
        (club,) = _clubs_passed(monkeypatch, "pitch")
    assert club == REPRESENTATIVE_CLUB["pitch"]
    assert club_class_of(club) is None
    assert "pitch" in caplog.text and "fallback" in caplog.text
