"""Impact-condition -> landing adapter over ``SwingBallFlightPipeline`` (#14567)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.sg_optimizer.shot_model.physics_simulator import (
    CLUB_LOFT_DEG,
    PipelineLandingSimulator,
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
    assert state.shape == (2, 5)
    np.testing.assert_allclose(state[:, :2], carry)
    assert np.all(state[:, 4] < 0.0)  # descending at first ground contact
