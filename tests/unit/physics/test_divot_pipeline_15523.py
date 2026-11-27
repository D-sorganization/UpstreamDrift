"""Pre-ball turf work must reach the impact and launch stages (#15523)."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from src.shared.python.physics.divot_model import SwingArc, compute_divot
from src.shared.python.physics.swing_ball_flight_pipeline import SwingBallFlightPipeline
from src.shared.python.physics.terrain.materials import SurfaceMaterial
from tests.unit.physics.test_swing_ball_flight_pipeline_5337 import (
    _make_post_impact,
    _make_swing,
    _MockFlightSimulator,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def rig():
    flight = _MockFlightSimulator()
    pipeline = SwingBallFlightPipeline(flight_simulator=flight)
    solver = Mock(return_value=_make_post_impact())
    pipeline._solve_impact = solver
    return pipeline, solver, flight


def test_fat_shot_reduces_solver_velocity_and_launch_spin(rig):
    pipeline, solver, flight = rig
    swing = _make_swing(velocity=np.array([35.0, 2.0, -3.0]))
    original = swing.clubhead_velocity.copy()
    arc = SwingArc(0.02, -0.015)
    turf = SurfaceMaterial("test turf")
    expected = compute_divot(
        arc, np.linalg.norm(original), turf, effective_mass_kg=swing.clubhead_mass
    )
    result = pipeline.run(swing, divot_arc=arc, turf_material=turf, spin_factor=0.8)
    np.testing.assert_allclose(
        solver.call_args.args[0].clubhead_velocity,
        original * expected.speed_retention,
    )
    base_spin = 300.0 * 60.0 / (2.0 * np.pi)
    assert flight.calls[0].spin_rate == pytest.approx(
        base_spin * expected.spin_retention * 0.8
    )
    assert result.divot == expected
    assert result.swing_state is swing
    np.testing.assert_array_equal(swing.clubhead_velocity, original)


@pytest.mark.parametrize("arc", [SwingArc(0.30, -0.005), SwingArc(0.0, 0.01)])
def test_ball_first_and_airborne_swings_preserve_launch(rig, arc):
    pipeline, solver, flight = rig
    swing = _make_swing()
    baseline = pipeline.run(swing)
    result = pipeline.run(swing, divot_arc=arc, turf_material=SurfaceMaterial("turf"))
    np.testing.assert_array_equal(
        solver.call_args.args[0].clubhead_velocity, swing.clubhead_velocity
    )
    assert result.launch_conditions.spin_rate == baseline.launch_conditions.spin_rate
    assert result.divot.speed_retention == 1.0
    assert baseline.divot is None
    assert len(flight.calls) == 2


@pytest.mark.parametrize(
    "kwargs",
    [{"divot_arc": SwingArc(0, -0.01)}, {"turf_material": SurfaceMaterial("turf")}],
)
def test_partial_ground_context_is_rejected_before_impact(rig, kwargs):
    pipeline, solver, flight = rig
    with pytest.raises(ValueError, match="together"):
        pipeline.run(_make_swing(), **kwargs)
    solver.assert_not_called()
    assert not flight.calls


def test_turf_stopped_club_never_reaches_impact_or_flight(rig):
    pipeline, solver, flight = rig
    swing = replace(_make_swing(), clubhead_velocity=np.array([0.1, 0.0, 0.0]))
    with pytest.raises(ValueError, match="stopped"):
        pipeline.run(
            swing, divot_arc=SwingArc(0, -0.05), turf_material=SurfaceMaterial("turf")
        )
    solver.assert_not_called()
    assert not flight.calls


def test_real_impact_solver_produces_slower_ball_after_turf_contact():
    pipeline = SwingBallFlightPipeline(flight_simulator=_MockFlightSimulator())
    swing = _make_swing()
    clean = pipeline.run(swing)
    fat = pipeline.run(
        swing, divot_arc=SwingArc(0.02, -0.015), turf_material=SurfaceMaterial("turf")
    )
    assert 0 < fat.launch_conditions.velocity < clean.launch_conditions.velocity
    assert np.linalg.norm(fat.impact_state.ball_velocity) == pytest.approx(
        fat.launch_conditions.velocity
    )
