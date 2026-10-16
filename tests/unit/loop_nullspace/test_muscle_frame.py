"""NSX-94 (#12723): the muscle frame value object and its qualification gate.

Closed forms.  For the elbow fixture ``G = R^T diag(F_a) = [40, 24, -42]`` N m,
so the reachable torque interval with ``0 <= a <= 1`` is ``[-42, 64]`` N m.  A
request of ``T`` N m above 64 needs a reserve of ``T - 64`` N m, i.e. a reserve
fraction ``(T - 64) / T``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace.muscle_frame import (
    FrameQualification,
    MuscleFrame,
    NotQualifiedError,
    QualificationStatus,
    qualify_frame,
    require_qualified,
)
from tests.unit.loop_nullspace._muscle_fixtures import MODEL, INPUT, elbow_frame

pytestmark = pytest.mark.unit


def test_elbow_frame_shapes_and_gain() -> None:
    frame = elbow_frame()
    assert (frame.n_muscles, frame.n_coordinates) == (3, 1)
    np.testing.assert_allclose(frame.passive_force, 0.0, atol=1e-9)
    np.testing.assert_allclose(frame.active_force, [1000.0, 800.0, 1200.0], atol=1e-6)
    np.testing.assert_allclose(frame.gain_matrix, [[40.0, 24.0, -42.0]], atol=1e-6)
    assert np.linalg.matrix_rank(frame.gain_matrix) == 1


def test_torque_is_linear_in_activation() -> None:
    frame = elbow_frame()
    a = np.array([0.5, 0.25, 0.2])
    expected = 0.5 * 40.0 + 0.25 * 24.0 - 0.2 * 42.0
    assert frame.torque(a)[0] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"moment_arms": np.array([[0.04], [0.03]])}, "moment_arms"),
        ({"force_grid": np.array([[0.0, 0.0, 0.0], [-1.0, 1.0, 1.0]])}, "force_grid"),
        ({"activation_grid": np.array([0.1, 1.0])}, "activation_grid"),
        ({"max_isometric_force": np.array([1.0, 0.0, 1.0])}, "max_isometric_force"),
        ({"names": ("a", "a", "b")}, "names"),
        ({"coordinates": ()}, "coordinates"),
    ],
)
def test_frame_rejects_bad_inputs(kwargs: dict, message: str) -> None:
    base = elbow_frame()
    fields = {
        "names": base.names,
        "coordinates": base.coordinates,
        "moment_arms": base.moment_arms,
        "activation_grid": base.activation_grid,
        "force_grid": base.force_grid,
        "max_isometric_force": base.max_isometric_force,
        "tendon": base.tendon,
        "engine": "fixture",
        "model_provenance": MODEL,
        "input_provenance": INPUT,
    }
    fields.update(kwargs)
    with pytest.raises(ValueError, match=message):
        MuscleFrame(**fields)


def test_non_finite_moment_arm_is_rejected() -> None:
    base = elbow_frame()
    arms = base.moment_arms.copy()
    arms[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        MuscleFrame(
            names=base.names,
            coordinates=base.coordinates,
            moment_arms=arms,
            activation_grid=base.activation_grid,
            force_grid=base.force_grid,
            max_isometric_force=base.max_isometric_force,
            tendon=base.tendon,
            engine="fixture",
            model_provenance=MODEL,
            input_provenance=INPUT,
        )


def test_linearity_residual_is_zero_for_two_point_grid_and_known_for_curved() -> None:
    base = elbow_frame()
    assert base.linearity_residual() == 0.0
    # Midpoint force 10 percent above the secant for the biceps: torque error
    # = R_biceps * 0.1 * 500 N = 0.04 * 50 = 2 N m.
    grid = np.array([base.activation_grid[0], 0.5, 1.0])
    forces = np.array(
        [
            np.zeros(3),
            0.5 * base.active_force * np.array([1.1, 1.0, 1.0]),
            base.active_force,
        ]
    )
    curved = MuscleFrame(
        names=base.names,
        coordinates=base.coordinates,
        moment_arms=base.moment_arms,
        activation_grid=grid,
        force_grid=forces,
        max_isometric_force=base.max_isometric_force,
        tendon=base.tendon,
        engine="fixture",
        model_provenance=MODEL,
        input_provenance=INPUT,
    )
    assert curved.linearity_residual() == pytest.approx(0.04 * 0.1 * 500.0, abs=1e-9)
    # The fibre algebra still uses the secant, never the curved table.
    np.testing.assert_allclose(curved.active_force, base.active_force)
    assert curved.engine_torque(np.array([0.5, 0.0, 0.0]))[0] == pytest.approx(
        0.04 * 550.0, abs=1e-9
    )


def test_reachable_target_is_qualified_with_zero_reserve() -> None:
    q = qualify_frame(elbow_frame(), np.array([20.0]))
    assert isinstance(q, FrameQualification)
    assert q.status is QualificationStatus.QUALIFIED
    assert q.reserve_fraction == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(q.reachable_torque, [20.0], atol=1e-9)
    assert q.reasons == ()


def test_reserve_carrying_the_load_is_not_qualified() -> None:
    q = qualify_frame(elbow_frame(), np.array([80.0]))
    assert q.status is QualificationStatus.NOT_QUALIFIED
    assert q.reserve_fraction == pytest.approx(16.0 / 80.0, abs=1e-9)
    assert any("reserve" in reason for reason in q.reasons)
    with pytest.raises(NotQualifiedError, match="reserve"):
        require_qualified(q)


def test_small_reserve_below_the_limit_stays_qualified() -> None:
    q = qualify_frame(elbow_frame(), np.array([66.0]))
    assert q.reserve_fraction == pytest.approx(2.0 / 66.0, abs=1e-9)
    assert q.status is QualificationStatus.QUALIFIED
    np.testing.assert_allclose(q.reachable_torque, [64.0], atol=1e-9)


def test_uncovered_coordinate_is_carried_entirely_by_the_reserve() -> None:
    base = elbow_frame()
    arms = np.hstack([base.moment_arms, np.zeros((3, 1))])
    two_dof = MuscleFrame(
        names=base.names,
        coordinates=("elbow", "wrist"),
        moment_arms=arms,
        activation_grid=base.activation_grid,
        force_grid=base.force_grid,
        max_isometric_force=base.max_isometric_force,
        tendon=base.tendon,
        engine="fixture",
        model_provenance=MODEL,
        input_provenance=INPUT,
    )
    q = qualify_frame(two_dof, np.array([20.0, 5.0]))
    assert q.status is QualificationStatus.NOT_QUALIFIED
    assert q.reserve_fraction == pytest.approx(5.0 / np.hypot(20.0, 5.0), abs=1e-9)


def test_nonlinear_engine_table_is_not_qualified() -> None:
    base = elbow_frame()
    forces = np.array(
        [
            np.zeros(3),
            0.5 * base.active_force * np.array([2.0, 1.0, 1.0]),
            base.active_force,
        ]
    )
    frame = MuscleFrame(
        names=base.names,
        coordinates=base.coordinates,
        moment_arms=base.moment_arms,
        activation_grid=np.array([0.0, 0.5, 1.0]),
        force_grid=forces,
        max_isometric_force=base.max_isometric_force,
        tendon=base.tendon,
        engine="fixture",
        model_provenance=MODEL,
        input_provenance=INPUT,
    )
    # residual = 0.04 * 500 = 20 N m against a 20 N m target -> fraction 1.0.
    q = qualify_frame(frame, np.array([20.0]))
    assert q.linearity_fraction == pytest.approx(1.0, abs=1e-9)
    assert q.status is QualificationStatus.NOT_QUALIFIED
    assert any("linear" in reason for reason in q.reasons)


def test_zero_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        qualify_frame(elbow_frame(), np.array([0.0]))
