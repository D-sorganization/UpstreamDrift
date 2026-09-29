"""Contracts for local nonlinear-system rank diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.local_linear_diagnostics import (
    LinearizationPoint,
    NondimensionalScales,
    RankTolerance,
    audit_double_pendulum_configuration_state,
    audit_local_linearization,
)
from src.shared.python.simulation_backends import GolfModelParams

pytestmark = pytest.mark.scientific


def _double_integrator_scales(unit_factor: float = 1.0) -> NondimensionalScales:
    return NondimensionalScales(
        state=(unit_factor, unit_factor),
        control=(unit_factor,),
        output=(unit_factor,),
        characteristic_time_s=1.0,
    )


def _double_pendulum_point(control: tuple[float, ...]) -> LinearizationPoint:
    return LinearizationPoint(
        state=(-0.8, 1.1, 2.5, -1.2),
        control=control,
        state_steps=(1e-6, 1e-6, 1e-6, 1e-6),
        control_steps=tuple(1e-5 for _ in control),
    )


def test_manufactured_double_integrator_is_observable_and_controllable() -> None:
    state_matrix = np.array([[0.0, 1.0], [0.0, 0.0]])
    input_matrix = np.array([[0.0], [1.0]])
    output_matrix = np.array([[1.0, 0.0]])

    result = audit_local_linearization(
        dynamics=lambda state, control: state_matrix @ state + input_matrix @ control,
        output=lambda state: output_matrix @ state,
        state=np.zeros(2),
        control=np.zeros(1),
        state_steps=np.full(2, 1e-6),
        control_steps=np.full(1, 1e-6),
        scales=_double_integrator_scales(),
        tolerance=RankTolerance(absolute=1e-10, relative=1e-8),
    )

    assert result.state_dimension == 2
    assert result.control_dimension == 1
    assert result.output_dimension == 1
    assert result.observability.rank == 2
    assert result.controllability.rank == 2
    assert result.observability.full_rank is True
    assert result.controllability.full_rank is True


def test_zero_input_and_zero_output_killswitches_lose_rank() -> None:
    state_matrix = np.array([[0.0, 1.0], [-2.0, -0.5]])
    tolerance = RankTolerance(absolute=1e-10, relative=1e-8)

    no_input = audit_local_linearization(
        dynamics=lambda state, control: state_matrix @ state,
        output=lambda state: state[:1],
        state=np.array([0.2, -0.1]),
        control=np.zeros(1),
        state_steps=np.full(2, 1e-6),
        control_steps=np.full(1, 1e-6),
        scales=_double_integrator_scales(),
        tolerance=tolerance,
    )
    no_output = audit_local_linearization(
        dynamics=lambda state, control: (
            state_matrix @ state + np.array([0.0, control[0]])
        ),
        output=lambda state: np.zeros(1),
        state=np.array([0.2, -0.1]),
        control=np.zeros(1),
        state_steps=np.full(2, 1e-6),
        control_steps=np.full(1, 1e-6),
        scales=_double_integrator_scales(),
        tolerance=tolerance,
    )

    assert no_input.controllability.rank == 0
    assert no_input.controllability.full_rank is False
    assert no_output.observability.rank == 0
    assert no_output.observability.full_rank is False


def test_double_pendulum_configuration_output_has_full_local_linear_rank() -> None:
    result = audit_double_pendulum_configuration_state(
        GolfModelParams.default(),
        point=_double_pendulum_point((35.0, -8.0)),
        scales=NondimensionalScales(
            state=(1.0, 1.0, 10.0, 10.0),
            control=(60.0, 15.0),
            output=(1.0, 1.0),
            characteristic_time_s=0.35,
        ),
        tolerance=RankTolerance(absolute=1e-9, relative=1e-8),
    )

    assert result.state_dimension == 4
    assert result.control_dimension == 2
    assert result.output_dimension == 2
    assert result.observability.rank == 4
    assert result.controllability.rank == 4
    assert "local first-order" in result.inference_boundary
    assert "structural identifiability" in result.inference_boundary


@pytest.mark.parametrize(
    ("state_steps", "control_steps", "match"),
    [
        (np.array([1e-6, 0.0]), np.array([1e-6]), "state_steps"),
        (np.full(2, 1e-6), np.array([-1e-6]), "control_steps"),
    ],
)
def test_finite_difference_steps_fail_closed(
    state_steps: np.ndarray, control_steps: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        audit_local_linearization(
            dynamics=lambda state, control: state,
            output=lambda state: state[:1],
            state=np.zeros(2),
            control=np.zeros(1),
            state_steps=state_steps,
            control_steps=control_steps,
            scales=_double_integrator_scales(),
            tolerance=RankTolerance(absolute=1e-10, relative=1e-8),
        )


def test_rank_tolerances_must_be_finite_and_positive() -> None:
    with pytest.raises(ValueError, match="absolute"):
        RankTolerance(absolute=0.0, relative=1e-8)
    with pytest.raises(ValueError, match="relative"):
        RankTolerance(absolute=1e-10, relative=float("nan"))


def test_dimensionless_linearization_is_invariant_to_equivalent_length_units() -> None:
    state_matrix = np.array([[0.0, 1.0], [-2.0, -0.5]])
    input_matrix = np.array([[0.0], [1.0]])
    output_matrix = np.array([[1.0, 0.0]])

    def run(unit_factor: float):
        return audit_local_linearization(
            dynamics=lambda state, control: (
                state_matrix @ state + input_matrix @ control
            ),
            output=lambda state: output_matrix @ state,
            state=np.array([0.2, -0.1]) * unit_factor,
            control=np.array([0.3]) * unit_factor,
            state_steps=np.full(2, 1e-6) * unit_factor,
            control_steps=np.full(1, 1e-6) * unit_factor,
            scales=_double_integrator_scales(unit_factor),
            tolerance=RankTolerance(absolute=1e-10, relative=1e-8),
        )

    metres = run(1.0)
    centimetres = run(100.0)

    np.testing.assert_allclose(
        metres.dimensionless_state_matrix,
        centimetres.dimensionless_state_matrix,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        metres.dimensionless_input_matrix,
        centimetres.dimensionless_input_matrix,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        metres.controllability.singular_values,
        centimetres.controllability.singular_values,
        rtol=1e-10,
        atol=1e-10,
    )


def test_nondimensional_scales_fail_closed_on_invalid_values_or_dimensions() -> None:
    with pytest.raises(ValueError, match="state scales"):
        NondimensionalScales(
            state=(1.0, 0.0),
            control=(1.0,),
            output=(1.0,),
            characteristic_time_s=1.0,
        )

    with pytest.raises(ValueError, match="state scales"):
        audit_local_linearization(
            dynamics=lambda state, control: state,
            output=lambda state: state[:1],
            state=np.zeros(2),
            control=np.zeros(1),
            state_steps=np.full(2, 1e-6),
            control_steps=np.full(1, 1e-6),
            scales=NondimensionalScales(
                state=(1.0,),
                control=(1.0,),
                output=(1.0,),
                characteristic_time_s=1.0,
            ),
            tolerance=RankTolerance(absolute=1e-10, relative=1e-8),
        )


def test_double_pendulum_accepts_declared_measurement_and_actuator_countermodels() -> (
    None
):
    params = GolfModelParams.default()
    common = {
        "params": params,
        "tolerance": RankTolerance(absolute=1e-9, relative=1e-8),
    }
    shoulder_only = audit_double_pendulum_configuration_state(
        **common,
        point=_double_pendulum_point((35.0,)),
        scales=NondimensionalScales(
            state=(1.0, 1.0, 10.0, 10.0),
            control=(60.0,),
            output=(1.0,),
            characteristic_time_s=0.35,
        ),
        generalized_control_map=np.array([[1.0], [0.0]]),
        output_map=np.array([[1.0, 0.0, 0.0, 0.0]]),
    )
    zero_actuator = audit_double_pendulum_configuration_state(
        **common,
        point=_double_pendulum_point((0.0,)),
        scales=NondimensionalScales(
            state=(1.0, 1.0, 10.0, 10.0),
            control=(1.0,),
            output=(1.0,),
            characteristic_time_s=0.35,
        ),
        generalized_control_map=np.zeros((2, 1)),
        output_map=np.array([[0.0, 1.0, 0.0, 0.0]]),
    )

    assert shoulder_only.control_dimension == 1
    assert shoulder_only.output_dimension == 1
    assert 0 <= shoulder_only.controllability.rank <= 4
    assert 0 <= shoulder_only.observability.rank <= 4
    assert zero_actuator.controllability.rank == 0
