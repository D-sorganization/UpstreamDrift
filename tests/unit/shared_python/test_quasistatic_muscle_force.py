from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, is_dataclass
import math

import pytest

from src.shared.python.biomechanics.hill_muscle import HillMuscleModel, MuscleParameters
from src.shared.python.biomechanics.muscle_equilibrium import EquilibriumSolver
from src.shared.python.biomechanics.muscle_force_evaluation import (
    QuasistaticMuscleForceEvaluation,
    evaluate_quasistatic_muscle_force,
)

pytestmark = pytest.mark.unit


def _model(pennation: float = 0.25) -> HillMuscleModel:
    return HillMuscleModel(
        MuscleParameters(
            F_max=100.0,
            l_opt=0.1,
            l_slack=0.2,
            pennation_angle=pennation,
            damping=4.0,
        )
    )


def test_force_components_match_hill_equilibrium_and_pennation_projection() -> None:
    model = _model()
    length = 0.31
    activation = 0.3
    fiber_length = EquilibriumSolver(model).solve_fiber_length(length, activation)
    cosine = math.cos(model.params.pennation_angle)
    tendon_length = length - fiber_length * cosine
    normalized_fiber_length = fiber_length / model.params.l_opt

    expected_active = (
        model.params.F_max
        * activation
        * model.force_length_active(normalized_fiber_length)
        * model.force_velocity(0.0)
    )
    expected_passive = model.params.F_max * model.force_length_passive(
        normalized_fiber_length
    )
    expected_tendon = model.params.F_max * model.tendon_force(
        tendon_length / model.params.l_slack
    )

    result = evaluate_quasistatic_muscle_force(model, length, activation)

    assert isinstance(result, QuasistaticMuscleForceEvaluation)
    assert result.fiber_length_m == pytest.approx(fiber_length)
    assert result.tendon_length_m == pytest.approx(tendon_length)
    assert result.active_fiber_along_fiber_n == pytest.approx(expected_active)
    assert result.passive_fiber_along_fiber_n == pytest.approx(expected_passive)
    assert result.damping_fiber_along_fiber_n == 0.0
    assert result.active_tendon_axis_n == pytest.approx(expected_active * cosine)
    assert result.passive_tendon_axis_n == pytest.approx(expected_passive * cosine)
    assert result.damping_tendon_axis_n == 0.0
    assert result.fiber_tendon_axis_n == pytest.approx(
        (expected_active + expected_passive) * cosine
    )
    assert result.tendon_force_n == pytest.approx(expected_tendon)
    assert result.force_equilibrium_residual_n == pytest.approx(
        result.fiber_tendon_axis_n - result.tendon_force_n
    )


def test_force_composition_matches_closed_form_and_preserves_large_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = HillMuscleModel(
        MuscleParameters(
            F_max=100.0,
            l_opt=0.1,
            l_slack=0.2,
            pennation_angle=math.pi / 3.0,
            damping=4.0,
        )
    )
    monkeypatch.setattr(
        EquilibriumSolver, "solve_fiber_length", lambda self, *_args: 0.1
    )

    result = evaluate_quasistatic_muscle_force(model, 0.258, 0.3)

    assert result.fiber_length_m == pytest.approx(0.1)
    assert result.tendon_length_m == pytest.approx(0.208)
    assert result.active_fiber_along_fiber_n == pytest.approx(30.0)
    assert result.passive_fiber_along_fiber_n == 0.0
    assert result.damping_fiber_along_fiber_n == 0.0
    assert result.active_tendon_axis_n == pytest.approx(15.0)
    assert result.fiber_tendon_axis_n == pytest.approx(15.0)
    assert result.tendon_force_n == pytest.approx(1.6)
    # The public evaluator reports this deliberately non-equilibrated probe;
    # tolerance acceptance belongs to its caller.
    assert result.force_equilibrium_residual_n == pytest.approx(13.4)


@pytest.mark.parametrize("fiber_length", [float("nan"), 0.0, 1.0])
def test_invalid_equilibrium_lengths_are_rejected(
    monkeypatch: pytest.MonkeyPatch, fiber_length: float
) -> None:
    monkeypatch.setattr(
        EquilibriumSolver, "solve_fiber_length", lambda self, *_args: fiber_length
    )
    with pytest.raises(ValueError, match="invalid fiber length|non-positive tendon"):
        evaluate_quasistatic_muscle_force(_model(), 0.315, 0.42)


@pytest.mark.parametrize(
    ("length", "activation"),
    [
        (0.0, 0.4),
        (-0.1, 0.4),
        (float("nan"), 0.4),
        (0.31, float("nan")),
        (0.31, -0.1),
        (0.31, 1.1),
    ],
)
def test_invalid_length_or_activation_is_rejected(
    length: float, activation: float
) -> None:
    with pytest.raises(ValueError):
        evaluate_quasistatic_muscle_force(_model(), length, activation)


def test_non_hill_engine_object_is_rejected() -> None:
    with pytest.raises(TypeError, match="HillMuscleModel"):
        evaluate_quasistatic_muscle_force(object(), 0.31, 0.4)  # type: ignore[arg-type]


def test_nonfinite_component_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    monkeypatch.setattr(
        EquilibriumSolver, "solve_fiber_length", lambda self, *args: 0.1
    )
    monkeypatch.setattr(model, "tendon_force", lambda _: float("inf"))
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_quasistatic_muscle_force(model, 0.315, 0.42)


def test_equilibrium_solver_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(self: EquilibriumSolver, *args: object, **kwargs: object) -> float:
        raise RuntimeError("equilibrium did not converge")

    monkeypatch.setattr(EquilibriumSolver, "solve_fiber_length", fail)
    with pytest.raises(RuntimeError, match="did not converge"):
        evaluate_quasistatic_muscle_force(_model(), 0.315, 0.42)


def test_result_is_immutable_and_serializes_only_scalar_values() -> None:
    result = evaluate_quasistatic_muscle_force(_model(), 0.315, 0.42)
    assert is_dataclass(result)
    with pytest.raises(FrozenInstanceError):
        result.tendon_force_n = 0.0  # type: ignore[misc]

    snapshot = asdict(result)
    snapshot["tendon_force_n"] = -1.0
    assert result.tendon_force_n >= 0.0
    assert all(isinstance(value, float) for value in asdict(result).values())
