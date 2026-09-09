"""Default reference routing and explicit discrete-map contracts for #9830."""

from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.optimization import casadi_backend as backend
from src.shared.python.optimization._swing_kinematics import JOINTS
from src.shared.python.optimization._swing_models import (
    ClubModel,
    GolferModel,
    OptimizationConfig,
)
from src.shared.python.optimization._swing_reference import ReferenceControls

pytestmark = pytest.mark.unit


def _case() -> tuple:
    count = len(JOINTS)
    positions = np.zeros((count, 2))
    positions[0] = 1.0
    state = np.r_[positions.ravel(), np.zeros(count * 2)]
    return (
        GolferModel(),
        ClubModel(),
        OptimizationConfig(n_nodes=2, swing_duration=0.12),
        state,
    )


@pytest.fixture
def oscillator_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "require_casadi", lambda: object())
    monkeypatch.setattr(
        backend,
        "build_forward_dynamics",
        lambda *args, **kwargs: lambda q, v, torque: -(80**2) * q,
    )
    # A deliberately inaccurate identity map makes the stored nodes feasible.
    monkeypatch.setattr(
        backend,
        "build_rk4_integrator",
        lambda *args, **kwargs: lambda q, v, torque: (q, v),
    )


def test_default_measures_the_ode_and_explicit_grid_measures_discrete_feasibility(
    oscillator_backend: None,
) -> None:
    case = _case()
    torques = np.zeros((len(JOINTS), 1))
    discrete = backend.dynamics_defect(*case, torques=torques, n_substeps=8)
    independent = backend.dynamics_defect(*case, torques=torques)
    assert discrete.max_position_defect == 0
    assert discrete.reference_resolution is None
    assert independent.max_position_defect == pytest.approx(
        abs(np.cos(9.6) - 1), abs=1e-8
    )
    assert independent.max_velocity_defect == pytest.approx(
        abs(80 * np.sin(9.6)), abs=1e-7
    )
    assert independent.reference_resolution is not None
    assert len(independent.reference_resolution) == 1
    assert max(independent.reference_resolution[0].normalized_refinement) <= 1
    assert set(independent.to_dict()) == set(discrete.to_dict())


def test_reference_controls_are_not_silently_ignored_in_fixed_grid_mode(
    oscillator_backend: None,
) -> None:
    controls = ReferenceControls(1e-8, 1e-10, 1e-10, 0.003)
    with pytest.raises(ValueError, match="fixed"):
        backend.dynamics_defect(
            *_case(),
            torques=np.zeros((len(JOINTS), 1)),
            n_substeps=8,
            reference_controls=controls,
        )


@pytest.mark.parametrize(
    "value, error",
    [(True, TypeError), (1.5, TypeError), (0, ValueError), (-1, ValueError)],
)
def test_explicit_grid_requires_a_positive_integer(
    value: object, error: type, oscillator_backend: None
) -> None:
    with pytest.raises(error):
        backend.dynamics_defect(
            *_case(), torques=np.zeros((len(JOINTS), 1)), n_substeps=value
        )


@pytest.mark.parametrize("value", [True, "0", np.nan])
def test_torque_values_are_not_coerced_or_nonfinite(
    value: object, oscillator_backend: None
) -> None:
    torques = [[value] for _ in JOINTS]
    with pytest.raises((TypeError, ValueError)):
        backend.dynamics_defect(*_case(), torques=torques)


@pytest.mark.parametrize("duration", [0.0, -1.0, np.inf, np.nan])
def test_duration_is_validated_before_evaluation(
    duration: float, oscillator_backend: None
) -> None:
    golfer, club, config, state = _case()
    with pytest.raises(ValueError):
        backend.dynamics_defect(
            golfer,
            club,
            replace(config, swing_duration=duration),
            state,
            torques=np.zeros((len(JOINTS), 1)),
        )
