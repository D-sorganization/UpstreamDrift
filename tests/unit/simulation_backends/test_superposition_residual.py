"""Forward counterfactual trajectories do not superpose (Issue #14493).

ZTCF (drift) + ZVCF (control, ``M(q)^-1 tau``) equals the total acceleration
only at one fixed state. Forked at ``t0`` and integrated, each counterfactual
trajectory drifts to its own state, where ``M(q)``, ``C(q, v)`` and ``g(q)``
differ, so the residual ``||base - (ztcf + control)||`` is zero at the fork
and positive afterwards.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.simulation_backends import GolfModelParams, make_backend
from src.shared.python.simulation_backends.superposition import (
    SuperpositionResidual,
    fork_counterfactual_accelerations,
    superposition_residual,
)

pytestmark = pytest.mark.unit

_Q0 = np.array([0.4, -0.9])
_V0 = np.array([1.5, -2.0])
_TAU = np.array([6.0, -3.0])


@pytest.fixture(scope="module")
def forked() -> tuple[np.ndarray, ...]:
    backend = make_backend("ode", GolfModelParams.default())
    return fork_counterfactual_accelerations(
        backend, _Q0, _V0, _TAU, dt=1e-3, steps=200
    )


def test_residual_is_zero_at_the_fork_and_positive_after(forked) -> None:
    times, base, ztcf, control = forked
    result = superposition_residual(base, ztcf, control, times=times)
    assert result.residual.shape == times.shape
    assert result.fork_residual <= 1e-9 * np.abs(base[0]).max()
    assert result.residual[-1] > 1e-3
    assert result.max_residual == pytest.approx(result.residual.max())


def test_fork_accelerations_match_the_pointwise_split(forked) -> None:
    times, base, ztcf, control = forked
    backend = make_backend("ode", GolfModelParams.default())
    assert times[0] == 0.0
    np.testing.assert_allclose(
        base[0], backend.forward_dynamics(_Q0, _V0, _TAU), atol=1e-9
    )
    np.testing.assert_allclose(
        control[0], np.linalg.solve(backend.mass_matrix(_Q0), _TAU), atol=1e-9
    )


def test_integrated_sums_are_never_labelled_exact(forked) -> None:
    times, base, ztcf, control = forked
    text = superposition_residual(base, ztcf, control, times=times).describe()
    assert "not exact" in text
    assert "fork" in text
    assert "exact" not in text.replace("not exact", "")


def test_identical_series_have_zero_residual() -> None:
    base = np.ones((3, 2))
    result = superposition_residual(base, 0.25 * base, 0.75 * base)
    assert isinstance(result, SuperpositionResidual)
    np.testing.assert_array_equal(result.times, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(result.residual, 0.0, atol=1e-15)


@pytest.mark.parametrize(
    ("ztcf", "times", "message"),
    [
        (np.ones((3, 3)), None, "shape"),
        (np.full((3, 2), np.nan), None, "finite"),
        (np.ones((3, 2)), np.array([0.0, 0.0, 1.0]), "increasing"),
        (np.ones((3, 2)), np.array([0.0, 1.0]), "times"),
    ],
)
def test_superposition_residual_rejects_bad_inputs(
    ztcf: np.ndarray, times: np.ndarray | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        superposition_residual(np.ones((3, 2)), ztcf, np.ones((3, 2)), times=times)


@pytest.mark.parametrize(
    ("kwargs", "message"), [({"dt": 0.0}, "dt"), ({"steps": 0}, "steps")]
)
def test_fork_rejects_bad_integration_settings(kwargs: dict, message: str) -> None:
    backend = make_backend("ode", GolfModelParams.default())
    settings = {"dt": 1e-3, "steps": 10, **kwargs}
    with pytest.raises(ValueError, match=message):
        fork_counterfactual_accelerations(backend, _Q0, _V0, _TAU, **settings)
