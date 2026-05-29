"""Unit tests for the cross-backend validation harness (M5 correctness gate).

Two tiers of tests:

* **Pure-logic tier** (always runs): drives :mod:`validation` with lightweight
  in-process fakes that satisfy the ``DynamicsProvider`` / ``SimulationBackend``
  Protocols. These exercise the tolerance semantics, precondition guards, and
  report wiring with no optional dependency.
* **MuJoCo cross-validation tier** (``@pytest.mark.requires_mujoco``): builds the
  real ``ode`` and ``mujoco`` backends via :func:`make_backend` and asserts they
  agree within the documented tolerances.

Every comparison the harness performs is tolerance-based (``numpy.allclose``),
never bit-equality — see the module docstring of :mod:`validation`.
"""

import numpy as np
import pytest

from src.shared.python.simulation_backends.capabilities import has_mujoco
from src.shared.python.simulation_backends.factory import make_backend
from src.shared.python.simulation_backends.model_params import GolfModelParams
from src.shared.python.simulation_backends.protocol import SimState, Trace
from src.shared.python.simulation_backends.validation import (
    ValidationReport,
    check_energy_conservation,
    cross_validate_bias,
    cross_validate_mass_matrix,
    cross_validate_trajectory,
    kinetic_energy,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Lightweight Protocol-satisfying fakes (no optional dependency required).
# --------------------------------------------------------------------------- #


class _FakeProvider:
    """Minimal ``DynamicsProvider`` with a configurable constant skew.

    ``skew`` is added to every entry of the mass matrix and bias vector so a
    test can dial the discrepancy above or below a tolerance.
    """

    def __init__(self, skew: float = 0.0) -> None:
        self.skew = float(skew)

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Return a deterministic, configuration-dependent SPD-ish matrix."""
        q = np.asarray(q, dtype=float).reshape(-1)
        c = np.cos(q[1])
        base = np.array([[2.0 + c, 0.5 * c], [0.5 * c, 1.0]])
        return base + self.skew

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Return a deterministic bias vector that depends on (q, v)."""
        q = np.asarray(q, dtype=float).reshape(-1)
        v = np.asarray(v, dtype=float).reshape(-1)
        return np.array([np.sin(q[0]) + v[0], np.sin(q[0] + q[1]) - v[1]]) + self.skew


class _LinearBackend:
    """Deterministic ``SimulationBackend`` with closed-form linear dynamics.

    State integrates as ``q_{k+1} = q_k + dt * v_k`` and ``v`` is held constant
    (passive) so the trajectory is exactly reproducible. A ``bias`` offset lets
    a test inject a controlled trajectory discrepancy. Also satisfies
    ``DynamicsProvider`` with a constant identity mass matrix, which makes
    kinetic energy ``0.5 * |v|^2`` — conserved for the constant-``v`` rollout.
    """

    def __init__(self, params: GolfModelParams, bias: float = 0.0) -> None:
        self.params = params
        self.bias = float(bias)
        self._state = SimState(q=np.zeros(2), v=np.zeros(2))

    def reset(self, state: SimState | None = None) -> None:
        self._state = (
            SimState(q=np.zeros(2), v=np.zeros(2)) if state is None else state.copy()
        )

    def get_state(self) -> SimState:
        return self._state.copy()

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        return np.eye(2)

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.zeros(2)

    def rollout(self, controls: np.ndarray | None, horizon: int, dt: float) -> Trace:
        # ``controls`` is the (horizon, nu) per-step history; the Trace.u field
        # is the (T, nu)=(horizon+1, nu) per-sample history. This passive fake
        # ignores torque, and the harness only compares q/v, so leave u=None.
        del controls
        q = np.empty((horizon + 1, 2))
        v = np.empty((horizon + 1, 2))
        q[0] = self._state.q
        v[0] = self._state.v
        for k in range(horizon):
            v[k + 1] = v[k]
            q[k + 1] = q[k] + dt * v[k] + self.bias
        t = np.arange(horizon + 1) * dt
        return Trace(t=t, q=q, v=v, u=None, dt=dt, backend="fake-linear")


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded RNG so every sampled configuration is reproducible."""
    return np.random.default_rng(0)


@pytest.fixture
def q_samples(rng: np.random.Generator) -> list[np.ndarray]:
    """A reproducible list of length-2 configuration samples."""
    return [rng.uniform(-np.pi, np.pi, size=2) for _ in range(8)]


@pytest.fixture
def qv_samples(rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    """A reproducible list of ``(q, v)`` state samples."""
    return [
        (rng.uniform(-np.pi, np.pi, size=2), rng.uniform(-3.0, 3.0, size=2))
        for _ in range(8)
    ]


# --------------------------------------------------------------------------- #
# Pure-logic tier: tolerance semantics and DbC guards.
# --------------------------------------------------------------------------- #


def test_mass_matrix_identical_providers_pass(q_samples: list[np.ndarray]) -> None:
    report = cross_validate_mass_matrix(_FakeProvider(), _FakeProvider(), q_samples)
    assert isinstance(report, ValidationReport)
    assert report.passed is True
    assert report.max_abs_error == pytest.approx(0.0, abs=1e-15)
    assert report.name == "mass_matrix"


def test_mass_matrix_skew_above_tol_fails(q_samples: list[np.ndarray]) -> None:
    report = cross_validate_mass_matrix(
        _FakeProvider(), _FakeProvider(skew=1e-3), q_samples
    )
    assert report.passed is False
    assert report.max_abs_error == pytest.approx(1e-3, rel=1e-6)


def test_mass_matrix_skew_within_loose_tol_passes(
    q_samples: list[np.ndarray],
) -> None:
    # A small skew passes once the tolerance is widened — demonstrates the
    # comparison is allclose-based, not exact.
    report = cross_validate_mass_matrix(
        _FakeProvider(), _FakeProvider(skew=1e-9), q_samples, rtol=1e-6, atol=1e-7
    )
    assert report.passed is True


def test_bias_identical_providers_pass(
    qv_samples: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    report = cross_validate_bias(_FakeProvider(), _FakeProvider(), qv_samples)
    assert report.passed is True
    assert report.name == "bias_forces"
    assert report.rtol == pytest.approx(1e-6)


def test_bias_skew_above_tol_fails(
    qv_samples: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    report = cross_validate_bias(_FakeProvider(), _FakeProvider(skew=0.5), qv_samples)
    assert report.passed is False
    assert report.max_abs_error == pytest.approx(0.5, rel=1e-6)


def test_trajectory_identical_backends_pass() -> None:
    params = GolfModelParams.default()
    report = cross_validate_trajectory(
        _LinearBackend(params), _LinearBackend(params), None, horizon=20, dt=0.01
    )
    assert report.passed is True
    assert report.name == "trajectory"
    # Documented loosest tolerance for an integrated trajectory.
    assert report.rtol == pytest.approx(1e-4)
    assert report.atol == pytest.approx(1e-5)


def test_trajectory_divergence_fails() -> None:
    params = GolfModelParams.default()
    report = cross_validate_trajectory(
        _LinearBackend(params),
        _LinearBackend(params, bias=0.05),
        None,
        horizon=20,
        dt=0.01,
    )
    assert report.passed is False
    assert report.max_abs_error > 0.0


def test_trajectory_with_controls_shape_validated() -> None:
    params = GolfModelParams.default()
    controls = np.zeros((10, 2))
    report = cross_validate_trajectory(
        _LinearBackend(params), _LinearBackend(params), controls, horizon=10, dt=0.01
    )
    assert report.passed is True
    assert "driven" in report.detail


def test_kinetic_energy_identity_mass() -> None:
    params = GolfModelParams.default()
    backend = _LinearBackend(params)  # identity mass matrix
    energy = kinetic_energy(backend, q=np.array([0.2, -0.4]), v=np.array([3.0, 4.0]))
    assert energy == pytest.approx(0.5 * (3.0**2 + 4.0**2))


def test_energy_conservation_constant_velocity_passes() -> None:
    # Constant velocity + identity mass => kinetic energy is exactly conserved.
    params = GolfModelParams.default()
    backend = _LinearBackend(params)
    report = check_energy_conservation(
        backend, backend, SimState(q=[0.0, 0.0], v=[1.0, -0.5]), horizon=50, dt=0.002
    )
    assert report.passed is True
    assert report.name == "energy_conservation"
    assert report.max_abs_error == pytest.approx(0.0, abs=1e-12)


def test_energy_conservation_detects_drift() -> None:
    # bias makes q drift but v constant; KE itself is constant here, so to test
    # detection we use a backend whose velocity grows.
    params = GolfModelParams.default()

    class _GrowingBackend(_LinearBackend):
        def rollout(self, controls, horizon, dt):  # type: ignore[override]
            del controls
            q = np.zeros((horizon + 1, 2))
            v = np.zeros((horizon + 1, 2))
            q[0] = self._state.q
            v[0] = self._state.v
            for k in range(horizon):
                v[k + 1] = v[k] * 1.05  # 5% growth per step -> energy explodes
                q[k + 1] = q[k] + dt * v[k]
            t = np.arange(horizon + 1) * dt
            return Trace(t=t, q=q, v=v, u=None, dt=dt, backend="growing")

    backend = _GrowingBackend(params)
    report = check_energy_conservation(
        backend, backend, SimState(q=[0.0, 0.0], v=[1.0, -0.5]), horizon=50, dt=0.002
    )
    assert report.passed is False
    assert report.max_abs_error > report.rtol


# --------------------------------------------------------------------------- #
# DbC precondition guards.
# --------------------------------------------------------------------------- #


def test_mass_matrix_empty_samples_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        cross_validate_mass_matrix(_FakeProvider(), _FakeProvider(), [])


def test_bias_empty_samples_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        cross_validate_bias(_FakeProvider(), _FakeProvider(), [])


def test_mass_matrix_wrong_dim_sample_raises() -> None:
    with pytest.raises(ValueError, match="exactly 2 entries"):
        cross_validate_mass_matrix(
            _FakeProvider(), _FakeProvider(), [np.array([1.0, 2.0, 3.0])]
        )


@pytest.mark.parametrize("bad_horizon", [0, -1, -10])
def test_trajectory_nonpositive_horizon_raises(bad_horizon: int) -> None:
    params = GolfModelParams.default()
    with pytest.raises(ValueError, match="horizon must be > 0"):
        cross_validate_trajectory(
            _LinearBackend(params),
            _LinearBackend(params),
            None,
            horizon=bad_horizon,
            dt=0.01,
        )


@pytest.mark.parametrize("bad_dt", [0.0, -0.01])
def test_trajectory_nonpositive_dt_raises(bad_dt: float) -> None:
    params = GolfModelParams.default()
    with pytest.raises(ValueError, match="dt must be > 0"):
        cross_validate_trajectory(
            _LinearBackend(params),
            _LinearBackend(params),
            None,
            horizon=10,
            dt=bad_dt,
        )


def test_trajectory_bad_control_shape_raises() -> None:
    params = GolfModelParams.default()
    with pytest.raises(ValueError, match=r"controls must have shape"):
        cross_validate_trajectory(
            _LinearBackend(params),
            _LinearBackend(params),
            np.zeros((5, 2)),  # wrong: horizon is 10
            horizon=10,
            dt=0.01,
        )


def test_energy_conservation_non_simstate_raises() -> None:
    params = GolfModelParams.default()
    backend = _LinearBackend(params)
    with pytest.raises(TypeError, match="must be a SimState"):
        check_energy_conservation(backend, backend, object(), horizon=10, dt=0.01)  # type: ignore[arg-type]


def test_energy_conservation_zero_initial_velocity_raises() -> None:
    params = GolfModelParams.default()
    backend = _LinearBackend(params)
    with pytest.raises(ValueError, match="initial kinetic energy is zero"):
        check_energy_conservation(
            backend, backend, SimState(q=[0.1, 0.2], v=[0.0, 0.0]), horizon=10, dt=0.01
        )


def test_report_is_frozen() -> None:
    report = cross_validate_mass_matrix(
        _FakeProvider(), _FakeProvider(), [np.array([0.1, 0.2])]
    )
    with pytest.raises((AttributeError, TypeError)):
        report.passed = False  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# MuJoCo cross-validation tier: the real M5 gate.
# --------------------------------------------------------------------------- #


@pytest.mark.requires_mujoco
@pytest.mark.skipif(not has_mujoco(), reason="mujoco not installed")
def test_ode_vs_mujoco_mass_matrix(q_samples: list[np.ndarray]) -> None:
    ode = make_backend("ode", GolfModelParams.default())
    mj = make_backend("mujoco", GolfModelParams.default())
    report = cross_validate_mass_matrix(ode, mj, q_samples)
    assert report.passed is True
    assert report.max_abs_error < 1e-7


@pytest.mark.requires_mujoco
@pytest.mark.skipif(not has_mujoco(), reason="mujoco not installed")
def test_ode_vs_mujoco_bias(
    qv_samples: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    ode = make_backend("ode", GolfModelParams.default())
    mj = make_backend("mujoco", GolfModelParams.default())
    report = cross_validate_bias(ode, mj, qv_samples)
    assert report.passed is True


@pytest.mark.requires_mujoco
@pytest.mark.skipif(not has_mujoco(), reason="mujoco not installed")
def test_ode_vs_mujoco_trajectory() -> None:
    # Trajectory tolerance is the loosest (rtol=1e-4, atol=1e-5): two distinct
    # integrators (analytical RK4 vs MuJoCo) accumulate per-step error over the
    # horizon, so we never expect bit-equality — only allclose agreement.
    ode = make_backend("ode", GolfModelParams.default())
    mj = make_backend("mujoco", GolfModelParams.default())
    report = cross_validate_trajectory(ode, mj, controls=None, horizon=50, dt=0.005)
    assert report.passed is True


@pytest.mark.requires_mujoco
@pytest.mark.skipif(not has_mujoco(), reason="mujoco not installed")
def test_energy_conservation_conservative_model() -> None:
    conservative = GolfModelParams.default().model_copy(
        update={
            "gravity_enabled": False,
            "damping_shoulder": 0.0,
            "damping_wrist": 0.0,
        }
    )
    backend = make_backend("ode", conservative)
    provider = make_backend("ode", conservative)
    report = check_energy_conservation(
        backend,
        provider,
        SimState(q=[0.3, -0.2], v=[1.0, -0.5]),
        horizon=300,
        dt=0.002,
    )
    assert report.passed is True
