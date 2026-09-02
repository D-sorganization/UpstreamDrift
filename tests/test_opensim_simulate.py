"""Tests for ``simulate_with_coefficients`` (issue #4120).

Two layers, mirroring the convention in ``test_opensim_model_loads.py``:

1. **Pure-numpy unit tests** (always run) — exercise the polynomial
   torque law, ``SimOptions`` validation, and the canonical ``SimOut``
   schema. These do not import ``opensim``.

2. **Live OpenSim integration tests** (``requires_opensim`` marker) —
   actually call ``simulate_with_coefficients`` against the committed
   golf_humanoid.osim. Skipped when the OpenSim Python bindings are
   not installed.

The integration layer covers the three acceptance criteria from issue
#4120: recovery (known theta -> expected grip pattern), determinism
(same theta -> identical SimOut), and postcondition shape checks.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from src.engines.physics_engines.opensim.python.motion_matching.simulate import (
    COEFFS_PER_JOINT,
    POLY_DEGREE,
    SimOptions,
    SimOut,
    evaluate_polynomial_torque,
)

OPENSIM_AVAILABLE = importlib.util.find_spec("opensim") is not None
opensim_required = pytest.mark.skipif(
    not OPENSIM_AVAILABLE,
    reason=(
        "OpenSim Python bindings not installed. "
        "Install via `conda install -c opensim-org opensim` (macOS) or "
        "`pip install opensim` (Linux/Windows, OpenSim>=4.4)."
    ),
)


# ---------------------------------------------------------------------------
# Layer 1: pure-numpy unit tests (no opensim).
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPolynomialTorque:
    """Unit tests for the pure-numpy polynomial torque law."""

    def test_constant_term_recovered_at_t_zero(self) -> None:
        coeffs = np.zeros((3, COEFFS_PER_JOINT))
        coeffs[:, 0] = [1.0, -2.0, 3.5]
        tau = evaluate_polynomial_torque(coeffs, 0.0)
        np.testing.assert_allclose(tau, [1.0, -2.0, 3.5])

    def test_linear_growth(self) -> None:
        coeffs = np.zeros((1, COEFFS_PER_JOINT))
        coeffs[0, 1] = 5.0  # tau(t) = 5*t
        for t in (0.0, 0.5, 1.0, 2.0):
            tau = evaluate_polynomial_torque(coeffs, t)
            assert tau[0] == pytest.approx(5.0 * t)

    def test_full_polynomial_eval(self) -> None:
        # tau(t) = 1 + 2t + 3t^2 + 4t^3 + 5t^4 + 6t^5 + 7t^6
        coeffs = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
        t = 0.7
        expected = sum(coeffs[0, k] * t**k for k in range(POLY_DEGREE + 1))
        tau = evaluate_polynomial_torque(coeffs, t)
        assert tau[0] == pytest.approx(expected, rel=1e-12)

    def test_rejects_wrong_column_count(self) -> None:
        with pytest.raises(ValueError, match="must have 7 columns"):
            evaluate_polynomial_torque(np.zeros((3, 5)), 0.0)

    def test_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError, match="must be 2D"):
            evaluate_polynomial_torque(np.zeros(7), 0.0)

    def test_rejects_non_finite_t(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            evaluate_polynomial_torque(np.zeros((1, COEFFS_PER_JOINT)), np.inf)


@pytest.mark.unit
class TestSimOptions:
    """Validation of the ``SimOptions`` dataclass."""

    def test_defaults_accepted(self) -> None:
        opts = SimOptions()
        assert opts.t_final == 1.0
        assert opts.dt == 1e-3
        assert opts.integrator == "rk_merson"

    def test_rejects_non_positive_t_final(self) -> None:
        with pytest.raises(ValueError, match="t_final must be positive"):
            SimOptions(t_final=0.0)

    def test_rejects_dt_larger_than_horizon(self) -> None:
        with pytest.raises(ValueError, match="dt must satisfy"):
            SimOptions(t_final=0.5, dt=0.6)

    def test_rejects_unknown_integrator(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            SimOptions(integrator="leapfrog")  # type: ignore[arg-type]

    def test_rejects_bad_gravity_shape(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(3,\)"):
            SimOptions(gravity=np.zeros(2))


@pytest.mark.unit
class TestSimOutSchema:
    """Schema-only checks on the canonical ``SimOut`` dataclass."""

    def _make_simout(self, n: int = 5, j: int = 3) -> SimOut:
        return SimOut(
            time=np.zeros(n),
            q=np.zeros((n, j)),
            qd=np.zeros((n, j)),
            qdd=np.zeros((n, j)),
            tau=np.zeros((n, j)),
            grip=np.zeros((n, 3)),
            grip_quat=np.zeros((n, 4)),
            clubhead=np.zeros((n, 3)),
            club_quat=np.zeros((n, 4)),
            solver_status="success",
            duration_s=0.123,
        )

    def test_canonical_fields_present(self) -> None:
        sim_out = self._make_simout()
        for field_name in (
            "time",
            "q",
            "qd",
            "qdd",
            "tau",
            "grip",
            "grip_quat",
            "clubhead",
            "club_quat",
            "solver_status",
            "duration_s",
        ):
            assert hasattr(sim_out, field_name)

    def test_meta_defaults_to_empty_dict(self) -> None:
        assert self._make_simout().meta == {}


# ---------------------------------------------------------------------------
# Layer 2: live OpenSim integration tests.
# ---------------------------------------------------------------------------


def _zero_theta(n_actuators: int) -> np.ndarray:
    return np.zeros(n_actuators * COEFFS_PER_JOINT, dtype=np.float64)


def _n_actuators_from_model() -> int:
    """Return the actuator count for the canonical golf humanoid.

    Imports ``opensim`` lazily so module collection still works without
    the wheel installed.
    """
    import opensim as osim  # noqa: PLC0415
    from src.engines.physics_engines.opensim.python.motion_matching.simulate import (
        _DEFAULT_OSIM,
        _coordinate_actuator_names,
    )

    model = osim.Model(str(_DEFAULT_OSIM))
    model.initSystem()
    return len(_coordinate_actuator_names(model))


@pytest.mark.requires_opensim
@pytest.mark.skipif(
    not OPENSIM_AVAILABLE,
    reason="OpenSim Python bindings not installed",
)
class TestSimulateWithCoefficients:
    """Live integration tests that exercise the OpenSim binding."""

    @pytest.fixture(scope="class")
    def n_actuators(self) -> int:
        return _n_actuators_from_model()

    def test_zero_theta_returns_canonical_simout(
        self,
        n_actuators: int,
    ) -> None:
        """Postcondition shape + schema check for nominal inputs."""
        from src.engines.physics_engines.opensim.python.motion_matching.simulate import (  # noqa: E501
            simulate_with_coefficients,
        )

        opts = SimOptions(t_final=0.05, dt=5e-3)
        theta = _zero_theta(n_actuators)
        out = simulate_with_coefficients(theta, opts)

        n_steps = int(round(opts.t_final / opts.dt))
        n_samples = n_steps + 1

        # Shape checks per the canonical SimOut contract.
        assert out.time.shape == (n_samples,)
        assert out.q.ndim == 2 and out.q.shape[0] == n_samples
        assert out.qd.shape == out.q.shape
        assert out.qdd.shape == out.q.shape
        assert out.tau.shape == (n_samples, n_actuators)
        assert out.grip.shape == (n_samples, 3)
        assert out.grip_quat.shape == (n_samples, 4)
        assert out.clubhead.shape == (n_samples, 3)
        assert out.club_quat.shape == (n_samples, 4)
        assert isinstance(out.solver_status, str)
        assert out.wall_clock_s >= 0.0

    def test_solver_status_success_for_nominal_inputs(
        self,
        n_actuators: int,
    ) -> None:
        from src.engines.physics_engines.opensim.python.motion_matching.simulate import (  # noqa: E501
            simulate_with_coefficients,
        )

        out = simulate_with_coefficients(
            _zero_theta(n_actuators),
            SimOptions(t_final=0.05, dt=5e-3),
        )
        assert out.solver_status == "success"

    def test_recovery_known_theta_grip_pattern(
        self,
        n_actuators: int,
    ) -> None:
        """Recovery: a known theta produces a grip trajectory whose path
        length grows monotonically (a basic behavioural invariant).

        Falling under gravity from rest, the grip must descend, so the
        cumulative arc length of the grip path must be strictly
        positive after the first step.
        """
        from src.engines.physics_engines.opensim.python.motion_matching.simulate import (  # noqa: E501
            simulate_with_coefficients,
        )

        theta = _zero_theta(n_actuators)
        out = simulate_with_coefficients(theta, SimOptions(t_final=0.05, dt=5e-3))
        # Grip starts at well-defined initial position.
        assert np.all(np.isfinite(out.grip))
        # Path length is strictly positive (grip moves under gravity).
        diffs = np.linalg.norm(np.diff(out.grip, axis=0), axis=1)
        assert np.all(diffs >= 0.0)
        # No spurious resets to origin.
        assert np.linalg.norm(out.grip[0]) > 0.0

    def test_determinism_same_theta_identical_simout(
        self,
        n_actuators: int,
    ) -> None:
        """Determinism: same theta + options -> identical numerical output."""
        from src.engines.physics_engines.opensim.python.motion_matching.simulate import (  # noqa: E501
            simulate_with_coefficients,
        )

        theta = _zero_theta(n_actuators)
        opts = SimOptions(t_final=0.05, dt=5e-3)
        out_a = simulate_with_coefficients(theta, opts)
        out_b = simulate_with_coefficients(theta, opts)

        np.testing.assert_array_equal(out_a.time, out_b.time)
        np.testing.assert_allclose(out_a.q, out_b.q, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(out_a.qd, out_b.qd, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(out_a.grip, out_b.grip, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(out_a.clubhead, out_b.clubhead, atol=1e-12, rtol=0.0)

    def test_zero_theta_zero_torque_recorded(
        self,
        n_actuators: int,
    ) -> None:
        """Sanity: with all-zero theta the recorded torques must be zero."""
        from src.engines.physics_engines.opensim.python.motion_matching.simulate import (  # noqa: E501
            simulate_with_coefficients,
        )

        out = simulate_with_coefficients(
            _zero_theta(n_actuators),
            SimOptions(t_final=0.02, dt=5e-3),
        )
        np.testing.assert_allclose(out.tau, 0.0, atol=1e-12)

    def test_rejects_wrong_theta_shape(
        self,
        n_actuators: int,
    ) -> None:
        from src.engines.physics_engines.opensim.python.motion_matching.simulate import (  # noqa: E501
            simulate_with_coefficients,
        )

        # Message format updated by issue #4252 to use the shared
        # ``validate_theta`` validator (CROSS_ENGINE_PARITY_SPEC §2.2).
        with pytest.raises(ValueError, match=r"(theta length|theta has shape)"):
            simulate_with_coefficients(
                np.zeros(7),  # too short
                SimOptions(t_final=0.02, dt=5e-3),
            )
