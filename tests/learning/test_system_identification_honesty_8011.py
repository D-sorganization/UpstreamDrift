"""Regression tests for issue #8011.

Before the fix, ``SystemIdentifier`` reported ``converged=True`` with the
initial parameter vector whenever the engine lacked the getters it probed via
``hasattr`` - which was every engine in the repository. It also mis-aligned
``param_vector`` indices when a subset of parameters was requested, and
returned ``com_offset_* = 1.0`` against declared bounds of +/-0.05.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.learning.imitation.dataset import Demonstration
from src.learning.sim2real.system_identification import (
    SystemIdentifier,
    UnsupportedParameterError,
)

pytestmark = pytest.mark.unit

TRUE_MASS_SCALE = 1.7


class MassEngine:
    """Point-mass engine whose only tunable parameter is link mass."""

    def __init__(self, mass_scale: float = 1.0, n: int = 2) -> None:
        self.n = n
        self._masses = np.ones(n) * mass_scale
        self._q = np.zeros(n)
        self._v = np.zeros(n)
        self._tau = np.zeros(n)

    def get_link_masses(self) -> np.ndarray:
        return self._masses.copy()

    def set_link_masses(self, masses: np.ndarray) -> None:
        self._masses = np.asarray(masses, dtype=float).copy()

    def set_joint_positions(self, q: np.ndarray) -> None:
        self._q = np.asarray(q, dtype=float).copy()

    def set_joint_velocities(self, v: np.ndarray) -> None:
        self._v = np.asarray(v, dtype=float).copy()

    def set_joint_torques(self, tau: np.ndarray) -> None:
        self._tau = np.asarray(tau, dtype=float).copy()

    def step(self, dt: float) -> None:
        self._v = self._v + (self._tau / self._masses) * dt
        self._q = self._q + self._v * dt

    def get_joint_positions(self) -> np.ndarray:
        return self._q.copy()

    def get_joint_velocities(self) -> np.ndarray:
        return self._v.copy()


class NoHookEngine:
    """Standard state surface, but no parameter getters/setters at all."""

    def __init__(self) -> None:
        self._inner = MassEngine()

    def __getattr__(self, name: str) -> object:
        if name in {"get_link_masses", "set_link_masses"}:
            raise AttributeError(name)
        return getattr(self._inner, name)


class NoOpSetterEngine(MassEngine):
    """Exposes both hooks, but the setter is a deferred no-op (Simscape)."""

    def set_link_masses(self, masses: np.ndarray) -> None:  # noqa: ARG002
        return None


def _make_demo(n: int = 50, seed: int = 0) -> Demonstration:
    """Record a trajectory from an engine whose mass_scale is TRUE_MASS_SCALE."""
    rng = np.random.default_rng(seed)
    real = MassEngine(mass_scale=TRUE_MASS_SCALE)
    dt = 0.01
    tau = rng.normal(size=(n, 2)) * 0.5
    qs, vs = [], []
    for i in range(n):
        qs.append(real.get_joint_positions())
        vs.append(real.get_joint_velocities())
        real.set_joint_torques(tau[i])
        real.step(dt)
    return Demonstration(
        timestamps=np.arange(n) * dt,
        joint_positions=np.array(qs),
        joint_velocities=np.array(vs),
        actions=tau,
    )


class TestIdentificationIsRealOrLoud:
    def test_recovers_ground_truth_when_hooks_work(self) -> None:
        """With a working setter the optimiser must find the true value."""
        sim = MassEngine(mass_scale=1.0)
        ident = SystemIdentifier(sim)
        result = ident.identify_from_trajectories(
            [_make_demo()],
            params_to_identify=["mass_scale"],
            max_iterations=200,
            tolerance=1e-14,
        )
        assert result.converged
        assert result.identified_params["mass_scale"] == pytest.approx(
            TRUE_MASS_SCALE, abs=1e-3
        )
        assert result.residual_error < 1e-12

    def test_missing_hooks_raise_instead_of_reporting_convergence(self) -> None:
        """No getter/setter -> loud failure, not converged=True with 1.0."""
        ident = SystemIdentifier(NoHookEngine())
        assert ident.supported_parameters() == []
        with pytest.raises(UnsupportedParameterError, match="get_link_masses"):
            ident.identify_from_trajectories(
                [_make_demo()], params_to_identify=["mass_scale"]
            )

    def test_no_op_setter_is_detected(self) -> None:
        """A setter that does not change the getter's value is not support."""
        ident = SystemIdentifier(NoOpSetterEngine())
        assert "mass_scale" not in ident.supported_parameters()
        with pytest.raises(UnsupportedParameterError):
            ident.identify_from_trajectories(
                [_make_demo()], params_to_identify=["mass_scale"]
            )

    def test_default_parameter_set_raises_when_nothing_supported(self) -> None:
        """params_to_identify=None must not silently 'identify' nothing."""
        ident = SystemIdentifier(NoHookEngine())
        with pytest.raises(UnsupportedParameterError):
            ident.identify_from_trajectories([_make_demo()])

    def test_probe_restores_nominal_values(self) -> None:
        """supported_parameters() must leave the model exactly as it found it."""
        sim = MassEngine(mass_scale=1.3)
        before = sim.get_link_masses()
        ident = SystemIdentifier(sim)
        ident.supported_parameters()
        np.testing.assert_allclose(sim.get_link_masses(), before)


class TestParameterAlignment:
    def test_subset_does_not_leak_into_another_parameter(self) -> None:
        """Requesting friction only must never write into the mass branch."""
        sim = MassEngine(mass_scale=1.0)
        ident = SystemIdentifier(sim)
        with pytest.raises(UnsupportedParameterError, match="friction"):
            ident.identify_from_trajectories(
                [_make_demo()], params_to_identify=["friction_scale"]
            )
        np.testing.assert_allclose(sim.get_link_masses(), np.ones(2))

    def test_apply_params_uses_supplied_names(self) -> None:
        """_apply_params indexes by the names given, not by param_bounds order."""
        sim = MassEngine(mass_scale=1.0)
        ident = SystemIdentifier(sim)
        ident._apply_params(np.array([2.0]), ["mass_scale"])
        np.testing.assert_allclose(sim.get_link_masses(), np.ones(2) * 2.0)

    def test_unknown_parameter_rejected(self) -> None:
        ident = SystemIdentifier(MassEngine())
        with pytest.raises(ValueError, match="Unknown parameter"):
            ident.identify_from_trajectories(
                [_make_demo()], params_to_identify=["not_a_param"]
            )

    def test_short_param_vector_rejected(self) -> None:
        ident = SystemIdentifier(MassEngine())
        with pytest.raises(ValueError, match="param_vector has"):
            ident._apply_params(np.array([]), ["mass_scale"])


class TestBoundsAreRespected:
    def test_no_default_parameter_starts_outside_its_bounds(self) -> None:
        """The initial point must lie inside every declared bound (#8011)."""
        ident = SystemIdentifier(MassEngine())
        names = list(ident.param_bounds)
        start = ident._nominal_vector(names)
        for value, name in zip(start, names, strict=True):
            low, high = ident.param_bounds[name]
            assert low <= value <= high, f"{name}={value} outside {(low, high)}"

    def test_com_offset_removed_from_defaults(self) -> None:
        """com_offset_* had no implementation and is no longer advertised."""
        ident = SystemIdentifier(MassEngine())
        assert not [n for n in ident.param_bounds if n.startswith("com_offset")]

    def test_additive_parameter_starts_at_zero_within_bounds(self) -> None:
        """A caller-supplied offset bound gets 0.0, not 1.0."""
        ident = SystemIdentifier(
            MassEngine(), param_bounds={"com_offset_x": (-0.05, 0.05)}
        )
        assert ident._nominal_vector(["com_offset_x"])[0] == pytest.approx(0.0)


class TestEmptyInputs:
    def test_empty_trajectories_rejected(self) -> None:
        ident = SystemIdentifier(MassEngine())
        with pytest.raises(ValueError, match="must not be empty"):
            ident.identify_from_trajectories([], params_to_identify=["mass_scale"])

    def test_validate_identification_rejects_unsupported(self) -> None:
        ident = SystemIdentifier(NoHookEngine())
        with pytest.raises(UnsupportedParameterError):
            ident.validate_identification([_make_demo()], {"mass_scale": 1.2})
