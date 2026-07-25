"""System Identification for sim-to-real transfer."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.engines.protocols import PhysicsEngineProtocol
    from src.learning.imitation.dataset import Demonstration


class UnsupportedParameterError(NotImplementedError):
    """Raised when a requested parameter cannot be applied to the model.

    A parameter is identifiable only when the engine exposes **both** a
    working getter and a working setter for it. If either is missing - or
    the setter is a deferred no-op - optimising over that parameter would
    return the initial value dressed up as a result (issue #8011), so the
    identifier refuses to run instead.
    """


#: Parameter name -> (nominal-cache key, getter name, setter name).
#: Only parameters listed here can be identified; anything else in
#: ``param_bounds`` has no implementation and is rejected loudly.
_PARAM_HOOKS: dict[str, tuple[str, str, str]] = {
    "mass_scale": ("masses", "get_link_masses", "set_link_masses"),
    "friction_scale": (
        "friction",
        "get_friction_coefficients",
        "set_friction_coefficients",
    ),
    "damping_scale": ("damping", "get_joint_damping", "set_joint_damping"),
    "motor_scale": ("motor", "get_motor_strength", "set_motor_strength"),
}

#: Additive (rather than multiplicative) parameters start from 0.0.
_ADDITIVE_PARAM_PREFIXES = ("com_offset",)


@dataclass
class IdentificationResult:
    """Result of system identification.

    Attributes:
        identified_params: Dictionary of identified parameters.
        residual_error: Final optimization residual.
        iterations: Number of optimization iterations.
        converged: Whether optimization converged.
    """

    identified_params: dict[str, float | NDArray[np.floating]]
    residual_error: float
    iterations: int
    converged: bool


class SystemIdentifier:
    """Identify real robot parameters from data.

    Uses optimization to find simulation parameters that best
    match observed real robot behavior.

    Attributes:
        model: Physics engine model.
        param_bounds: Parameter bounds for optimization.
    """

    def __init__(
        self,
        model: PhysicsEngineProtocol,
        param_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Initialize system identifier.

        Args:
            model: Physics engine to tune.
            param_bounds: Bounds for each parameter.
        """
        if model is None:
            raise ValueError("model must be provided")
        self.model = model
        self.param_bounds = param_bounds or self._default_bounds()
        self._nominal_params = self._get_current_params()
        self._supported_cache: list[str] | None = None

    def _default_bounds(self) -> dict[str, tuple[float, float]]:
        """Get default parameter bounds.

        ``com_offset_x/y/z`` were removed in the fix for #8011: no engine
        exposes a centre-of-mass offset hook, and ``_apply_params`` never had
        an implementation for them, so including them only produced
        "identified" values of 1.0 against declared bounds of +/-0.05.

        Returns:
            Dictionary of parameter bounds.
        """
        return {
            "mass_scale": (0.5, 2.0),
            "friction_scale": (0.2, 3.0),
            "damping_scale": (0.5, 2.0),
            "motor_scale": (0.5, 1.5),
        }

    def _get_current_params(self) -> dict[str, Any]:
        """Get current model parameters.

        Returns:
            Dictionary of current parameters, keyed by the nominal-cache key
            in :data:`_PARAM_HOOKS`.
        """
        params: dict[str, Any] = {}
        for key, getter, _setter in _PARAM_HOOKS.values():
            fn = getattr(self.model, getter, None)
            if fn is None:
                continue
            params[key] = np.asarray(fn(), dtype=float).copy()
        return params

    def supported_parameters(self) -> list[str]:
        """Return the parameters this model can actually be tuned through.

        A parameter qualifies only when the engine exposes a getter and a
        setter *and* the setter demonstrably changes what the getter returns.
        The round-trip probe is what catches deferred no-op setters such as
        ``SimscapeAdapter.set_link_masses`` (deferred to #4006), which would
        otherwise make the optimisation silently vacuous.

        Returns:
            Parameter names from ``param_bounds`` that can be identified,
            in ``param_bounds`` order.
        """
        if self._supported_cache is not None:
            return list(self._supported_cache)

        supported: list[str] = []
        for name in self.param_bounds:
            hooks = _PARAM_HOOKS.get(name)
            if hooks is None:
                continue
            key, getter, setter = hooks
            if not hasattr(self.model, getter) or not hasattr(self.model, setter):
                continue
            if key not in self._nominal_params:
                continue
            if self._setter_is_effective(key, getter, setter):
                supported.append(name)

        self._supported_cache = supported
        return list(supported)

    def _setter_is_effective(self, key: str, getter: str, setter: str) -> bool:
        """Probe whether ``setter`` measurably changes what ``getter`` returns.

        Args:
            key: Nominal-cache key holding the untouched value.
            getter: Name of the model's getter method.
            setter: Name of the model's setter method.

        Returns:
            True if a perturbation round-trips; False for missing, empty or
            no-op parameter surfaces.
        """
        nominal = np.asarray(self._nominal_params[key], dtype=float)
        if nominal.size == 0:
            return False
        probe = nominal * 1.5 + 0.5
        try:
            getattr(self.model, setter)(probe)
            observed = np.asarray(getattr(self.model, getter)(), dtype=float)
        except (TypeError, ValueError, NotImplementedError, AttributeError):
            return False
        finally:
            # Always restore the nominal value, even if the probe raised.
            with contextlib.suppress(
                TypeError, ValueError, NotImplementedError, AttributeError
            ):
                getattr(self.model, setter)(nominal.copy())
        return bool(
            observed.shape == probe.shape and not np.allclose(observed, nominal)
        )

    def _validate_parameters(self, param_names: list[str]) -> None:
        """Reject parameters that cannot be applied to this model.

        Args:
            param_names: Requested parameter names.

        Raises:
            ValueError: If a name is not in ``param_bounds``.
            UnsupportedParameterError: If a name has no working getter/setter
                pair on the model.
        """
        unknown = [n for n in param_names if n not in self.param_bounds]
        if unknown:
            raise ValueError(
                f"Unknown parameter(s) {unknown}; "
                f"param_bounds declares {list(self.param_bounds)}"
            )
        supported = set(self.supported_parameters())
        missing = [n for n in param_names if n not in supported]
        if missing:
            details = []
            for name in missing:
                hooks = _PARAM_HOOKS.get(name)
                if hooks is None:
                    details.append(f"{name}: no implementation in SystemIdentifier")
                else:
                    _key, getter, setter = hooks
                    details.append(f"{name}: requires working {getter}()/{setter}()")
            raise UnsupportedParameterError(
                "Cannot identify parameter(s) on "
                f"{type(self.model).__name__}: " + "; ".join(details) + ". "
                "Identifying them would return the initial values unchanged, "
                "so the run is refused (issue #8011)."
            )

    def _nominal_vector(self, param_names: list[str]) -> NDArray[np.floating]:
        """Build the bound-respecting starting point for ``param_names``.

        Scale parameters start at 1.0 and additive parameters at 0.0, each
        clipped into its declared bounds so the initial point can never be
        reported as an out-of-bounds "identified" value (issue #8011).
        """
        values = []
        for name in param_names:
            nominal = 0.0 if name.startswith(_ADDITIVE_PARAM_PREFIXES) else 1.0
            low, high = self.param_bounds[name]
            values.append(float(np.clip(nominal, low, high)))
        return np.array(values, dtype=float)

    def _apply_params(
        self,
        param_vector: NDArray[np.floating],
        param_names: list[str] | None = None,
    ) -> None:
        """Apply a parameter vector to the model.

        Args:
            param_vector: Values aligned positionally with ``param_names``.
            param_names: Names the vector refers to. Defaults to the full
                ``param_bounds`` ordering.

        Raises:
            ValueError: If ``param_vector`` is missing or too short.
            UnsupportedParameterError: If a name has no implementation.
        """
        if param_vector is None:
            raise ValueError("param_vector must be provided")
        names = list(self.param_bounds) if param_names is None else list(param_names)
        if len(param_vector) < len(names):
            raise ValueError(
                f"param_vector has {len(param_vector)} entries but "
                f"{len(names)} parameters were requested"
            )

        for idx, name in enumerate(names):
            hooks = _PARAM_HOOKS.get(name)
            if hooks is None:
                raise UnsupportedParameterError(
                    f"No implementation for parameter '{name}' (issue #8011)"
                )
            key, _getter, setter = hooks
            if key not in self._nominal_params:
                raise UnsupportedParameterError(
                    f"Model exposes no nominal value for '{name}' (issue #8011)"
                )
            getattr(self.model, setter)(self._nominal_params[key] * param_vector[idx])

    def _simulate_trajectory(
        self,
        initial_state: NDArray[np.floating],
        actions: NDArray[np.floating],
        dt: float,
    ) -> NDArray[np.floating]:
        """Simulate a trajectory with current parameters.

        Args:
            initial_state: Initial joint positions and velocities.
            actions: Action sequence to apply.
            dt: Timestep.

        Returns:
            Simulated state trajectory.
        """
        if initial_state is None:
            raise ValueError("initial_state must be provided")
        n_steps = len(actions)
        n_q = len(initial_state) // 2
        states = [initial_state.copy()]

        # Set initial state
        q0 = initial_state[:n_q]
        v0 = initial_state[n_q:]

        if hasattr(self.model, "set_joint_positions"):
            self.model.set_joint_positions(q0)
        if hasattr(self.model, "set_joint_velocities"):
            self.model.set_joint_velocities(v0)

        for i in range(n_steps):
            # Apply action
            if hasattr(self.model, "set_joint_torques"):
                self.model.set_joint_torques(actions[i])

            # Step simulation. PhysicsEngineProtocol.step() takes no timestep,
            # but every concrete engine used here accepts one, so the call is
            # dispatched dynamically.
            step = getattr(self.model, "step", None)
            if callable(step):
                step(dt)

            # Record state
            if hasattr(self.model, "get_joint_positions"):
                q = self.model.get_joint_positions()
            else:
                q = np.zeros(n_q)

            if hasattr(self.model, "get_joint_velocities"):
                v = self.model.get_joint_velocities()
            else:
                v = np.zeros(n_q)

            states.append(np.concatenate([q, v]))

        return np.array(states)

    def _compute_trajectory_error(
        self,
        sim_trajectory: NDArray[np.floating],
        real_trajectory: NDArray[np.floating],
        weights: NDArray[np.floating] | None = None,
    ) -> float:
        """Compute error between simulated and real trajectory.

        Args:
            sim_trajectory: Simulated state trajectory.
            real_trajectory: Real robot state trajectory.
            weights: State component weights.

        Returns:
            Weighted mean squared error.
        """
        # Ensure same length
        if sim_trajectory is None:
            raise ValueError("sim_trajectory must be provided")
        n = min(len(sim_trajectory), len(real_trajectory))
        sim = sim_trajectory[:n]
        real = real_trajectory[:n]

        diff = sim - real

        if weights is not None:
            diff = diff * weights

        return float(np.vdot(diff, diff) / diff.size)

    def identify_from_trajectories(
        self,
        trajectories: list[Demonstration],
        params_to_identify: list[str] | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> IdentificationResult:
        """Identify parameters from real robot trajectories.

        Uses gradient-free optimization to find parameters
        that minimize trajectory prediction error.

        Args:
            trajectories: List of real robot demonstrations.
            params_to_identify: Which parameters to identify. Defaults to
                every parameter the model actually supports.
            max_iterations: Maximum optimization iterations.
            tolerance: Convergence tolerance.

        Returns:
            Identification result. ``identified_params`` is keyed by, and
            aligned with, ``params_to_identify``.

        Raises:
            ValueError: If ``trajectories`` is missing/empty or a requested
                parameter is not declared in ``param_bounds``.
            UnsupportedParameterError: If the model cannot apply a requested
                parameter (see :meth:`supported_parameters`).
        """
        if trajectories is None:
            raise ValueError("trajectories must be provided")
        if not trajectories:
            raise ValueError("trajectories must not be empty")
        if params_to_identify is None:
            params_to_identify = self.supported_parameters()
            if not params_to_identify:
                self._validate_parameters(list(self.param_bounds))
        self._validate_parameters(params_to_identify)

        n_params = len(params_to_identify)
        param_vector = self._nominal_vector(params_to_identify)

        lower_bounds = np.array([self.param_bounds[p][0] for p in params_to_identify])
        upper_bounds = np.array([self.param_bounds[p][1] for p in params_to_identify])

        def objective(params: NDArray[np.floating]) -> float:
            """Compute total error over all trajectories."""
            return self._evaluate_params(params, trajectories, params_to_identify)

        best_params = param_vector.copy()
        best_error = objective(best_params)

        best_params, best_error, converged, _iteration = self._coordinate_descent(  # type: ignore[assignment]
            objective,
            best_params,
            best_error,
            lower_bounds,
            upper_bounds,
            n_params,
            max_iterations,
            tolerance,
        )

        identified = {}
        for i, name in enumerate(params_to_identify):
            identified[name] = float(best_params[i])

        return IdentificationResult(
            identified_params=identified,  # type: ignore[arg-type]
            residual_error=best_error,
            iterations=_iteration + 1,
            converged=converged,
        )

    def _evaluate_params(
        self,
        params: NDArray[np.floating],
        trajectories: list[Demonstration],
        param_names: list[str] | None = None,
    ) -> float:
        if params is None:
            raise ValueError("params must be provided")
        self._apply_params(params, param_names)
        total_error = 0.0

        # initial_state / real_traj / dt depend only on the (fixed) demos, not
        # on `params`, but _evaluate_params is called once per coordinate-descent
        # probe. Precompute them once per trajectory set instead of rebuilding
        # the concatenations and dt on every probe.
        prepared = self._prepared_trajectories(trajectories)
        if not prepared:
            raise ValueError(
                "No demonstration carried actions; system identification needs "
                "recorded control inputs"
            )
        for initial_state, actions, dt, real_traj in prepared:
            sim_traj = self._simulate_trajectory(initial_state, actions, dt)
            total_error += self._compute_trajectory_error(sim_traj, real_traj)

        return total_error / len(trajectories)

    def _prepared_trajectories(
        self,
        trajectories: list[Demonstration],
    ) -> list[tuple[NDArray[np.floating], Any, float, NDArray[np.floating]]]:
        """Cache per-demo (initial_state, actions, dt, real_traj) tuples.

        Keyed by the identity of the trajectory list so repeated probes during
        coordinate descent reuse the concatenations and dt instead of rebuilding
        them each call. Values are identical to the inline computation.
        """
        cache = getattr(self, "_prepared_traj_cache", None)
        if cache is not None and cache[0] is trajectories:
            return cache[1]

        prepared: list[
            tuple[NDArray[np.floating], Any, float, NDArray[np.floating]]
        ] = []
        for demo in trajectories:
            if demo.actions is None:
                continue
            initial_state = np.concatenate(
                [demo.joint_positions[0], demo.joint_velocities[0]],
            )
            real_traj = np.concatenate(
                [demo.joint_positions, demo.joint_velocities],
                axis=1,
            )
            dt = float(np.mean(np.diff(demo.timestamps)))
            prepared.append((initial_state, demo.actions, dt, real_traj))

        self._prepared_traj_cache = (trajectories, prepared)
        return prepared

    def _coordinate_descent(
        self,
        objective: Any,
        best_params: NDArray[np.floating],
        best_error: float,
        lower_bounds: NDArray[np.floating],
        upper_bounds: NDArray[np.floating],
        n_params: int,
        max_iterations: int,
        tolerance: float,
    ) -> tuple[NDArray[np.floating], float, bool, int]:
        if objective is None:
            raise ValueError("objective must be provided")
        converged = False
        _iteration = 0
        base_deltas = (0.1, -0.1, 0.05, -0.05, 0.01, -0.01)
        step_scale = 1.0
        min_step_scale = 1e-2

        for _iteration in range(max_iterations):
            improved = False

            for i in range(n_params):
                for base in base_deltas:
                    test_params = best_params.copy()
                    test_params[i] = np.clip(
                        test_params[i] + base * step_scale,
                        lower_bounds[i],
                        upper_bounds[i],
                    )

                    error = objective(test_params)
                    if error < best_error - tolerance:
                        best_error = error
                        best_params = test_params.copy()
                        improved = True

            if not improved:
                # Refine the step size before declaring convergence, so
                # "converged" means "no smaller step helps either" rather than
                # "the coarse 0.01 grid stalled" (issue #8011).
                if step_scale > min_step_scale:
                    step_scale *= 0.5
                    continue
                converged = True
                break

        return best_params, best_error, converged, _iteration

    def compute_reality_gap(
        self,
        sim_trajectory: NDArray[np.floating],
        real_trajectory: NDArray[np.floating],
    ) -> dict[str, float]:
        """Quantify the sim-to-real gap.

        Args:
            sim_trajectory: Simulated state trajectory.
            real_trajectory: Real robot trajectory.

        Returns:
            Dictionary of gap metrics.
        """
        if sim_trajectory is None:
            raise ValueError("sim_trajectory must be provided")
        n = min(len(sim_trajectory), len(real_trajectory))
        sim = sim_trajectory[:n]
        real = real_trajectory[:n]

        diff = sim - real
        n_dof = sim.shape[1] // 2

        metrics = {
            "total_mse": float(np.vdot(diff, diff) / diff.size),
            "position_mse": float(
                np.vdot(diff[:, :n_dof], diff[:, :n_dof]) / diff[:, :n_dof].size
            ),
            "velocity_mse": float(
                np.vdot(diff[:, n_dof:], diff[:, n_dof:]) / diff[:, n_dof:].size
            ),
            "max_position_error": float(np.max(np.abs(diff[:, :n_dof]))),
            "max_velocity_error": float(np.max(np.abs(diff[:, n_dof:]))),
            "mean_position_error": float(np.mean(np.abs(diff[:, :n_dof]))),
            "mean_velocity_error": float(np.mean(np.abs(diff[:, n_dof:]))),
            "trajectory_length": n,
        }

        # Per-joint errors
        for j in range(n_dof):
            metrics[f"joint_{j}_position_mse"] = float(
                np.vdot(diff[:, j], diff[:, j]) / diff[:, j].size
            )
            metrics[f"joint_{j}_velocity_mse"] = float(
                np.vdot(diff[:, n_dof + j], diff[:, n_dof + j])
                / diff[:, n_dof + j].size
            )

        return metrics

    def validate_identification(
        self,
        test_trajectories: list[Demonstration],
        identified_params: dict[str, float | NDArray[np.floating]],
    ) -> dict[str, float]:
        """Validate identified parameters on test data.

        Args:
            test_trajectories: Held-out test demonstrations.
            identified_params: Previously identified parameters.

        Returns:
            Validation metrics.

        Raises:
            ValueError: If ``test_trajectories`` is missing, or none of the
                demonstrations carry actions.
            UnsupportedParameterError: If a supplied parameter has no
                implementation on this model.
        """
        # Apply identified parameters (only those actually supplied, so the
        # vector cannot silently drift out of alignment - issue #8011).
        if test_trajectories is None:
            raise ValueError("test_trajectories must be provided")
        param_names = list(identified_params)
        self._validate_parameters(param_names)
        param_vector = np.array(
            [float(np.asarray(identified_params[name]).item()) for name in param_names],
        )
        self._apply_params(param_vector, param_names)

        # Compute errors on test set
        errors = []
        for demo in test_trajectories:
            if demo.actions is None:
                continue

            initial_state = np.concatenate(
                [
                    demo.joint_positions[0],
                    demo.joint_velocities[0],
                ],
            )

            real_traj = np.concatenate(
                [
                    demo.joint_positions,
                    demo.joint_velocities,
                ],
                axis=1,
            )

            dt = float(np.mean(np.diff(demo.timestamps)))
            sim_traj = self._simulate_trajectory(initial_state, demo.actions, dt)

            error = self._compute_trajectory_error(sim_traj, real_traj)
            errors.append(error)

        if not errors:
            raise ValueError(
                "No test trajectory carried actions; nothing could be validated"
            )

        return {
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "max_error": float(np.max(errors)),
            "min_error": float(np.min(errors)),
            "n_trajectories": len(errors),
        }
