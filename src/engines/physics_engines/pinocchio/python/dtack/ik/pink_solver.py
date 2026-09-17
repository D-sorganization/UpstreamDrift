"""Validated one-step inverse kinematics adapters for Pink."""

from __future__ import annotations

import inspect
import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

# Guard Pink and Pinocchio imports: both are optional heavy dependencies.
try:
    import pink
    import pinocchio as pin
    from pink import Task

    PINK_SOLVER_AVAILABLE = True
except (ImportError, OSError):
    PINK_SOLVER_AVAILABLE = False
    pink = None  # type: ignore[assignment]
    pin = None  # type: ignore[assignment]
    Task = Any  # type: ignore[assignment,misc]

Array = npt.NDArray[np.float64]

_INSTALL_HINT = (
    "Optional Pink dependencies are required; see docs/engines/pinocchio.md."
)


@dataclass(frozen=True)
class SolverSettings:
    """Settings for one Pink differential-IK step."""

    solver: str = "quadprog"
    damping: float = 1e-6


@dataclass(frozen=True)
class _PinkStepOptions:
    """Validated inputs that control one Pink solve step."""

    dt: float
    settings: SolverSettings
    constraints: Iterable[Task] | None = None
    limits: Iterable[Any] | None = None


@dataclass
class _PinkConfigurationCache:
    """Mutable Pink configuration state shared across solve attempts."""

    configuration: Any | None = None
    collision_model: Any | None = None


class _PinkStepResult(NamedTuple):
    q_next: Array
    configuration: Any


def _finite_vector(value: Any, size: int, name: str) -> Array:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric vector") from exc
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must be finite")
    return vector


def _model_dimension(model: Any, name: str) -> int:
    value = getattr(model, name, None)
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"model.{name} must be a nonnegative integer")
    return int(value)


def _validate_settings(dt: float, solver: str, damping: float) -> None:
    if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if not isinstance(solver, str) or not solver.strip():
        raise ValueError("solver must be a nonempty string")
    if (
        not isinstance(damping, (int, float))
        or not math.isfinite(damping)
        or damping < 0.0
    ):
        raise ValueError("damping must be nonnegative and finite")


def _require_solve_keyword(name: str) -> None:
    """Fail before solving when the installed Pink API lacks a requested input."""
    try:
        parameters = inspect.signature(pink.solve_ik).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Installed Pink solve_ik API cannot verify support for {name}"
        ) from exc
    if name not in parameters:
        raise RuntimeError(f"Installed Pink solve_ik API does not support {name}")


def _solve_pink_step(
    model: Any,
    data: Any,
    q_init: Any,
    tasks: Iterable[Task],
    options: _PinkStepOptions,
    cache: _PinkConfigurationCache,
) -> _PinkStepResult:
    """Run the single validated Pink solve path shared by both public adapters."""
    if not PINK_SOLVER_AVAILABLE:
        raise ImportError(f"Pink and Pinocchio are required. {_INSTALL_HINT}")
    settings = options.settings
    _validate_settings(options.dt, settings.solver, settings.damping)
    nq = _model_dimension(model, "nq")
    nv = _model_dimension(model, "nv")
    q = _finite_vector(q_init, nq, "q_init")

    configuration = cache.configuration
    if configuration is None:
        configuration_kwargs = (
            {"collision_model": cache.collision_model}
            if cache.collision_model is not None
            else {}
        )
        configuration = pink.Configuration(model, data, q, **configuration_kwargs)
        cache.configuration = configuration
    else:
        configuration.update(q)

    solve_kwargs: dict[str, Any] = {
        "solver": settings.solver,
        "damping": settings.damping,
    }
    if options.constraints is not None:
        _require_solve_keyword("constraints")
        solve_kwargs["constraints"] = options.constraints
    if options.limits is not None:
        _require_solve_keyword("limits")
        solve_kwargs["limits"] = options.limits

    try:
        velocity_raw = pink.solve_ik(configuration, tasks, options.dt, **solve_kwargs)
    except Exception as exc:
        exc.add_note(
            "Pink IK solve failed with "
            f"solver={settings.solver!r}, dt={options.dt}, "
            f"damping={settings.damping}"
        )
        raise
    velocity = _finite_vector(velocity_raw, nv, "velocity")

    try:
        integrated = pin.integrate(model, q, velocity * options.dt)
    except Exception as exc:
        exc.add_note("Pinocchio integration failed after the Pink IK solve")
        raise
    q_next = _finite_vector(integrated, nq, "q_next")
    configuration.update(q_next)
    return _PinkStepResult(q_next=q_next, configuration=configuration)


class PinkSolver:
    """Inverse kinematics solver wrapper for one validated Pink step."""

    def __init__(
        self,
        robot_model: Any,
        robot_data: Any,
        robot_visual: Any,
        robot_collision: Any,
    ) -> None:
        """Initialize while preserving distinct visual and collision models."""
        if not PINK_SOLVER_AVAILABLE:
            raise ImportError(
                f"Pink and Pinocchio are required for PinkSolver. {_INSTALL_HINT}"
            )
        self.model = robot_model
        self.data = robot_data
        self.visual_model = robot_visual
        self.collision_model = robot_collision
        self._cache = _PinkConfigurationCache(collision_model=robot_collision)

    @property
    def configuration(self) -> Any | None:
        """Return configuration state from the most recent solve attempt."""
        return self._cache.configuration

    def solve(
        self,
        q_init: Array,
        tasks: list[Task],
        dt: float,
        settings: SolverSettings | None = None,
        *,
        constraints: Iterable[Task] | None = None,
        limits: Iterable[Any] | None = None,
    ) -> Array:
        """Solve and integrate one differential-IK step.

        Tasks are weighted costs. ``constraints`` are hard task equalities and
        ``limits`` are Pink limit objects. ``None`` preserves Pink's defaults.
        After input validation, the configuration cache is refreshed to ``q_init``
        before the QP solve and to the integrated result after success.
        """
        effective_settings = settings if settings is not None else SolverSettings()
        options = _PinkStepOptions(
            dt=dt,
            settings=effective_settings,
            constraints=constraints,
            limits=limits,
        )
        result = _solve_pink_step(
            self.model,
            self.data,
            q_init,
            tasks,
            options,
            self._cache,
        )
        return result.q_next
