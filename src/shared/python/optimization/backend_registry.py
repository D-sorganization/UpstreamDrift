"""Capability-probed registry of swing-optimizer backends (#9760).

Six solver paths exist for the same seven-DOF swing. Before this module
the choice was a string compare buried in ``SwingOptimizer.optimize`` and
nothing said which backend is importable in the current environment or
which problem class it is for. The registry is that single place; the ADR
``docs/adr/0050-optimizer-backend-registry-and-bioptim.md`` records the
problem-class assignment.

Every ``solve`` callable shares one signature and returns a
:class:`CasadiSwingResult`-shaped object in the flagship decision layout,
so :class:`SwingOptimizer` needs no backend-specific branches. Backends
are probed, never imported, at registration time: importing this module
must succeed with no optional extra installed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

from src.shared.python.optimization._swing_kinematics import JOINTS
from src.shared.python.optimization._swing_models import (
    ClubModel,
    GolferModel,
    OptimizationConfig,
)
from src.shared.python.optimization.casadi_backend import (
    CasadiSolveOptions,
    CasadiSwingResult,
    casadi_available,
    solve_swing_casadi,
)
from src.shared.python.optimization.crocoddyl_backend import (
    crocoddyl_available,
)

__all__ = [
    "BackendNotAvailableError",
    "BackendSpec",
    "SolveFn",
    "available_backends",
    "get_backend",
    "list_backends",
    "register_backend",
    "require_backend",
]

SolveFn = Callable[
    [
        GolferModel,
        ClubModel,
        OptimizationConfig,
        dict[str, float],
        dict[str, tuple[float, float]],
        np.ndarray,
    ],
    CasadiSwingResult,
]


class BackendNotAvailableError(RuntimeError):
    """Raised by :func:`require_backend` when the backend's stack is absent."""


@dataclass(frozen=True)
class BackendSpec:
    """One swing-optimizer backend.

    Attributes:
        name: Registry key and the value of ``OptimizationConfig.solver``.
        problem_class: The problem this backend owns (see ADR-0050).
        description: One line for humans.
        available: Mock-tolerant probe; must not import the optional stack.
        install_hint: What to install when ``available()`` is false.
        solve: Flagship-layout solver, or ``None`` for the scipy fallback
            that :class:`SwingOptimizer` runs itself (it needs the iteration
            callback and the scipy method name).
        deprecated: Emit a ``DeprecationWarning`` when selected.
    """

    name: str
    problem_class: str
    description: str
    available: Callable[[], bool]
    install_hint: str
    solve: SolveFn | None = None
    deprecated: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("backend name must be non-empty")


_REGISTRY: dict[str, BackendSpec] = {}


def register_backend(spec: BackendSpec) -> BackendSpec:
    """Register (or replace) a backend. Returns ``spec`` for chaining."""
    _REGISTRY[spec.name] = spec
    return spec


def get_backend(name: str) -> BackendSpec | None:
    """Return the backend registered under ``name`` or ``None``."""
    return _REGISTRY.get(name)


def list_backends() -> list[BackendSpec]:
    """All registered backends in registration order."""
    return list(_REGISTRY.values())


def available_backends() -> list[BackendSpec]:
    """Backends whose optional stack is importable right now."""
    return [spec for spec in _REGISTRY.values() if spec.available()]


def require_backend(name: str) -> BackendSpec:
    """Return the named backend or raise with the install hint.

    Raises:
        KeyError: Unknown backend name.
        BackendNotAvailableError: Known backend whose stack is absent.
    """
    spec = _REGISTRY.get(name)
    if spec is None:
        raise KeyError(
            f"unknown optimizer backend {name!r}; known: {sorted(_REGISTRY)}"
        )
    if not spec.available():
        raise BackendNotAvailableError(
            f"backend {name!r} unavailable: {spec.install_hint}"
        )
    return spec


# --------------------------------------------------------------------------
# Built-in backends
# --------------------------------------------------------------------------


def _always() -> bool:
    return True


def _solve_casadi_fd(
    golfer: GolferModel,
    club: ClubModel,
    config: OptimizationConfig,
    torque_limits: dict[str, float],
    joint_limits: dict[str, tuple[float, float]],
    x0: np.ndarray,
) -> CasadiSwingResult:
    return solve_swing_casadi(golfer, club, config, torque_limits, joint_limits, x0)


def _solve_casadi_multiple_shooting(
    golfer: GolferModel,
    club: ClubModel,
    config: OptimizationConfig,
    torque_limits: dict[str, float],
    joint_limits: dict[str, tuple[float, float]],
    x0: np.ndarray,
) -> CasadiSwingResult:
    return solve_swing_casadi(
        golfer,
        club,
        config,
        torque_limits,
        joint_limits,
        x0,
        options=CasadiSolveOptions(transcription="multiple_shooting"),
    )


def _solve_crocoddyl(
    golfer: GolferModel,
    club: ClubModel,
    config: OptimizationConfig,
    torque_limits: dict[str, float],
    joint_limits: dict[str, tuple[float, float]],
    x0: np.ndarray,
) -> CasadiSwingResult:
    """Adapt the FDDP solve to the flagship node grid and layout."""
    from src.shared.python.optimization.crocoddyl_backend import solve_swing_ddp

    n = len(JOINTS)
    n_nodes = config.n_nodes
    outcome = solve_swing_ddp(
        golfer,
        club,
        horizon=n_nodes - 1,
        dt=config.swing_duration / (n_nodes - 1),
        max_iterations=int(config.max_iterations),
    )
    if not outcome.success or outcome.xs.size == 0:
        return CasadiSwingResult(
            success=False,
            x=np.asarray(x0, dtype=float),
            fun=float("nan"),
            message=outcome.message,
            iterations=outcome.iterations,
            transcription="crocoddyl-fddp",
        )
    q = np.asarray(outcome.xs[:, :n], dtype=float).T
    v = np.asarray(outcome.xs[:, n:], dtype=float).T
    return CasadiSwingResult(
        success=True,
        x=np.concatenate([q.flatten(), v.flatten()]),
        fun=float(outcome.cost),
        message=outcome.message,
        iterations=outcome.iterations,
        torques=np.asarray(outcome.us, dtype=float).T,
        transcription="crocoddyl-fddp",
    )


def _bioptim_available() -> bool:
    compat = import_module("src.shared.python.optimization.ocp._compat")
    return bool(compat.bioptim_available())


def _solve_bioptim(
    golfer: GolferModel,
    club: ClubModel,
    config: OptimizationConfig,
    torque_limits: dict[str, float],
    joint_limits: dict[str, tuple[float, float]],
    x0: np.ndarray,
) -> CasadiSwingResult:
    swing_ocp: Any = import_module("src.shared.python.optimization.ocp.swing_ocp")
    result: CasadiSwingResult = swing_ocp.solve_max_speed_swing(
        golfer, club, config, torque_limits, joint_limits, x0
    )
    return result


_BIOPTIM_HINT = (
    "bioptim is not installed. Install the bioptim extra: "
    "pip install 'upstream-drift[bioptim]' (Debian/Ubuntu also need "
    "python3-tk)."
)

register_backend(
    BackendSpec(
        name="scipy",
        problem_class="quick / legacy smooth NLP on the node grid",
        description="scipy.optimize.minimize with the config's method name",
        available=_always,
        install_hint="scipy is a core dependency",
    )
)
register_backend(
    BackendSpec(
        name="casadi",
        problem_class="legacy kinematic fit with a torque check",
        description="CasADi + IPOPT, finite-difference kinematics (#9756)",
        available=casadi_available,
        install_hint="pip install 'upstream-drift[optimal-control]'",
        solve=_solve_casadi_fd,
        deprecated=True,
    )
)
register_backend(
    BackendSpec(
        name="casadi-multiple-shooting",
        problem_class="single-phase torque-driven OCP without bioptim",
        description="CasADi + IPOPT, RK4 multiple shooting with torque variables",
        available=casadi_available,
        install_hint="pip install 'upstream-drift[optimal-control]'",
        solve=_solve_casadi_multiple_shooting,
    )
)
register_backend(
    BackendSpec(
        name="crocoddyl",
        problem_class="fast unconstrained-ish DDP / FDDP",
        description="Crocoddyl FDDP on the Pinocchio swing model",
        available=crocoddyl_available,
        install_hint="conda install -c conda-forge crocoddyl pinocchio",
        solve=_solve_crocoddyl,
    )
)
register_backend(
    BackendSpec(
        name="bioptim",
        problem_class="constrained, multiphase, tracking and estimation OCPs",
        description="bioptim custom-model OCP driven by the symbolic swing model",
        available=_bioptim_available,
        install_hint=_BIOPTIM_HINT,
        solve=_solve_bioptim,
    )
)
