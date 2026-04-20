"""Physics model registry for extensible model management.

Allows physics models to be registered and discovered dynamically,
enabling new pendulum configurations to be added without modifying
existing code.

Design by Contract
------------------
- register_model(name, config) requires name to be a non-empty string.
- get_model(name) raises KeyError if name not registered.
- list_models() returns all registered model names.

DRY
---
Model configuration is defined once in a ModelConfig dataclass.
Registration replaces hardcoded model selection in main_window.py.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a registered physics model.

    Attributes
    ----------
    name : str
        Human-readable model name (e.g., "Double Pendulum").
    n_dof : int
        Number of degrees of freedom.
    state_size : int
        Size of the state vector (typically 2 * n_dof).
    param_class : type
        Dataclass type for model parameters.
    simulation_runner : Callable
        Function to run simulation (e.g., simulation.run_simulation).
    result_class : type
        Class for simulation results.
    description : str
        Brief description of the model.
    """

    name: str
    n_dof: int
    state_size: int
    param_class: type
    simulation_runner: Callable
    result_class: type
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_registry: dict[str, ModelConfig] = {}


def register_model(name: str, config: ModelConfig) -> None:
    """Register a physics model configuration.

    Pre: name is a non-empty string. config is a ModelConfig.
    Post: model is retrievable via get_model(name).
    """
    assert name and isinstance(
        name, str
    ), f"Model name must be non-empty string, got {name!r}"
    assert isinstance(config, ModelConfig), f"Expected ModelConfig, got {type(config)}"
    if name in _registry:
        logger.warning("Overwriting existing model registration: %s", name)
    _registry[name] = config
    logger.info("Registered physics model: %s (%d DOF)", name, config.n_dof)


def get_model(name: str) -> ModelConfig:
    """Retrieve a registered model configuration.

    Pre: name is registered.
    Post: returns a ModelConfig.
    Raises: KeyError if not found.
    """
    if name not in _registry:
        raise KeyError(
            f"Model {name!r} not registered. Available: {list(_registry.keys())}"
        )
    return _registry[name]


def list_models() -> list[str]:
    """Return all registered model names.

    Post: returned list is sorted alphabetically.
    """
    return sorted(_registry.keys())


def clear_registry() -> None:
    """Clear all registered models. Mainly for testing."""
    _registry.clear()


# ---------------------------------------------------------------------------
# Built-in model registrations
# ---------------------------------------------------------------------------


def _register_builtins() -> None:
    """Register the three built-in pendulum models."""
    try:
        from .physics import PendulumParams
        from .simulation import SimulationResult
        from .simulation import run_simulation as run_double

        register_model(
            "double",
            ModelConfig(
                name="Double Pendulum",
                n_dof=2,
                state_size=4,
                param_class=PendulumParams,
                simulation_runner=run_double,
                result_class=SimulationResult,
                description="2-DOF driven double pendulum (shoulder + wrist)",
            ),
        )
    except ImportError:
        logger.debug("Could not register double pendulum model")

    try:
        from .physics_triple import TriplePendulumParams
        from .simulation_triple import (
            TripleSimulationResult,
        )
        from .simulation_triple import run_simulation as run_triple

        register_model(
            "triple",
            ModelConfig(
                name="Triple Pendulum",
                n_dof=3,
                state_size=6,
                param_class=TriplePendulumParams,
                simulation_runner=run_triple,
                result_class=TripleSimulationResult,
                description="3-DOF driven triple pendulum (hub + arm + club)",
            ),
        )
    except ImportError:
        logger.debug("Could not register triple pendulum model")

    try:
        from .physics_golfer import N_DOF, GolferParams
        from .simulation_golfer import (
            GolferSimulationResult,
        )
        from .simulation_golfer import run_simulation as run_golfer

        register_model(
            "golfer",
            ModelConfig(
                name="Golfer Upper Body",
                n_dof=N_DOF,
                state_size=2 * N_DOF,
                param_class=GolferParams,
                simulation_runner=run_golfer,
                result_class=GolferSimulationResult,
                description="8-DOF golfer upper-body with closed kinematic loop",
            ),
        )
    except ImportError:
        logger.debug("Could not register golfer model")


# Auto-register on import
_register_builtins()
