# mypy: ignore-errors
"""
GPU-accelerated torque profile optimization using JAX autodiff.

This module provides gradient-based optimization of torque profiles using
JAX's automatic differentiation and optax optimizers.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import jax
    import jax.numpy as jnp
    import optax
except ImportError:
    raise ImportError(
        "JAX and optax are required for GPU optimization. "
        "Install with: pip install jax jaxlib optax"
    ) from None

from .physics_golfer_jax import GolferParamsJAX, analytical_fk_jacobians_jax
from .simulation_golfer_gpu import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    extract_final_state,
    run_single_simulation_jax,
)

logger = logging.getLogger(__name__)

# Type aliases
JaxArray = jnp.ndarray


def clubhead_speed_objective(
    torque_coeffs: JaxArray,
    params: GolferParamsJAX,
    initial_state: JaxArray,
    t_end: float,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    dt: float = 0.005,
) -> JaxArray:
    """Objective function: maximize clubhead speed at impact.

    Parameters
    ----------
    torque_coeffs : JaxArray, shape (7,)
        Torque coefficients (one per actuated joint)
    params : GolferParamsJAX
        Physical parameters
    initial_state : JaxArray, shape (16,)
        Initial state
    t_end : float
        Simulation end time
    alpha, beta : float
        Baumgarte stabilization gains
    dt : float
        Target timestep

    Returns
    -------
    neg_speed : JaxArray, shape ()
        Negative clubhead speed (for minimization)
    """
    assert torque_coeffs.shape == (7,), (
        f"Expected (7,) coeffs, got {torque_coeffs.shape}"
    )
    assert t_end > 0, f"t_end must be positive, got {t_end}"
    assert dt > 0, f"dt must be positive, got {dt}"
    assert initial_state.shape == (16,), (
        f"Expected (16,) state, got {initial_state.shape}"
    )
    sol = run_single_simulation_jax(
        params, initial_state, t_end, torque_coeffs, alpha, beta, dt
    )
    final_state = extract_final_state(sol)

    q = final_state[:8]
    qdot = final_state[8:]

    # Compute clubhead velocity via FK Jacobian
    jacobians = analytical_fk_jacobians_jax(q, params)
    J_tip = jacobians["club_tip"]
    v_tip = J_tip @ qdot

    speed = jnp.sqrt(v_tip[0] ** 2 + v_tip[1] ** 2)

    return -speed  # minimize negative speed = maximize speed


def clubhead_velocity_at_final_time(
    torque_coeffs: JaxArray,
    params: GolferParamsJAX,
    initial_state: JaxArray,
    t_end: float,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    dt: float = 0.005,
) -> JaxArray:
    """Compute clubhead velocity magnitude at final time.

    Parameters
    ----------
    torque_coeffs : JaxArray, shape (7,)
        Torque coefficients
    params : GolferParamsJAX
    initial_state : JaxArray, shape (16,)
    t_end : float
    alpha, beta : float
    dt : float

    Returns
    -------
    speed : JaxArray, shape ()
        Clubhead speed magnitude (positive)
    """
    assert torque_coeffs.shape == (7,), (
        f"Expected (7,) coeffs, got {torque_coeffs.shape}"
    )
    assert t_end > 0, f"t_end must be positive, got {t_end}"
    assert dt > 0, f"dt must be positive, got {dt}"
    assert initial_state.shape == (16,), (
        f"Expected (16,) state, got {initial_state.shape}"
    )
    sol = run_single_simulation_jax(
        params, initial_state, t_end, torque_coeffs, alpha, beta, dt
    )
    final_state = extract_final_state(sol)

    q = final_state[:8]
    qdot = final_state[8:]

    jacobians = analytical_fk_jacobians_jax(q, params)
    J_tip = jacobians["club_tip"]
    v_tip = J_tip @ qdot

    speed = jnp.sqrt(v_tip[0] ** 2 + v_tip[1] ** 2)

    return speed


def optimize_torque_profile(
    params: GolferParamsJAX,
    initial_state: JaxArray,
    t_end: float,
    n_coeffs_per_joint: int = 3,
    n_iterations: int = 100,
    learning_rate: float = 0.01,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    dt: float = 0.005,
    seed: int = 42,
) -> tuple[JaxArray, list[float]]:
    """Optimize torque polynomial coefficients using Adam optimizer.

    Parameters
    ----------
    params : GolferParamsJAX
        Physical parameters
    initial_state : JaxArray, shape (16,)
        Initial state
    t_end : float
        Simulation end time
    n_coeffs_per_joint : int
        Number of polynomial coefficients per joint
    n_iterations : int
        Number of optimization iterations
    learning_rate : float
        Adam optimizer learning rate
    alpha, beta : float
        Baumgarte stabilization gains
    dt : float
        Target timestep
    seed : int
        Random seed for initialization

    Returns
    -------
    optimal_coeffs : JaxArray, shape (7, n_coeffs_per_joint)
        Optimized torque coefficients
    history : list[float]
        Loss values at each iteration (negative speeds, so larger = better)
    """
    assert n_coeffs_per_joint >= 1, f"n_coeffs must be >= 1, got {n_coeffs_per_joint}"
    assert n_iterations >= 1, f"n_iterations must be >= 1, got {n_iterations}"
    assert learning_rate > 0, f"learning_rate must be positive, got {learning_rate}"
    assert t_end > 0, f"t_end must be positive, got {t_end}"
    # Initialize: 7 joints Ã— n_coeffs_per_joint
    key = jax.random.PRNGKey(seed)
    torque_coeffs = jax.random.normal(key, (7, n_coeffs_per_joint)) * 0.1
    torque_coeffs = torque_coeffs.reshape(-1)  # Flatten to 1D for optimization

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(torque_coeffs)

    @jax.jit
    @jax.value_and_grad
    def loss_fn(coeffs: Any) -> float:
        coeffs_reshaped = coeffs.reshape(7, n_coeffs_per_joint)
        # Use mean torque across coefficients as the single torque profile
        torque_simple = jnp.mean(coeffs_reshaped, axis=1)
        return clubhead_speed_objective(
            torque_simple, params, initial_state, t_end, alpha, beta, dt
        )

    history = []

    for i in range(n_iterations):
        loss, grads = loss_fn(torque_coeffs)
        updates, opt_state = optimizer.update(grads, opt_state)
        torque_coeffs = optax.apply_updates(torque_coeffs, updates)

        loss_val = float(loss)
        history.append(loss_val)

        if (i + 1) % max(1, n_iterations // 10) == 0:
            logger.info("Iteration %d/%d: loss = %.6f", i + 1, n_iterations, loss_val)

    optimal_coeffs = torque_coeffs.reshape(7, n_coeffs_per_joint)
    assert len(history) == n_iterations, (
        f"Expected {n_iterations} history entries, got {len(history)}"
    )
    assert optimal_coeffs.shape == (7, n_coeffs_per_joint)

    return optimal_coeffs, history


def optimize_simple_torque_profile(
    params: GolferParamsJAX,
    initial_state: JaxArray,
    t_end: float,
    n_iterations: int = 100,
    learning_rate: float = 0.01,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    dt: float = 0.005,
    seed: int = 42,
) -> tuple[JaxArray, list[float]]:
    """Optimize a simple torque profile (one coefficient per joint) using Adam.

    Parameters
    ----------
    params : GolferParamsJAX
        Physical parameters
    initial_state : JaxArray, shape (16,)
        Initial state
    t_end : float
        Simulation end time
    n_iterations : int
        Number of optimization iterations
    learning_rate : float
        Adam optimizer learning rate
    alpha, beta : float
        Baumgarte stabilization gains
    dt : float
        Target timestep
    seed : int
        Random seed

    Returns
    -------
    optimal_torques : JaxArray, shape (7,)
        Optimized constant torques for each joint
    history : list[float]
        Loss values at each iteration
    """
    assert n_iterations >= 1, f"n_iterations must be >= 1, got {n_iterations}"
    assert learning_rate > 0, f"learning_rate must be positive, got {learning_rate}"
    assert t_end > 0, f"t_end must be positive, got {t_end}"
    # Initialize: one torque per joint
    key = jax.random.PRNGKey(seed)
    torque_coeffs = jax.random.normal(key, (7,)) * 0.1

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(torque_coeffs)

    @jax.jit
    @jax.value_and_grad
    def loss_fn(coeffs: Any) -> float:
        return clubhead_speed_objective(
            coeffs, params, initial_state, t_end, alpha, beta, dt
        )

    history = []

    for i in range(n_iterations):
        loss, grads = loss_fn(torque_coeffs)
        updates, opt_state = optimizer.update(grads, opt_state)
        torque_coeffs = optax.apply_updates(torque_coeffs, updates)

        loss_val = float(loss)
        history.append(loss_val)

        if (i + 1) % max(1, n_iterations // 10) == 0:
            logger.info("Iteration %d/%d: loss = %.6f", i + 1, n_iterations, loss_val)

    assert len(history) == n_iterations
    return torque_coeffs, history


def compute_gradient_via_finite_difference(
    torque_coeffs: JaxArray,
    params: GolferParamsJAX,
    initial_state: JaxArray,
    t_end: float,
    eps: float = 1e-5,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    dt: float = 0.005,
) -> JaxArray:
    """Compute objective gradient via finite differences (for testing).

    Parameters
    ----------
    torque_coeffs : JaxArray, shape (7,)
    params : GolferParamsJAX
    initial_state : JaxArray, shape (16,)
    t_end : float
    eps : float
        Finite difference step size
    alpha, beta : float
    dt : float

    Returns
    -------
    grad : JaxArray, shape (7,)
    """
    assert torque_coeffs.shape == (7,), (
        f"Expected (7,) coeffs, got {torque_coeffs.shape}"
    )
    assert eps > 0, f"eps must be positive, got {eps}"
    grad = jnp.zeros(7)

    f0 = clubhead_speed_objective(
        torque_coeffs, params, initial_state, t_end, alpha, beta, dt
    )

    for i in range(7):
        torque_plus = torque_coeffs.at[i].add(eps)
        f_plus = clubhead_speed_objective(
            torque_plus, params, initial_state, t_end, alpha, beta, dt
        )
        grad = grad.at[i].set((f_plus - f0) / eps)

    assert grad.shape == (7,)
    return grad
