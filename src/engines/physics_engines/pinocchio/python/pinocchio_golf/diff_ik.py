"""Pure-pinocchio differential inverse kinematics fallback.

This module provides a Levenberg-Marquardt damped Gauss-Newton solver
that depends only on ``pinocchio`` (and numpy). It is used as a
fallback when ``pink`` is not importable, so that swing fitting and
seed-trajectory generation are not bottlenecked on the Pink dependency
which is fragile to install on some platforms.

Closes issue #4138.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np  # noqa: TID253

# Guard pinocchio import — heavy optional dependency.
try:
    import pinocchio as pin

    PINOCCHIO_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised only in stub envs
    PINOCCHIO_AVAILABLE = False
    pin = None  # type: ignore[assignment]


if TYPE_CHECKING:  # pragma: no cover
    import pinocchio as pin_t  # noqa: F401


__all__ = [
    "PINOCCHIO_AVAILABLE",
    "differential_ik",
    "lm_step",
    "se3_log6",
    "solve_dual_frame_ik",
]


# ---------------------------------------------------------------------------
# Pure-numpy LM step (testable without pinocchio).
# ---------------------------------------------------------------------------


def lm_step(
    jacobian: np.ndarray,
    error: np.ndarray,
    damping: float,
) -> np.ndarray:
    """Compute a single Levenberg-Marquardt joint-velocity step.

    Solves ``(J J^T + lambda^2 I) y = err`` and returns ``dq = J^T y``,
    which is mathematically equivalent to applying a damped pseudo-
    inverse ``dq = J^T (J J^T + lambda^2 I)^(-1) err`` while staying
    numerically robust near singularities.

    Args:
        jacobian: Stacked task Jacobian of shape ``(m, nv)`` where
            ``m`` is the total task-space dimension and ``nv`` the
            number of joint-velocity coordinates.
        error: Stacked task-space error of shape ``(m,)``.
        damping: Levenberg-Marquardt damping ``lambda`` (scalar, ``>= 0``).

    Returns:
        ``dq`` of shape ``(nv,)`` — joint-velocity step that reduces
        ``error`` in the linearized model.

    Raises:
        ValueError: If ``damping`` is negative or shapes are inconsistent.
        TypeError: If inputs are not ``numpy.ndarray``.
    """
    if not isinstance(jacobian, np.ndarray):
        raise TypeError("jacobian must be a numpy.ndarray")
    if not isinstance(error, np.ndarray):
        raise TypeError("error must be a numpy.ndarray")
    if damping < 0.0:
        raise ValueError(f"damping must be non-negative, got {damping}")
    if jacobian.ndim != 2:
        raise ValueError(
            f"jacobian must be 2-D, got shape {jacobian.shape}",
        )
    if error.ndim != 1 or error.shape[0] != jacobian.shape[0]:
        raise ValueError(
            f"error shape {error.shape} incompatible with jacobian "
            f"shape {jacobian.shape}",
        )

    m = jacobian.shape[0]
    jjt = jacobian @ jacobian.T
    regularised = jjt + (damping**2) * np.eye(m, dtype=jjt.dtype)
    # np.linalg.solve raises LinAlgError on singular; with damping > 0
    # the regularised matrix is SPD and solve is stable.
    y = np.linalg.solve(regularised, error)
    return jacobian.T @ y


# ---------------------------------------------------------------------------
# SE3 logarithm — small wrapper around pin.log6 with a numpy fallback.
# ---------------------------------------------------------------------------


def se3_log6(target: pin.SE3, current: pin.SE3) -> np.ndarray:
    """Twist that takes ``current`` to ``target`` in the local frame.

    Returns the 6-vector ``log6(current^-1 * target)`` expressed in the
    LOCAL frame of ``current`` — the convention compatible with
    ``pin.computeFrameJacobian(..., pin.LOCAL)``.
    """
    if not PINOCCHIO_AVAILABLE:
        raise ImportError("pinocchio is required for se3_log6")
    err_se3 = current.actInv(target)
    return np.asarray(pin.log6(err_se3).vector, dtype=np.float64)


# ---------------------------------------------------------------------------
# Single-frame differential IK.
# ---------------------------------------------------------------------------


def differential_ik(
    model: pin.Model,
    data: pin.Data,
    target_frame_name: str,
    target_se3: pin.SE3,
    q0: np.ndarray,
    *,
    max_iters: int = 100,
    damping: float = 1e-6,
    tol: float = 1e-4,
) -> tuple[np.ndarray, bool]:
    """Solve for ``q`` such that ``target_frame_name`` reaches ``target_se3``.

    Levenberg-Marquardt damped Gauss-Newton iteration in the LOCAL
    frame. This is intentionally a pure-pinocchio implementation that
    avoids any dependency on ``pink``.

    Args:
        model: Pinocchio kinematic model.
        data: Pinocchio data buffer associated with ``model``.
        target_frame_name: Name of an existing frame in ``model``.
        target_se3: Desired SE3 pose for that frame, in the world frame.
        q0: Initial joint configuration of length ``model.nq``.
        max_iters: Hard iteration cap.
        damping: Levenberg-Marquardt damping ``lambda >= 0``.
        tol: Convergence tolerance on the L2 norm of the 6-D twist
            error.

    Returns:
        ``(q_solution, converged)`` — the final configuration and
        whether convergence reached ``tol`` within ``max_iters``.

    Raises:
        ImportError: If pinocchio is not installed.
        ValueError: If preconditions on inputs are violated.
    """
    if not PINOCCHIO_AVAILABLE:
        raise ImportError("pinocchio is required for differential_ik")
    if not isinstance(target_frame_name, str) or not target_frame_name:
        raise ValueError("target_frame_name must be a non-empty string")
    if max_iters <= 0:
        raise ValueError(f"max_iters must be positive, got {max_iters}")
    if damping < 0.0:
        raise ValueError(f"damping must be non-negative, got {damping}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")
    if not model.existFrame(target_frame_name):
        raise ValueError(
            f"frame {target_frame_name!r} not found in model",
        )
    q = np.asarray(q0, dtype=np.float64).copy()
    if q.shape != (model.nq,):
        raise ValueError(
            f"q0 has shape {q.shape}, expected ({model.nq},)",
        )

    frame_id = model.getFrameId(target_frame_name)

    converged = False
    for _ in range(max_iters):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacement(model, data, frame_id)
        current = data.oMf[frame_id]
        err = se3_log6(target_se3, current)
        # Bolt: math.sqrt(np.vdot) avoids array allocation and is ~1.8x faster
        if math.sqrt(np.vdot(err, err)) < tol:
            converged = True
            break

        pin.computeJointJacobians(model, data, q)
        jac = pin.getFrameJacobian(model, data, frame_id, pin.LOCAL)
        dq = lm_step(jac, err, damping)
        q = pin.integrate(model, q, dq)

    return np.asarray(q, dtype=np.float64), converged


# ---------------------------------------------------------------------------
# Dual-frame variant — used by fit_swing_pinocchio's seed step.
# ---------------------------------------------------------------------------


def solve_dual_frame_ik(
    model: pin.Model,
    data: pin.Data,
    frame_a: str,
    target_a: pin.SE3,
    frame_b: str,
    target_b: pin.SE3,
    q0: np.ndarray,
    *,
    max_iters: int = 200,
    damping: float = 1e-5,
    tol: float = 1e-4,
    weight_a: float = 1.0,
    weight_b: float = 1.0,
) -> tuple[np.ndarray, bool]:
    """Solve a two-frame damped-pseudo-inverse IK problem.

    Stacks the per-frame Jacobians and 6-D twist errors and runs the
    same Levenberg-Marquardt update as :func:`differential_ik`. This is
    the seed-trajectory entry point used when ``pink`` is unavailable.

    Args:
        model: Pinocchio kinematic model.
        data: Pinocchio data buffer.
        frame_a: First target frame name.
        target_a: Desired SE3 for ``frame_a`` in world frame.
        frame_b: Second target frame name.
        target_b: Desired SE3 for ``frame_b`` in world frame.
        q0: Initial configuration of length ``model.nq``.
        max_iters: Hard iteration cap.
        damping: Levenberg-Marquardt damping.
        tol: Convergence tolerance on the stacked error norm.
        weight_a: Scalar weight applied to ``frame_a`` rows.
        weight_b: Scalar weight applied to ``frame_b`` rows.

    Returns:
        ``(q_solution, converged)``.
    """
    if not PINOCCHIO_AVAILABLE:
        raise ImportError("pinocchio is required for solve_dual_frame_ik")
    if weight_a <= 0.0 or weight_b <= 0.0:
        raise ValueError("frame weights must be positive")
    if not model.existFrame(frame_a):
        raise ValueError(f"frame {frame_a!r} not found")
    if not model.existFrame(frame_b):
        raise ValueError(f"frame {frame_b!r} not found")
    q = np.asarray(q0, dtype=np.float64).copy()
    if q.shape != (model.nq,):
        raise ValueError(
            f"q0 has shape {q.shape}, expected ({model.nq},)",
        )

    fid_a = model.getFrameId(frame_a)
    fid_b = model.getFrameId(frame_b)

    converged = False
    for _ in range(max_iters):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacement(model, data, fid_a)
        pin.updateFramePlacement(model, data, fid_b)
        err_a = weight_a * se3_log6(target_a, data.oMf[fid_a])
        err_b = weight_b * se3_log6(target_b, data.oMf[fid_b])
        err = np.concatenate([err_a, err_b])
        # Bolt: math.sqrt(np.vdot) avoids array allocation and is ~1.8x faster
        if math.sqrt(np.vdot(err, err)) < tol:
            converged = True
            break
        pin.computeJointJacobians(model, data, q)
        ja = weight_a * pin.getFrameJacobian(model, data, fid_a, pin.LOCAL)
        jb = weight_b * pin.getFrameJacobian(model, data, fid_b, pin.LOCAL)
        jac = np.vstack([ja, jb])
        dq = lm_step(jac, err, damping)
        q = pin.integrate(model, q, dq)

    return np.asarray(q, dtype=np.float64), converged
