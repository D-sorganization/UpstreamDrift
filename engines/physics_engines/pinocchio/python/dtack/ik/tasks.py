"""Task definitions for Pink IK."""

from __future__ import annotations

import typing

import numpy as np  # noqa: TID253
import pink.tasks
import pinocchio as pin

# Type alias for transformation matrices
Transform = pin.SE3 | np.ndarray


def create_frame_task(
    body_name: str,
    position_cost: float = 1.0,
    orientation_cost: float = 1.0,
    lm_damping: float = 0.0,
) -> pink.tasks.FrameTask:
    """Create a FrameTask (SE3 target) for a specific body frame.

    Args:
        body_name: Name of the frame in the model
        position_cost: Weight for position error
        orientation_cost: Weight for orientation error
        lm_damping: Levenberg-Marquardt damping

    Returns:
        Configured FrameTask
    """
    task = pink.tasks.FrameTask(
        body_name, position_cost=position_cost, orientation_cost=orientation_cost
    )
    task.lm_damping = lm_damping
    return task


def create_posture_task(
    cost: float = 1e-3, q_ref: np.ndarray[typing.Any, typing.Any] | None = None
) -> pink.tasks.PostureTask:
    """Create a PostureTask to regularize joint configuration.

    Args:
        cost: Weight for the posture regularization
        q_ref: Reference configuration (target). If None, must be set later.

    Returns:
        Configured PostureTask
    """
    task = pink.tasks.PostureTask(cost=cost)
    if q_ref is not None:
        task.set_target(q_ref)
    return task


def create_joint_coupling_task(
    _joint_names: list[str],
    _ratios: list[float],
    _cost: float = 100.0,
) -> pink.tasks.LinearHolonomicTask:
    """Create a task to enforce linear coupling between joints.

    Useful for anatomical constraints (e.g. Scapula moving with Shoulder).
    Equation: sum(ratio_i * q_i) = constant (or 0)

    Args:
        joint_names: List of joint names involved
        ratios: Coefficients for each joint
        cost: Task weight

    Returns:
        LinearHolonomicTask (Note: Requires building matrix A and vector b)
    """
    # This is a placeholder. Pink's LinearHolonomicTask takes A, b in A * q = b
    # Implementation depends on how we want to construct 'A' from names.
    # Typically A is shape (k, nq).
    # For now, we return usage instructions or a base implementation if feasible.
    msg = "Joint coupling task requires mapping names to joint indices dynamically."
    raise NotImplementedError(msg)
