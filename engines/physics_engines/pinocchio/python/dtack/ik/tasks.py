"""Task definitions for Pink IK."""

from __future__ import annotations

import typing

import numpy as np  # noqa: TID253
import pinocchio as pin

# Pink is optional but we import it for type hinting if available
try:
    import pink.tasks
    PINK_AVAILABLE = True
except ImportError:
    PINK_AVAILABLE = False
    # Mock for type hints if not available
    if typing.TYPE_CHECKING:
        from unittest.mock import MagicMock
        pink = MagicMock()
        pink.tasks = MagicMock()

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
    if not PINK_AVAILABLE:
        raise ImportError("Pink is not installed.")

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
    if not PINK_AVAILABLE:
        raise ImportError("Pink is not installed.")

    task = pink.tasks.PostureTask(cost=cost)
    if q_ref is not None:
        task.set_target(q_ref)
    return task


def create_joint_coupling_task(
    model: pin.Model,
    joint_names: list[str],
    ratios: list[float],
    cost: float = 100.0,
) -> pink.tasks.LinearHolonomicTask:
    """Create a task to enforce linear coupling between joints.

    Useful for anatomical constraints (e.g. Scapula moving with Shoulder).
    Equation: sum(ratio_i * q_i) = constant (or 0)

    Args:
        model: Pinocchio model (required to map names to indices)
        joint_names: List of joint names involved
        ratios: Coefficients for each joint
        cost: Task weight

    Returns:
        LinearHolonomicTask A * q = b
    """
    if not PINK_AVAILABLE:
        raise ImportError("Pink is not installed.")

    if len(joint_names) != len(ratios):
        raise ValueError("joint_names and ratios must have same length")

    # Construct A matrix (1 x nq)
    # Note: Pink/Pinocchio config vector size might differ from nq if using Lie algebra
    # But usually LinearHolonomicTask operates on tangent space (nv) or config space (nq)?
    # Pink documentation says: A * q = b.

    nv = model.nv
    A = np.zeros((1, nv))

    for name, ratio in zip(joint_names, ratios, strict=True):
        if not model.existJointName(name):
            raise ValueError(f"Joint '{name}' not found in model")

        joint_id = model.getJointId(name)
        # Assuming simple 1-DOF joints where joint_id maps to a single index in q/v
        # In Pinocchio, idx_v gives the index in velocity vector.
        idx_v = model.joints[joint_id].idx_v

        if idx_v >= 0:
            A[0, idx_v] = ratio

    b = np.zeros(1)

    return pink.tasks.LinearHolonomicTask(A, b, cost=cost)
