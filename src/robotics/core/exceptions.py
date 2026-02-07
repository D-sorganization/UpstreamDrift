"""Robotics-specific exceptions.

Design by Contract:
    All exceptions inherit from RoboticsError for unified handling.
    Each exception type represents a specific failure category.
"""

from __future__ import annotations

from typing import Any


class RoboticsError(Exception):
    """Base exception for all robotics-related errors.

    Attributes:
        message: Human-readable error description.
        details: Optional dictionary with additional context.
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize robotics error.

        Args:
            message: Error description.
            details: Additional context for debugging.
        """
        self.message = message
        self.details = details or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with details."""
        msg = self.message
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            msg = f"{msg} [{detail_str}]"
        return msg


class ContactError(RoboticsError):
    """Error related to contact detection or processing.

    Raised when:
        - Contact detection fails
        - Invalid contact configuration
        - Contact constraint violation
    """

    def __init__(
        self,
        message: str,
        contact_id: int | None = None,
        body_names: tuple[str, str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize contact error.

        Args:
            message: Error description.
            contact_id: ID of problematic contact.
            body_names: Tuple of (body_a, body_b) names.
            details: Additional context.
        """
        details = details or {}
        if contact_id is not None:
            details["contact_id"] = contact_id
        if body_names is not None:
            details["body_a"] = body_names[0]
            details["body_b"] = body_names[1]
        super().__init__(message, details)
        self.contact_id = contact_id
        self.body_names = body_names


class ControlError(RoboticsError):
    """Error related to control computation.

    Raised when:
        - Control law computation fails
        - Joint limits violated
        - Actuator saturation
    """

    def __init__(
        self,
        message: str,
        joint_indices: list[int] | None = None,
        control_values: list[float] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize control error.

        Args:
            message: Error description.
            joint_indices: Indices of problematic joints.
            control_values: Control values that caused error.
            details: Additional context.
        """
        details = details or {}
        if joint_indices is not None:
            details["joint_indices"] = joint_indices
        if control_values is not None:
            details["control_values"] = control_values
        super().__init__(message, details)
        self.joint_indices = joint_indices
        self.control_values = control_values


class SolverError(RoboticsError):
    """Error from optimization solver.

    Raised when:
        - QP solver fails to find solution
        - Infeasible constraints
        - Numerical issues
    """

    def __init__(
        self,
        message: str,
        solver_name: str | None = None,
        status_code: int | None = None,
        iterations: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize solver error.

        Args:
            message: Error description.
            solver_name: Name of the solver that failed.
            status_code: Solver-specific status code.
            iterations: Number of iterations before failure.
            details: Additional context.
        """
        details = details or {}
        if solver_name is not None:
            details["solver"] = solver_name
        if status_code is not None:
            details["status_code"] = status_code
        if iterations is not None:
            details["iterations"] = iterations
        super().__init__(message, details)
        self.solver_name = solver_name
        self.status_code = status_code
        self.iterations = iterations


class LocomotionError(RoboticsError):
    """Error related to locomotion and balance.

    Raised when:
        - Balance lost (CoM outside support polygon)
        - Invalid gait parameters
        - Footstep planning failure
    """

    def __init__(
        self,
        message: str,
        gait_phase: str | None = None,
        support_state: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize locomotion error.

        Args:
            message: Error description.
            gait_phase: Current gait phase when error occurred.
            support_state: Support state (single/double) when error occurred.
            details: Additional context.
        """
        details = details or {}
        if gait_phase is not None:
            details["gait_phase"] = gait_phase
        if support_state is not None:
            details["support_state"] = support_state
        super().__init__(message, details)
        self.gait_phase = gait_phase
        self.support_state = support_state


class KinematicsError(RoboticsError):
    """Error related to kinematics computation.

    Raised when:
        - IK fails to converge
        - Singularity detected
        - Joint limits reached
    """

    def __init__(
        self,
        message: str,
        body_name: str | None = None,
        configuration: list[float] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize kinematics error.

        Args:
            message: Error description.
            body_name: Name of body in kinematics chain.
            configuration: Configuration where error occurred.
            details: Additional context.
        """
        details = details or {}
        if body_name is not None:
            details["body"] = body_name
        if configuration is not None:
            details["config_size"] = len(configuration)
        super().__init__(message, details)
        self.body_name = body_name
        self.configuration = configuration
