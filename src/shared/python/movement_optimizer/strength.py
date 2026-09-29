# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Joint strength models and load-capacity helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .constants import (
    BENCH_PRESS_HILL_OPTIMAL_ANGLES,
    BENCH_PRESS_JOINT_NAMES,
    BENCH_PRESS_MAX_JOINT_TORQUES,
    DEFAULT_MAX_JOINT_TORQUES,
    HILL_ANGLE_WIDTH,
    HILL_ECCENTRIC_FACTOR,
    HILL_K_SHAPE,
    HILL_MAX_ANGULAR_VELOCITY,
    HILL_MAX_ECCENTRIC_RATIO,
    HILL_OPTIMAL_ANGLES,
    JOINT_NAMES,
)


class HillTorqueModel:
    """Hill-type torque-angle-velocity model for a joint."""

    def __init__(
        self,
        tau_max: float,
        q_optimal: float,
        angle_width: float = HILL_ANGLE_WIDTH,
        v_max: float = HILL_MAX_ANGULAR_VELOCITY,
        k_shape: float = HILL_K_SHAPE,
        ecc_factor: float = HILL_ECCENTRIC_FACTOR,
        max_ecc_ratio: float = HILL_MAX_ECCENTRIC_RATIO,
    ) -> None:
        import math

        if not math.isfinite(tau_max) or tau_max <= 0:
            raise ValueError("tau_max must be a finite positive number")
        if not math.isfinite(q_optimal):
            raise ValueError("q_optimal must be a finite number")
        if not math.isfinite(angle_width) or angle_width <= 0:
            raise ValueError("angle_width must be a finite positive number")
        if not math.isfinite(v_max) or v_max <= 0:
            raise ValueError("v_max must be a finite positive number")
        if not math.isfinite(k_shape):
            raise ValueError("k_shape must be a finite number")
        if not math.isfinite(ecc_factor):
            raise ValueError("ecc_factor must be a finite number")
        if not math.isfinite(max_ecc_ratio):
            raise ValueError("max_ecc_ratio must be a finite number")

        self.tau_max = tau_max
        self.q_optimal = q_optimal
        self.angle_width = angle_width
        self.v_max = v_max
        self.k_shape = k_shape
        self.ecc_factor = ecc_factor
        self.max_ecc_ratio = max_ecc_ratio

    def torque_angle_factor(self, q: float | NDArray) -> NDArray:
        """Gaussian torque-angle scaling factor in [0, 1]."""
        q_arr = np.asarray(q, dtype=float)
        if np.any(np.isnan(q_arr)):
            raise ValueError("q must not contain NaN values")
        return np.exp(-(((q_arr - self.q_optimal) / self.angle_width) ** 2))

    def torque_velocity_factor(self, qd: float | NDArray, torque_sign: float = -1.0) -> NDArray:
        """Hill-type force-velocity scaling factor.

        Branch selection is based on whether the muscle is shortening
        (concentric) or lengthening (eccentric).  The muscle shortens when
        the angular velocity ``qd`` has the same sign as ``torque_sign``
        (the direction in which the joint's prime movers produce torque).

        Parameters:
            qd: Joint angular velocity (rad/s).  Scalar or array.
            torque_sign: Sign convention for the joint's primary torque
                direction.  Default is -1.0 (flexor-dominant, e.g. elbow
                curl produces negative qd).  Use +1.0 for extensor-dominant
                joints.  When ``qd * torque_sign > 0`` the muscle is
                shortening (concentric); otherwise it is lengthening
                (eccentric).
        """
        qd = np.asarray(qd, dtype=float)
        if np.any(np.isnan(qd)):
            raise ValueError("qd must not contain NaN values")
        speed = np.abs(qd)
        conc = (self.v_max - speed) / (self.v_max + speed / self.k_shape)
        ecc = (1.0 + self.ecc_factor * speed) / (1.0 + speed / self.k_shape)
        # Concentric when the muscle is shortening: qd and torque_sign
        # have the same sign.  Eccentric when lengthening (braking).
        is_concentric = qd * torque_sign >= 0
        factor = np.where(is_concentric, conc, ecc)
        return np.clip(factor, 0.0, self.max_ecc_ratio)

    def available_torque(self, q: float | NDArray, qd: float | NDArray) -> NDArray:
        """Maximum torque the joint can produce at given angle and velocity."""
        return self.tau_max * self.torque_angle_factor(q) * self.torque_velocity_factor(qd)


class JointTorqueSet:
    """Collection of Hill torque models for all joints in a chain."""

    def __init__(
        self,
        joint_names: tuple[str, ...],
        max_torques: dict[str, float],
        optimal_angles: dict[str, float],
        angle_width: float = HILL_ANGLE_WIDTH,
        v_max: float = HILL_MAX_ANGULAR_VELOCITY,
    ) -> None:
        if not all(name in max_torques for name in joint_names):
            raise ValueError("max_torques must cover all joints")
        if not all(name in optimal_angles for name in joint_names):
            raise ValueError("optimal_angles must cover all joints")

        self.joint_names = joint_names
        self._models = {
            name: HillTorqueModel(
                tau_max=max_torques[name],
                q_optimal=optimal_angles[name],
                angle_width=angle_width,
                v_max=v_max,
            )
            for name in joint_names
        }

    def set_max_torque(self, joint_name: str, tau_max: float) -> None:
        """Update the maximum isometric torque for a joint."""
        if joint_name not in self._models:
            raise ValueError(f"Unknown joint: {joint_name}")
        if tau_max <= 0:
            raise ValueError("tau_max must be positive")
        self._models[joint_name].tau_max = tau_max

    def get_max_torque(self, joint_name: str) -> float:
        """Return current max isometric torque for a joint."""
        if joint_name not in self._models:
            raise ValueError(f"Unknown joint: {joint_name}")
        return self._models[joint_name].tau_max

    def available_torques(self, q: NDArray, qd: NDArray) -> NDArray:
        """Compute available torque at each joint for a single pose."""
        result = np.empty(len(self.joint_names))
        for index, name in enumerate(self.joint_names):
            result[index] = self._models[name].available_torque(q[index], qd[index])
        return result

    def available_torques_batch(self, q: NDArray, qd: NDArray) -> NDArray:
        """Compute available torque at each joint for N poses."""
        result = np.empty((q.shape[0], len(self.joint_names)))
        for index, name in enumerate(self.joint_names):
            result[:, index] = self._models[name].available_torque(q[:, index], qd[:, index])
        return result

    def torque_utilization(self, q: NDArray, qd: NDArray, required_torques: NDArray) -> NDArray:
        """Ratio of required torque to available torque."""
        available = self.available_torques_batch(q, qd)
        safe_available = np.maximum(available, 1e-10)
        return np.abs(required_torques) / safe_available

    def find_sticking_point(
        self, q: NDArray, qd: NDArray, required_torques: NDArray
    ) -> tuple[int, str, float]:
        """Find the time step and joint where torque utilization is highest."""
        utilization = self.torque_utilization(q, qd, required_torques)
        flat_index = int(np.argmax(utilization))
        time_index, joint_index = np.unravel_index(flat_index, utilization.shape)
        return (
            int(time_index),
            self.joint_names[joint_index],
            float(utilization[time_index, joint_index]),
        )


def make_default_torque_set() -> JointTorqueSet:
    """Create a JointTorqueSet with default parameters for the standing chain."""
    return JointTorqueSet(
        joint_names=JOINT_NAMES,
        max_torques=DEFAULT_MAX_JOINT_TORQUES,
        optimal_angles=HILL_OPTIMAL_ANGLES,
    )


def make_bench_press_torque_set() -> JointTorqueSet:
    """Create a JointTorqueSet with bench press parameters."""
    return JointTorqueSet(
        joint_names=BENCH_PRESS_JOINT_NAMES,
        max_torques=BENCH_PRESS_MAX_JOINT_TORQUES,
        optimal_angles=BENCH_PRESS_HILL_OPTIMAL_ANGLES,
    )


def compute_max_load(
    dynamics_factory,
    body,
    torque_set: JointTorqueSet,
    exercise_type: str,
    load_range: tuple[float, float] = (0.0, 500.0),
    tol: float = 0.5,
    n_eval: int = 40,
) -> tuple[float, int, str, float]:
    """Binary search for the maximum load a lifter can handle."""
    from .trajectory import TrajectoryOptimizer

    if load_range[0] < 0:
        raise ValueError("min load must be non-negative")
    if load_range[1] <= load_range[0]:
        raise ValueError("max must exceed min")
    if tol <= 0:
        raise ValueError("tolerance must be positive")

    lo, hi = load_range

    def is_feasible(bar_mass: float) -> tuple[bool, int, str, float]:
        config = dynamics_factory(body, bar_mass)
        if len(config) == 5:
            dynamics, q_start, q_end, q_bounds, q_via = config
        else:
            dynamics, q_start, q_end, q_bounds = config
            q_via = None

        optimizer = TrajectoryOptimizer(
            body,
            dynamics,
            exercise_type,
            bar_mass,
            q_start,
            q_end,
            q_bounds,
            q_via=q_via,
            duration=2.0,
            n_waypoints=8,
            n_eval=n_eval,
            n_starts=1,
        )
        result = optimizer.optimize()
        if not result.success:
            return False, 0, "", float("inf")

        time_index, joint_name, peak_utilization = torque_set.find_sticking_point(
            result.q,
            result.qd,
            result.torques,
        )
        return peak_utilization <= 1.0, time_index, joint_name, peak_utilization

    best_time_index, best_joint, best_utilization = 0, "", 0.0
    last_feasible_load = lo

    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        feasible, time_index, joint_name, utilization = is_feasible(mid)
        if feasible:
            lo = mid
            last_feasible_load = mid
            best_time_index, best_joint, best_utilization = (
                time_index,
                joint_name,
                utilization,
            )
        else:
            hi = mid

    return last_feasible_load, best_time_index, best_joint, best_utilization
