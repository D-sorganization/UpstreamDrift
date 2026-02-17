"""Power, work, impulse, and stiffness metrics for joint analysis.

Includes mechanical work, joint power, impulse metrics, phase space
path length, and joint stiffness calculations.
"""

from __future__ import annotations

import numpy as np

from src.shared.python.analysis.dataclasses import (
    ImpulseMetrics,
    JointPowerMetrics,
    JointStiffnessMetrics,
)
from src.shared.python.core.contracts import ensure


class PowerWorkMetricsMixin:
    """Mixin for power, work, and stiffness analysis.

    Expects the following attributes to be available on the instance:
    - times: np.ndarray
    - joint_positions: np.ndarray
    - joint_velocities: np.ndarray
    - joint_torques: np.ndarray
    - ground_forces: np.ndarray | None
    - dt: float
    - _work_metrics_cache: dict[int, dict[str, float]]
    """

    times: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    ground_forces: np.ndarray | None
    dt: float
    _work_metrics_cache: dict[int, dict[str, float]]

    def compute_work_metrics(self, joint_idx: int) -> dict[str, float] | None:
        """Compute mechanical work metrics for a joint.

        Work is calculated as the time integral of power (torque * angular velocity).
        Results are cached for performance.

        Design by Contract:
            Postconditions:
                - positive_work >= 0
                - negative_work <= 0
                - all values are finite

        Args:
            joint_idx: Index of the joint

        Returns:
            Dictionary with 'positive_work', 'negative_work', 'net_work' (Joules)
            or None if data unavailable.
        """
        if joint_idx in self._work_metrics_cache:
            return self._work_metrics_cache[joint_idx]

        if (
            joint_idx >= self.joint_torques.shape[1]
            or joint_idx >= self.joint_velocities.shape[1]
        ):
            return None

        torque = self.joint_torques[:, joint_idx]
        velocity = self.joint_velocities[:, joint_idx]

        n_samples = min(len(torque), len(velocity), len(self.times))
        if n_samples < 2:
            return None

        torque = torque[:n_samples]
        velocity = velocity[:n_samples]
        dt = self.dt

        power = torque * velocity

        pos_power = np.maximum(power, 0)
        neg_power = np.minimum(power, 0)

        if hasattr(np, "trapezoid"):
            positive_work = np.trapezoid(pos_power, dx=dt)
            negative_work = np.trapezoid(neg_power, dx=dt)
        else:
            trapz_func = getattr(np, "trapz")  # noqa: B009
            positive_work = trapz_func(pos_power, dx=dt)
            negative_work = trapz_func(neg_power, dx=dt)

        net_work = positive_work + negative_work

        result = {
            "positive_work": float(positive_work),
            "negative_work": float(negative_work),
            "net_work": float(net_work),
        }

        # Postconditions
        ensure(
            result["positive_work"] >= 0,
            "positive work must be non-negative",
            result["positive_work"],
        )
        ensure(
            result["negative_work"] <= 0,
            "negative work must be non-positive",
            result["negative_work"],
        )
        ensure(
            np.isfinite(result["net_work"]),
            "net work must be finite",
            result["net_work"],
        )

        self._work_metrics_cache[joint_idx] = result
        return result

    def compute_joint_power_metrics(self, joint_idx: int) -> JointPowerMetrics | None:
        """Compute detailed power metrics for a joint.

        Design by Contract:
            Postconditions:
                - peak_generation >= 0
                - peak_absorption <= 0
                - generation_duration >= 0 and absorption_duration >= 0
                - all values are finite

        Args:
            joint_idx: Index of the joint

        Returns:
            JointPowerMetrics object or None
        """
        if (
            joint_idx >= self.joint_torques.shape[1]
            or joint_idx >= self.joint_velocities.shape[1]
        ):
            return None

        torque = self.joint_torques[:, joint_idx]
        velocity = self.joint_velocities[:, joint_idx]

        n_samples = min(len(torque), len(velocity), len(self.times))
        if n_samples < 2:
            return None

        torque = torque[:n_samples]
        velocity = velocity[:n_samples]
        dt = self.dt

        power = torque * velocity

        gen_indices = power > 0
        peak_gen = float(np.max(power)) if np.any(gen_indices) else 0.0
        avg_gen = float(np.mean(power[gen_indices])) if np.any(gen_indices) else 0.0
        gen_dur = float(np.sum(gen_indices) * dt)

        abs_indices = power < 0
        peak_abs = float(np.min(power)) if np.any(abs_indices) else 0.0
        avg_abs = float(np.mean(power[abs_indices])) if np.any(abs_indices) else 0.0
        abs_dur = float(np.sum(abs_indices) * dt)

        if hasattr(np, "trapezoid"):
            net_work = float(np.trapezoid(power, dx=dt))
        else:
            trapz_func = getattr(np, "trapz")  # noqa: B009
            net_work = float(trapz_func(power, dx=dt))

        result = JointPowerMetrics(
            peak_generation=peak_gen,
            peak_absorption=peak_abs,
            avg_generation=avg_gen,
            avg_absorption=avg_abs,
            net_work=net_work,
            generation_duration=gen_dur,
            absorption_duration=abs_dur,
        )

        # Postconditions
        ensure(
            result.peak_generation >= 0,
            "peak generation must be non-negative",
            result.peak_generation,
        )
        ensure(
            result.peak_absorption <= 0,
            "peak absorption must be non-positive",
            result.peak_absorption,
        )
        ensure(
            result.generation_duration >= 0,
            "generation duration must be non-negative",
            result.generation_duration,
        )
        ensure(
            result.absorption_duration >= 0,
            "absorption duration must be non-negative",
            result.absorption_duration,
        )
        ensure(
            np.isfinite(result.net_work),
            "net work must be finite",
            result.net_work,
        )

        return result

    def compute_impulse_metrics(
        self,
        data_type: str = "torque",
        joint_idx: int = 0,
    ) -> ImpulseMetrics | None:
        """Compute impulse metrics (integral of force/torque over time).

        Args:
            data_type: 'torque' or 'force'
            joint_idx: Index of joint or force component

        Returns:
            ImpulseMetrics or None
        """
        if data_type == "torque":
            if joint_idx >= self.joint_torques.shape[1]:
                return None
            data = self.joint_torques[:, joint_idx]
        elif data_type == "force":
            if self.ground_forces is None or joint_idx >= self.ground_forces.shape[1]:
                return None
            data = self.ground_forces[:, joint_idx]
        else:
            return None

        n_samples = min(len(data), len(self.times))
        if n_samples < 2:
            return None

        data = data[:n_samples]
        dt = self.dt

        pos_data = np.maximum(data, 0)
        neg_data = np.minimum(data, 0)

        if hasattr(np, "trapezoid"):
            net_impulse = float(np.trapezoid(data, dx=dt))
            pos_impulse = float(np.trapezoid(pos_data, dx=dt))
            neg_impulse = float(np.trapezoid(neg_data, dx=dt))
        else:
            trapz_func = getattr(np, "trapz")  # noqa: B009
            net_impulse = float(trapz_func(data, dx=dt))
            pos_impulse = float(trapz_func(pos_data, dx=dt))
            neg_impulse = float(trapz_func(neg_data, dx=dt))

        return ImpulseMetrics(
            net_impulse=net_impulse,
            positive_impulse=pos_impulse,
            negative_impulse=neg_impulse,
        )

    def compute_phase_space_path_length(self, joint_idx: int) -> float:
        """Compute path length in phase space (Angle vs Angular Velocity).

        Design by Contract:
            Postcondition:
                - result >= 0 (path length is non-negative)

        Args:
            joint_idx: Index of the joint

        Returns:
            Total path length in phase space.
        """
        if (
            joint_idx >= self.joint_positions.shape[1]
            or joint_idx >= self.joint_velocities.shape[1]
        ):
            return 0.0

        pos = self.joint_positions[:, joint_idx]
        vel = self.joint_velocities[:, joint_idx]

        d_pos = pos[1:] - pos[:-1]
        d_vel = vel[1:] - vel[:-1]

        dist = np.sqrt(d_pos**2 + d_vel**2)
        result = float(np.sum(dist))
        ensure(result >= 0, "phase space path length must be non-negative", result)
        return result

    def compute_joint_stiffness(
        self,
        joint_idx: int,
        window: slice | None = None,
    ) -> JointStiffnessMetrics | None:
        """Compute Quasi-Stiffness from Moment-Angle relationship.

        Args:
            joint_idx: Joint index
            window: Optional slice to compute stiffness over a specific phase

        Returns:
            JointStiffnessMetrics object or None
        """
        if (
            joint_idx >= self.joint_positions.shape[1]
            or joint_idx >= self.joint_torques.shape[1]
        ):
            return None

        angles = self.joint_positions[:, joint_idx]
        torques = self.joint_torques[:, joint_idx]

        if window:
            angles = angles[window]
            torques = torques[window]

        if len(angles) < 2:
            return None

        slope, intercept = np.polyfit(angles, torques, 1)

        predicted = slope * angles + intercept
        residuals = torques - predicted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((torques - np.mean(torques)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        if hasattr(np, "trapezoid"):
            area = float(np.abs(np.trapezoid(torques, x=angles)))
        else:
            trapz_func = getattr(np, "trapz")  # noqa: B009
            area = float(np.abs(trapz_func(torques, x=angles)))

        return JointStiffnessMetrics(
            stiffness=float(slope),
            r_squared=float(r_squared),
            hysteresis_area=area,
            intercept=float(intercept),
        )

    def compute_dynamic_stiffness(
        self,
        joint_idx: int,
        window_size: int = 20,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute rolling Quasi-Stiffness.

        Args:
            joint_idx: Joint index
            window_size: Window size in samples

        Returns:
            Tuple of (times, stiffness_values, r_squared_values)
        """
        if (
            joint_idx >= self.joint_positions.shape[1]
            or joint_idx >= self.joint_torques.shape[1]
        ):
            return np.array([]), np.array([]), np.array([])

        angles = self.joint_positions[:, joint_idx]
        torques = self.joint_torques[:, joint_idx]

        if len(angles) < window_size:
            return np.array([]), np.array([]), np.array([])

        kernel = np.ones(window_size)
        n = window_size

        xy = angles * torques
        xx = angles * angles
        yy = torques * torques

        s_x = np.convolve(angles, kernel, mode="valid")
        s_y = np.convolve(torques, kernel, mode="valid")
        s_xy = np.convolve(xy, kernel, mode="valid")
        s_xx = np.convolve(xx, kernel, mode="valid")
        s_yy = np.convolve(yy, kernel, mode="valid")

        cov = s_xy - (s_x * s_y) / n
        var_x = s_xx - (s_x**2) / n
        var_y = s_yy - (s_y**2) / n

        slope = np.zeros_like(cov)
        valid_var_x = var_x > 1e-9
        np.divide(cov, var_x, out=slope, where=valid_var_x)

        r2 = np.zeros_like(cov)
        valid_both = valid_var_x & (var_y > 1e-9)
        np.divide(cov**2, var_x * var_y, out=r2, where=valid_both)

        valid_indices = np.arange(len(slope)) + window_size // 2
        time_points = self.times[valid_indices]

        return time_points, slope, r2
