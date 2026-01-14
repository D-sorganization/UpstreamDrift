"""Statistical analysis module for golf swing biomechanics.

Provides comprehensive statistical analysis including:
- Peak detection
- Summary statistics
- Swing quality metrics
- Phase-specific analysis
- Advanced stability and coordination metrics
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform


@dataclass
class PeakInfo:
    """Information about a detected peak."""

    value: float
    time: float
    index: int
    prominence: float | None = None
    width: float | None = None


@dataclass
class SummaryStatistics:
    """Summary statistics for a time series."""

    mean: float
    median: float
    std: float
    min: float
    max: float
    range: float
    min_time: float
    max_time: float
    rms: float  # Root mean square


@dataclass
class SwingPhase:
    """Information about a swing phase."""

    name: str
    start_time: float
    end_time: float
    start_index: int
    end_index: int
    duration: float


@dataclass
class KinematicSequenceInfo:
    """Information about the kinematic sequence."""

    segment_name: str
    peak_velocity: float
    peak_time: float
    peak_index: int
    order_index: int


@dataclass
class GRFMetrics:
    """Ground Reaction Force and Center of Pressure metrics."""

    cop_path_length: float
    cop_max_velocity: float
    cop_x_range: float
    cop_y_range: float
    peak_vertical_force: float | None = None
    peak_shear_force: float | None = None


@dataclass
class AngularMomentumMetrics:
    """Metrics related to system angular momentum."""

    peak_magnitude: float
    peak_time: float
    mean_magnitude: float
    # Component peaks
    peak_lx: float
    peak_ly: float
    peak_lz: float
    # Conservation error (std dev / mean) if no external torques were present
    # (Not strictly applicable to golf as it's an open system with gravity/GRF,
    # but variability is useful)
    variability: float


@dataclass
class StabilityMetrics:
    """Metrics related to postural stability."""

    # Dynamic Stability Margin proxies
    min_com_cop_distance: float  # Minimum horizontal distance between CoM and CoP
    max_com_cop_distance: float
    mean_com_cop_distance: float

    # Inclination Angles (Angle between vertical and CoP-CoM vector)
    peak_inclination_angle: float  # Maximum lean
    mean_inclination_angle: float


@dataclass
class CoordinationMetrics:
    """Metrics quantifying inter-segment coordination patterns."""

    # Percentage of swing duration in each coordination state
    in_phase_pct: float  # Both segments rotating same direction
    anti_phase_pct: float  # Segments rotating opposite directions
    proximal_leading_pct: float  # Proximal segment dominant
    distal_leading_pct: float  # Distal segment dominant

    # Mean coupling angle (if meaningful)
    mean_coupling_angle: float
    coordination_variability: float  # Std dev of coupling angle


@dataclass
class JointPowerMetrics:
    """Metrics related to joint power generation and absorption."""

    peak_generation: float  # Max positive power (W)
    peak_absorption: float  # Max negative power (W)
    avg_generation: float  # Mean positive power (W)
    avg_absorption: float  # Mean negative power (W)
    net_work: float  # Total work done (J)
    generation_duration: float  # Time spent generating power (s)
    absorption_duration: float  # Time spent absorbing power (s)


@dataclass
class ImpulseMetrics:
    """Metrics related to force/torque impulse."""

    net_impulse: float  # Integrated force/torque over time
    positive_impulse: float  # Integrated positive force/torque
    negative_impulse: float  # Integrated negative force/torque


@dataclass
class RQAMetrics:
    """Recurrence Quantification Analysis metrics.

    Quantifies the structure of the recurrence plot.
    """

    recurrence_rate: float  # Density of recurrence points (RR)
    determinism: float  # Percentage of recurrence points forming diagonal lines (DET)
    laminarity: float  # Percentage of recurrence points forming vertical lines (LAM)
    longest_diagonal_line: int  # L_max
    trapping_time: float  # Average length of vertical lines (TT)


@dataclass
class SwingProfileMetrics:
    """Normalized metrics (0-100) for Swing Profile visualization."""

    speed_score: float
    sequence_score: float
    stability_score: float
    efficiency_score: float
    power_score: float


@dataclass
class PCAResult:
    """Result of Principal Component Analysis."""

    components: np.ndarray  # (n_components, n_features) - Eigenvectors
    explained_variance: np.ndarray  # (n_components,) - Variance of each PC
    explained_variance_ratio: np.ndarray  # (n_components,) - Ratio of variance
    projected_data: np.ndarray  # (n_samples, n_components) - Scores
    mean: np.ndarray  # (n_features,) - Mean of original data


@dataclass
class JointStiffnessMetrics:
    """Metrics related to joint stiffness (Moment-Angle relationship)."""

    stiffness: float  # Slope of the regression line (Nm/rad)
    r_squared: float  # Goodness of fit
    hysteresis_area: float  # Area inside the loop (Energy dissipated/generated)
    intercept: float  # Y-intercept of the regression line


@dataclass
class JerkMetrics:
    """Metrics related to movement smoothness (Jerk)."""

    peak_jerk: float  # Max absolute jerk
    rms_jerk: float  # Root mean square jerk
    dimensionless_jerk: float  # Normalized for duration and amplitude


class StatisticalAnalyzer:
    """Comprehensive statistical analysis for golf swing data."""

    def __init__(
        self,
        times: np.ndarray,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_torques: np.ndarray,
        club_head_speed: np.ndarray | None = None,
        club_head_position: np.ndarray | None = None,
        cop_position: np.ndarray | None = None,
        ground_forces: np.ndarray | None = None,
        com_position: np.ndarray | None = None,
        angular_momentum: np.ndarray | None = None,
        joint_accelerations: np.ndarray | None = None,
    ) -> None:
        """Initialize analyzer with recorded data.

        Args:
            times: Time array (N,)
            joint_positions: Joint positions (N, nq)
            joint_velocities: Joint velocities (N, nv)
            joint_torques: Joint torques (N, nu)
            club_head_speed: Club head speed (N,) [optional]
            club_head_position: Club head 3D position (N, 3) [optional]
            cop_position: Center of Pressure position (N, 2) or (N, 3) [optional]
            ground_forces: Ground reaction forces (N, 3) or (N, 6) [optional]
            com_position: Center of Mass position (N, 3) [optional]
            angular_momentum: System angular momentum (N, 3) [optional]
            joint_accelerations: Joint accelerations (N, nv) [optional]
        """
        self.times = times
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.joint_torques = joint_torques
        self.club_head_speed = club_head_speed
        self.club_head_position = club_head_position
        self.cop_position = cop_position
        self.ground_forces = ground_forces
        self.com_position = com_position
        self.angular_momentum = angular_momentum
        self.joint_accelerations = joint_accelerations

        self.dt = float(np.mean(np.diff(times))) if len(times) > 1 else 0.0
        self.duration = times[-1] - times[0] if len(times) > 1 else 0.0

        # Performance optimization: Cache for expensive computations
        self._work_metrics_cache: dict[int, dict[str, float]] = {}
        self._power_metrics_cache: dict[int, JointPowerMetrics] = {}

    def compute_summary_stats(self, data: np.ndarray) -> SummaryStatistics:
        """Compute summary statistics for a 1D array.

        Args:
            data: 1D numpy array

        Returns:
            SummaryStatistics object
        """
        min_idx = np.argmin(data)
        max_idx = np.argmax(data)
        min_val = float(data[min_idx])
        max_val = float(data[max_idx])

        return SummaryStatistics(
            mean=float(np.mean(data)),
            median=float(np.median(data)),
            std=float(np.std(data)),
            min=min_val,
            max=max_val,
            range=max_val - min_val,
            min_time=float(self.times[min_idx]),
            max_time=float(self.times[max_idx]),
            rms=float(np.sqrt(np.mean(data**2))),
        )

    def find_peaks_in_data(
        self,
        data: np.ndarray,
        height: float | None = None,
        prominence: float | None = None,
        distance: int | None = None,
    ) -> list[PeakInfo]:
        """Find peaks in time series data.

        Args:
            data: 1D array
            height: Minimum peak height
            prominence: Minimum peak prominence
            distance: Minimum samples between peaks

        Returns:
            List of PeakInfo objects
        """
        peaks, properties = find_peaks(
            data,
            height=height,
            prominence=prominence,
            distance=distance,
        )

        peak_list = []
        for i, peak_idx in enumerate(peaks):
            peak_info = PeakInfo(
                value=float(data[peak_idx]),
                time=float(self.times[peak_idx]),
                index=int(peak_idx),
                prominence=(
                    float(properties["prominences"][i])
                    if "prominences" in properties
                    else None
                ),
                width=(
                    float(properties["widths"][i]) if "widths" in properties else None
                ),
            )
            peak_list.append(peak_info)

        return peak_list

    def find_club_head_speed_peak(self) -> PeakInfo | None:
        """Find peak club head speed.

        Returns:
            PeakInfo for maximum club head speed
        """
        if self.club_head_speed is None or len(self.club_head_speed) == 0:
            return None

        max_idx = np.argmax(self.club_head_speed)
        return PeakInfo(
            value=float(self.club_head_speed[max_idx]),
            time=float(self.times[max_idx]),
            index=int(max_idx),
        )

    def compute_range_of_motion(self, joint_idx: int) -> tuple[float, float, float]:
        """Compute range of motion for a joint.

        Args:
            joint_idx: Joint index

        Returns:
            (min_angle, max_angle, rom) in degrees
        """
        if joint_idx >= self.joint_positions.shape[1]:
            return (0.0, 0.0, 0.0)

        angles_deg = np.rad2deg(self.joint_positions[:, joint_idx])
        min_angle = float(np.min(angles_deg))
        max_angle = float(np.max(angles_deg))
        rom = max_angle - min_angle

        return (min_angle, max_angle, rom)

    def compute_tempo(self) -> tuple[float, float, float] | None:
        """Compute swing tempo (backswing:downswing ratio).

        Uses club head speed to identify transition point.

        Returns:
            (backswing_duration, downswing_duration, ratio) or None
        """
        if self.club_head_speed is None or len(self.club_head_speed) < 10:
            return None

        # Find peak club head speed (impact)
        impact_idx = np.argmax(self.club_head_speed)

        # Find transition (minimum speed before impact, after initial movement)
        # Look in first 70% of time before impact
        search_end = int(impact_idx * 0.7)
        if search_end <= 5:
            return None

        # Smooth speed for better transition detection
        segment = self.club_head_speed[:impact_idx]
        window_len = min(11, len(segment))
        if window_len % 2 == 0:
            window_len -= 1
        smoothed_speed = (
            segment if window_len <= 3 else savgol_filter(segment, window_len, 3)
        )

        # Find minimum after initial acceleration
        start_search = 5
        transition_idx = start_search + np.argmin(
            smoothed_speed[start_search:search_end],
        )

        # Compute durations
        backswing_duration = float(self.times[transition_idx] - self.times[0])
        downswing_duration = float(self.times[impact_idx] - self.times[transition_idx])

        ratio = (
            backswing_duration / downswing_duration if downswing_duration > 0 else 0.0
        )

        return (backswing_duration, downswing_duration, ratio)

    def compute_x_factor(
        self,
        shoulder_joint_idx: int,
        hip_joint_idx: int,
    ) -> np.ndarray | None:
        """Compute X-Factor (shoulder-hip rotation difference).

        Args:
            shoulder_joint_idx: Index of shoulder/torso rotation joint
            hip_joint_idx: Index of hip rotation joint

        Returns:
            X-Factor time series (degrees) or None
        """
        if (
            shoulder_joint_idx >= self.joint_positions.shape[1]
            or hip_joint_idx >= self.joint_positions.shape[1]
        ):
            return None

        shoulder_rotation = np.rad2deg(self.joint_positions[:, shoulder_joint_idx])
        hip_rotation = np.rad2deg(self.joint_positions[:, hip_joint_idx])

        return np.asarray(shoulder_rotation - hip_rotation)

    def compute_x_factor_stretch(
        self,
        shoulder_joint_idx: int,
        hip_joint_idx: int,
    ) -> tuple[np.ndarray, float] | None:
        """Compute X-Factor velocity (stretch rate) and peak stretch.

        Args:
            shoulder_joint_idx: Index of shoulder/torso rotation joint
            hip_joint_idx: Index of hip rotation joint

        Returns:
            Tuple of (x_factor_velocity_array, peak_stretch_rate) or None
        """
        x_factor = self.compute_x_factor(shoulder_joint_idx, hip_joint_idx)
        if x_factor is None:
            return None

        # Calculate derivative (finite difference)
        x_factor_velocity = np.gradient(x_factor, self.dt)
        peak_stretch_rate = float(np.max(np.abs(x_factor_velocity)))

        return x_factor_velocity, peak_stretch_rate

    def detect_impact_time(self) -> float | None:
        """Detect ball impact time.

        Uses peak club head speed as proxy for impact.

        Returns:
            Impact time in seconds, or None
        """
        peak = self.find_club_head_speed_peak()
        return peak.time if peak else None

    def compute_energy_metrics(
        self,
        kinetic_energy: np.ndarray,
        potential_energy: np.ndarray,
    ) -> dict[str, Any]:
        """Compute energy-related metrics.

        Args:
            kinetic_energy: Kinetic energy time series
            potential_energy: Potential energy time series

        Returns:
            Dictionary of energy metrics
        """
        total_energy = kinetic_energy + potential_energy

        # Energy efficiency: ratio of kinetic energy at impact to max total energy
        if self.club_head_speed is not None:
            impact_idx = np.argmax(self.club_head_speed)
            ke_at_impact = kinetic_energy[impact_idx]
            max_total = np.max(total_energy)
            efficiency = (ke_at_impact / max_total * 100) if max_total > 0 else 0.0
        else:
            efficiency = 0.0

        # Energy conservation (should be ~constant for conservative system)
        energy_variation = np.std(total_energy)
        energy_drift = total_energy[-1] - total_energy[0]

        return {
            "max_kinetic_energy": float(np.max(kinetic_energy)),
            "max_potential_energy": float(np.max(potential_energy)),
            "max_total_energy": float(np.max(total_energy)),
            "energy_efficiency": float(efficiency),
            "energy_variation": float(energy_variation),
            "energy_drift": float(energy_drift),
        }

    def detect_swing_phases(self) -> list[SwingPhase]:
        """Automatically detect swing phases.

        Uses heuristics based on club head speed and position.

        Returns:
            List of SwingPhase objects
        """
        phases = []

        if self.club_head_speed is None or len(self.club_head_speed) < 20:
            # If no club head data, return single phase
            return [
                SwingPhase(
                    name="Complete Swing",
                    start_time=float(self.times[0]),
                    end_time=float(self.times[-1]),
                    start_index=0,
                    end_index=len(self.times) - 1,
                    duration=float(self.duration),
                ),
            ]

        # Smooth speed for phase detection
        window_len = min(11, len(self.club_head_speed))
        if window_len % 2 == 0:
            window_len -= 1

        if window_len <= 3:
            smoothed_speed = self.club_head_speed
        else:
            smoothed_speed = savgol_filter(self.club_head_speed, window_len, 3)

        # Key events
        impact_idx = np.argmax(smoothed_speed)  # Peak speed = impact

        # Find transition (top of backswing) - minimum speed before impact
        search_end = int(impact_idx * 0.7)
        if search_end > 5:
            transition_idx = 5 + np.argmin(smoothed_speed[5:search_end])
        else:
            transition_idx = impact_idx // 2

        # Find takeaway start (first significant movement)
        speed_threshold = 0.1 * smoothed_speed[transition_idx]
        takeaway_idx = 0
        for i in range(1, transition_idx):
            if smoothed_speed[i] > speed_threshold:
                takeaway_idx = i
                break

        # Find finish (speed drops after impact)
        finish_threshold = 0.3 * smoothed_speed[impact_idx]
        finish_idx = len(smoothed_speed) - 1
        for i in range(impact_idx + 1, len(smoothed_speed)):
            if smoothed_speed[i] < finish_threshold:
                finish_idx = i
                break

        # Define phases
        phase_definitions = [
            ("Address", 0, takeaway_idx),
            (
                "Takeaway",
                takeaway_idx,
                int(takeaway_idx + (transition_idx - takeaway_idx) * 0.3),
            ),
            (
                "Backswing",
                int(takeaway_idx + (transition_idx - takeaway_idx) * 0.3),
                transition_idx,
            ),
            (
                "Transition",
                transition_idx,
                int(transition_idx + (impact_idx - transition_idx) * 0.2),
            ),
            (
                "Downswing",
                int(transition_idx + (impact_idx - transition_idx) * 0.2),
                impact_idx,
            ),
            (
                "Impact",
                int(max(0, int(impact_idx) - 2)),
                int(min(len(smoothed_speed) - 1, int(impact_idx) + 2)),
            ),
            ("Follow-through", impact_idx, finish_idx),
            ("Finish", finish_idx, len(smoothed_speed) - 1),
        ]

        for name, start_idx, end_idx in phase_definitions:
            # Type cast to handle tuple unpacking
            start_idx_val: int = int(start_idx)  # type: ignore[call-overload]
            end_idx_val: int = int(end_idx)  # type: ignore[call-overload]
            start_idx = int(max(0, min(start_idx_val, len(self.times) - 1)))
            end_idx = int(max(start_idx, min(end_idx_val, len(self.times) - 1)))

            phases.append(
                SwingPhase(
                    name=name,
                    start_time=float(self.times[start_idx]),
                    end_time=float(self.times[end_idx]),
                    start_index=int(start_idx),
                    end_index=int(end_idx),
                    duration=float(self.times[end_idx] - self.times[start_idx]),
                ),
            )

        return phases

    def compute_phase_statistics(
        self,
        phases: list[SwingPhase],
        data: np.ndarray,
    ) -> dict[str, SummaryStatistics]:
        """Compute statistics for each phase.

        Args:
            phases: List of swing phases
            data: 1D data array

        Returns:
            Dictionary mapping phase name to statistics
        """
        phase_stats = {}

        for phase in phases:
            phase_data = data[phase.start_index : phase.end_index + 1]
            if len(phase_data) > 0:
                # Temporarily override times for this phase
                original_times = self.times
                self.times = self.times[phase.start_index : phase.end_index + 1]

                phase_stats[phase.name] = self.compute_summary_stats(phase_data)

                self.times = original_times

        return phase_stats

    def compute_grf_metrics(self) -> GRFMetrics | None:
        """Compute Ground Reaction Force and Center of Pressure metrics.

        Returns:
            GRFMetrics object or None if data unavailable
        """
        if self.cop_position is None or len(self.cop_position) == 0:
            return None

        # CoP Path Length
        # OPTIMIZATION: Manual slicing is slightly faster than np.diff
        cop_diff = self.cop_position[1:] - self.cop_position[:-1]
        # OPTIMIZATION: np.hypot is faster than np.linalg.norm for 2D vectors
        # OPTIMIZATION: Explicit sqrt calculation is faster than np.linalg.norm for 3D
        if cop_diff.shape[1] == 2:
            cop_dist = np.hypot(cop_diff[:, 0], cop_diff[:, 1])
        else:
            cop_dist = np.sqrt(np.sum(cop_diff**2, axis=1))
        path_length = np.sum(cop_dist)

        # CoP Velocity
        # OPTIMIZATION: Use pre-computed distance norm to avoid redundant allocations
        max_vel = np.max(cop_dist) / self.dt

        # CoP Range
        # Note: Vectorized min/max (axis=0) was found to be slower than separate column access
        x_range = float(
            np.max(self.cop_position[:, 0]) - np.min(self.cop_position[:, 0])
        )
        y_range = float(
            np.max(self.cop_position[:, 1]) - np.min(self.cop_position[:, 1])
        )

        # Force metrics
        peak_vertical = None
        peak_shear = None
        if self.ground_forces is not None:
            # Assuming Z is vertical (index 2)
            if self.ground_forces.shape[1] >= 3:
                peak_vertical = float(np.max(self.ground_forces[:, 2]))
                # Shear is magnitude of X and Y forces
                shear = np.hypot(self.ground_forces[:, 0], self.ground_forces[:, 1])
                peak_shear = float(np.max(shear))

        return GRFMetrics(
            cop_path_length=float(path_length),
            cop_max_velocity=float(max_vel),
            cop_x_range=x_range,
            cop_y_range=y_range,
            peak_vertical_force=peak_vertical,
            peak_shear_force=peak_shear,
        )

    def compute_angular_momentum_metrics(self) -> AngularMomentumMetrics | None:
        """Compute metrics related to system angular momentum.

        Returns:
            AngularMomentumMetrics object or None if data unavailable
        """
        if self.angular_momentum is None or len(self.angular_momentum) == 0:
            return None

        # OPTIMIZATION: Explicit sqrt calculation is faster than np.linalg.norm
        mag = np.sqrt(np.sum(self.angular_momentum**2, axis=1))

        peak_mag = float(np.max(mag))
        peak_idx = int(np.argmax(mag))
        peak_time = float(self.times[peak_idx])

        # Mean
        mean_mag = float(np.mean(mag))

        # Components peaks (absolute)
        peak_lx = float(np.max(np.abs(self.angular_momentum[:, 0])))
        peak_ly = float(np.max(np.abs(self.angular_momentum[:, 1])))
        peak_lz = float(np.max(np.abs(self.angular_momentum[:, 2])))

        # Variability
        std_mag = float(np.std(mag))
        variability = std_mag / mean_mag if mean_mag > 0 else 0.0

        return AngularMomentumMetrics(
            peak_magnitude=peak_mag,
            peak_time=peak_time,
            mean_magnitude=mean_mag,
            peak_lx=peak_lx,
            peak_ly=peak_ly,
            peak_lz=peak_lz,
            variability=variability,
        )

    def compute_stability_metrics(self) -> StabilityMetrics | None:
        """Compute postural stability metrics.

        Requires both CoP and CoM positions.

        Returns:
            StabilityMetrics object or None if data unavailable
        """
        if (
            self.cop_position is None
            or self.com_position is None
            or len(self.cop_position) != len(self.com_position)
        ):
            return None

        # Horizontal plane distance (X-Y)
        # Note: Depending on coordinate system, vertical might be Z or Y.
        # MuJoCo standard is Z-up. We assume Z is vertical.
        # CoP is usually 3D on floor (Z=0) or 2D.

        cop_xy = self.cop_position[:, :2]
        com_xy = self.com_position[:, :2]

        # OPTIMIZATION: np.hypot is faster for 2D vectors
        diff = cop_xy - com_xy
        dist = np.hypot(diff[:, 0], diff[:, 1])

        # Inclination Angle (Angle between vertical and CoP-CoM vector)
        # Vector P = CoM - CoP
        # If CoP is 2D, assume Z=0
        if self.cop_position.shape[1] == 2:
            cop_z = np.zeros(len(self.cop_position))
        else:
            cop_z = self.cop_position[:, 2]  # type: ignore[assignment]

        vec_temp = self.com_position - np.column_stack((cop_xy, cop_z))
        vec: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = vec_temp  # type: ignore[assignment]

        # Angle with vertical (Z-axis [0, 0, 1])
        # dot(v, k) = |v| * |k| * cos(theta)
        # theta = arccos( v_z / |v| )

        # OPTIMIZATION: Explicit sqrt sum is faster than np.linalg.norm
        vec_norm = np.sqrt(np.sum(vec**2, axis=1))
        # Avoid division by zero
        vec_norm[vec_norm < 1e-6] = 1.0

        cos_theta = vec[:, 2] / vec_norm
        # Clip for numerical stability
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        angles_rad = np.arccos(cos_theta)
        angles_deg = np.rad2deg(angles_rad)

        return StabilityMetrics(
            min_com_cop_distance=float(np.min(dist)),
            max_com_cop_distance=float(np.max(dist)),
            mean_com_cop_distance=float(np.mean(dist)),
            peak_inclination_angle=float(np.max(angles_deg)),
            mean_inclination_angle=float(np.mean(angles_deg)),
        )

    def compute_coordination_metrics(
        self, joint_idx_1: int, joint_idx_2: int
    ) -> CoordinationMetrics | None:
        """Compute coordination metrics from coupling angles (Vector Coding).

        Classifies coordination into 4 patterns:
        - In-Phase: Both segments rotating in same direction
        - Anti-Phase: Segments rotating in opposite directions
        - Proximal Leading: Proximal segment dominates motion
        - Distal Leading: Distal segment dominates motion

        Standard binning (Chang et al.):
        - In-Phase: 45 +/- 22.5, 225 +/- 22.5
        - Proximal: 0 +/- 22.5, 180 +/- 22.5
        - Distal: 90 +/- 22.5, 270 +/- 22.5
        - Anti-Phase: 135 +/- 22.5, 315 +/- 22.5

        Args:
            joint_idx_1: Proximal joint index (X-axis)
            joint_idx_2: Distal joint index (Y-axis)

        Returns:
            CoordinationMetrics object or None
        """
        angles = self.compute_coupling_angles(joint_idx_1, joint_idx_2)
        if len(angles) == 0:
            return None

        # Bin counts
        # Map 0-360 to 0-360

        # Define bins centers
        # Proximal: 0, 180, 360
        # In-Phase: 45, 225
        # Distal: 90, 270
        # Anti-Phase: 135, 315

        # We can map angle to 8 bins of 45 degrees, centered on 0, 45, 90...
        # Shift by 22.5 to make integer division work easier
        # (angle + 22.5) // 45

        binned = np.floor((angles + 22.5) / 45.0) % 8

        # 0: 0 +/- 22.5 -> Proximal
        # 1: 45 +/- 22.5 -> In-Phase
        # 2: 90 +/- 22.5 -> Distal
        # 3: 135 +/- 22.5 -> Anti-Phase
        # 4: 180 +/- 22.5 -> Proximal
        # 5: 225 +/- 22.5 -> In-Phase
        # 6: 270 +/- 22.5 -> Distal
        # 7: 315 +/- 22.5 -> Anti-Phase

        counts = np.bincount(binned.astype(int), minlength=8)
        total = len(angles)

        proximal_cnt = counts[0] + counts[4]
        in_phase_cnt = counts[1] + counts[5]
        distal_cnt = counts[2] + counts[6]
        anti_phase_cnt = counts[3] + counts[7]

        # Circular statistics for mean and variability
        # Mean vector
        angles_rad = np.deg2rad(angles)
        R = (
            np.sqrt(np.sum(np.cos(angles_rad)) ** 2 + np.sum(np.sin(angles_rad)) ** 2)
            / total
        )
        mean_angle_rad = np.arctan2(
            np.sum(np.sin(angles_rad)), np.sum(np.cos(angles_rad))
        )
        mean_angle_deg = np.degrees(mean_angle_rad) % 360.0

        # Circular standard deviation = sqrt(-2 * ln(R))
        # Note: R is mean resultant length [0, 1]
        if R < 1.0:
            circ_std = np.sqrt(-2 * np.log(R))
            circ_std_deg = np.degrees(circ_std)
        else:
            circ_std_deg = 0.0

        return CoordinationMetrics(
            in_phase_pct=float(in_phase_cnt / total * 100),
            anti_phase_pct=float(anti_phase_cnt / total * 100),
            proximal_leading_pct=float(proximal_cnt / total * 100),
            distal_leading_pct=float(distal_cnt / total * 100),
            mean_coupling_angle=float(mean_angle_deg),
            coordination_variability=float(circ_std_deg),
        )

    def generate_comprehensive_report(self) -> dict[str, Any]:
        """Generate comprehensive statistical report.

        Returns:
            Dictionary with all analysis results
        """
        report: dict[str, Any] = {
            "duration": float(self.duration),
            "sample_rate": float(1.0 / self.dt) if self.dt > 0 else 0.0,
            "num_samples": len(self.times),
        }

        # Club head speed analysis
        if self.club_head_speed is not None:
            peak_speed = self.find_club_head_speed_peak()
            if peak_speed:
                report["club_head_speed"] = {
                    "peak_value": peak_speed.value,
                    "peak_time": peak_speed.time,
                    "statistics": asdict(
                        self.compute_summary_stats(
                            self.club_head_speed,
                        )
                    ),
                }

        # Tempo analysis
        tempo_result = self.compute_tempo()
        if tempo_result:
            report["tempo"] = {
                "backswing_duration": tempo_result[0],
                "downswing_duration": tempo_result[1],
                "ratio": tempo_result[2],
            }

        # Swing phases
        phases = self.detect_swing_phases()
        report["phases"] = [
            {
                "name": p.name,
                "start_time": p.start_time,
                "end_time": p.end_time,
                "duration": p.duration,
            }
            for p in phases
        ]

        # GRF Metrics
        grf_metrics = self.compute_grf_metrics()
        if grf_metrics:
            report["grf_metrics"] = asdict(grf_metrics)

        # Angular Momentum Metrics
        am_metrics = self.compute_angular_momentum_metrics()
        if am_metrics:
            report["angular_momentum_metrics"] = asdict(am_metrics)

        # Stability Metrics
        stability_metrics = self.compute_stability_metrics()
        if stability_metrics:
            report["stability_metrics"] = asdict(stability_metrics)

        # Joint statistics
        report["joints"] = {}
        for i in range(self.joint_positions.shape[1]):
            angles_deg = np.rad2deg(self.joint_positions[:, i])
            position_stats = self.compute_summary_stats(angles_deg)

            velocities = (
                self.joint_velocities[:, i]
                if i < self.joint_velocities.shape[1]
                else None
            )

            joint_stats = {
                "range_of_motion": {
                    "min_deg": position_stats.min,
                    "max_deg": position_stats.max,
                    "rom_deg": position_stats.range,
                },
                "position_stats": asdict(position_stats),
            }

            if velocities is not None:
                joint_stats["velocity_stats"] = asdict(
                    self.compute_summary_stats(
                        np.rad2deg(velocities),
                    )
                )

            if i < self.joint_torques.shape[1]:
                joint_stats["torque_stats"] = asdict(
                    self.compute_summary_stats(
                        self.joint_torques[:, i],
                    )
                )

            report["joints"][f"joint_{i}"] = joint_stats

        return report

    def compute_frequency_analysis(
        self,
        data: np.ndarray,
        window: str = "hann",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute frequency analysis (PSD).

        Args:
            data: Input time series data
            window: Window function

        Returns:
            (frequencies, psd_values)
        """
        fs = 1.0 / self.dt if self.dt > 0 else 0.0
        if fs == 0.0:
            return np.array([]), np.array([])

        try:
            from shared.python import signal_processing

            return signal_processing.compute_psd(data, fs, window=window)
        except ImportError:
            # Fallback if shared module not found
            from scipy import signal

            freqs, psd = signal.welch(data, fs=fs, window=window)
            return freqs, psd

    def compute_smoothness_metric(self, data: np.ndarray) -> float:
        """Compute smoothness metric (Spectral Arc Length).

        Args:
            data: Velocity profile (or other signal)

        Returns:
            Smoothness score (negative dimensionless value)
        """
        fs = 1.0 / self.dt if self.dt > 0 else 0.0
        if fs == 0.0:
            return 0.0

        try:
            from shared.python import signal_processing

            return signal_processing.compute_spectral_arc_length(data, fs)
        except ImportError:
            return 0.0

    def analyze_kinematic_sequence(
        self,
        segment_indices: dict[str, int],
    ) -> tuple[list[KinematicSequenceInfo], float]:
        """Analyze the kinematic sequence of the swing.

        The kinematic sequence refers to the proximal-to-distal sequencing of
        peak rotational velocities (e.g., Pelvis -> Thorax -> Arm -> Club).

        Args:
            segment_indices: Dictionary mapping segment names to joint indices.
                             Example: {'Pelvis': 0, 'Thorax': 1, 'Arm': 2}

        Returns:
            Tuple of:
            - List of KinematicSequenceInfo objects sorted by peak time
            - Sequence efficiency score (0.0 to 1.0, 1.0 being perfect order)
        """
        sequence_info = []

        # Analyze each segment
        for segment_name, joint_idx in segment_indices.items():
            if joint_idx >= self.joint_velocities.shape[1]:
                continue

            # Get velocity magnitude
            velocities = np.abs(self.joint_velocities[:, joint_idx])

            # Find peak
            max_idx = np.argmax(velocities)
            peak_val = float(velocities[max_idx])
            peak_time = float(self.times[max_idx])

            sequence_info.append(
                KinematicSequenceInfo(
                    segment_name=segment_name,
                    peak_velocity=peak_val,
                    peak_time=peak_time,
                    peak_index=int(max_idx),
                    order_index=0,  # Will be set later
                ),
            )

        # Sort by peak time
        sequence_info.sort(key=lambda x: x.peak_time)

        # Assign order indices
        for i, info in enumerate(sequence_info):
            info.order_index = i

        # Calculate efficiency score
        # Ideally, the order should match the expected proximal-to-distal order
        # which is implied by the order of keys in segment_indices
        # (if it's an OrderedDict or Python 3.7+ dict).
        # However, since we can't guarantee the input order is the "correct" order,
        # we'll assume the user provides a list of segments in the expected order.
        expected_order = list(segment_indices.keys())
        actual_order = [info.segment_name for info in sequence_info]

        # Calculate Levenshtein distance or simpler match score
        matches = sum(
            1 for e, a in zip(expected_order, actual_order, strict=False) if e == a
        )
        efficiency_score = matches / len(expected_order) if expected_order else 0.0

        return sequence_info, efficiency_score

    def compute_correlations(
        self,
        data_type: str = "velocity",
    ) -> tuple[np.ndarray, list[str]]:
        """Compute correlation matrix for joint data.

        Args:
            data_type: Type of data to correlate ('position', 'velocity', 'torque')

        Returns:
            Tuple of (correlation_matrix, labels)
        """
        if data_type == "position":
            data = self.joint_positions
        elif data_type == "torque":
            data = self.joint_torques
        else:  # velocity
            data = self.joint_velocities

        if data.shape[1] == 0:
            return np.array([]), []

        # Calculate correlation matrix
        # Transpose so that rows are variables (joints), columns are observations
        corr_matrix = np.corrcoef(data.T)

        labels = [f"J{i}" for i in range(data.shape[1])]

        return corr_matrix, labels

    def compute_rolling_correlation(
        self,
        joint_idx_1: int,
        joint_idx_2: int,
        window_size: int = 20,
        data_type: str = "velocity",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute rolling correlation between two joints.

        Args:
            joint_idx_1: First joint index
            joint_idx_2: Second joint index
            window_size: Size of rolling window in samples
            data_type: 'position', 'velocity', or 'torque'

        Returns:
            Tuple of (times, correlations). Times correspond to window centers.
        """
        if data_type == "position":
            data = self.joint_positions
        elif data_type == "torque":
            data = self.joint_torques
        else:
            data = self.joint_velocities

        if (
            data.shape[1] == 0
            or joint_idx_1 >= data.shape[1]
            or joint_idx_2 >= data.shape[1]
            or len(data) < window_size
        ):
            return np.array([]), np.array([])

        x = data[:, joint_idx_1]
        y = data[:, joint_idx_2]

        # Use stride tricks or pandas-like rolling
        # Implementing manually with vectorized approach for efficiency
        # Rolling correlation:
        # corr = (mean(xy) - mean(x)mean(y)) / (std(x)std(y))
        # But for rolling window.

        # OPTIMIZATION: Use convolution for O(N) complexity instead of O(N*W)
        # Reduces memory allocation and computation significantly
        # Previous implementation used sliding_window_view which created (N, W) views
        kernel = np.ones(window_size)
        n = window_size

        s_x = np.convolve(x, kernel, mode="valid")
        s_y = np.convolve(y, kernel, mode="valid")
        s_xy = np.convolve(x * y, kernel, mode="valid")
        s_xx = np.convolve(x**2, kernel, mode="valid")
        s_yy = np.convolve(y**2, kernel, mode="valid")

        # Numerator: sum((x-mx)(y-my)) = sum(xy) - sum(x)sum(y)/n
        numerator = s_xy - (s_x * s_y) / n

        # Denominator terms: sum((x-mx)^2) = sum(x^2) - sum(x)^2/n
        var_x = s_xx - (s_x**2) / n
        var_y = s_yy - (s_y**2) / n

        # Clamp variances to be non-negative (handle numerical noise)
        var_x = np.maximum(var_x, 0)
        var_y = np.maximum(var_y, 0)

        denominator = np.sqrt(var_x * var_y)

        # Handle divide by zero (constant signal in window)
        with np.errstate(divide="ignore", invalid="ignore"):
            correlations = numerator / denominator
        correlations[np.isnan(correlations)] = 0.0

        # Time points (center of window)
        # Indices: 0 to N-window.
        # Window 0 covers 0..window-1. Center ~ window/2
        valid_indices = np.arange(len(correlations)) + window_size // 2
        window_times = self.times[valid_indices]

        return window_times, correlations

    def compute_local_divergence_rate(
        self,
        joint_idx: int = 0,
        tau: int = 1,
        dim: int = 3,
        window: int = 50,
        data_type: str = "velocity",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute local divergence rate over time (Local Lyapunov proxy).

        Instead of a single exponent, this returns a time series showing how
        fast nearby trajectories are diverging *at each point in time*.

        Args:
            joint_idx: Joint index
            tau: Time lag
            dim: Embedding dimension
            window: Theiler window
            data_type: 'position' or 'velocity'

        Returns:
            Tuple of (times, divergence_rates)
        """
        if data_type == "position":
            data = self.joint_positions[:, joint_idx]
        else:
            data = self.joint_velocities[:, joint_idx]

        N = len(data)
        M = N - (dim - 1) * tau
        if M < window + 1:
            return np.array([]), np.array([])

        # 1. Reconstruct Phase Space
        orbit = np.zeros((M, dim))
        for d in range(dim):
            orbit[:, d] = data[d * tau : d * tau + M]

        # 2. Find Nearest Neighbors for each point
        from scipy.spatial.distance import cdist

        # We compute distance matrix but need to be careful with memory for large N
        # For typical N=500, 500x500 is fine.
        dists_mat = cdist(orbit, orbit, metric="euclidean")

        # Apply Theiler window
        for i in range(M):
            start = max(0, i - window)
            end = min(M, i + window + 1)
            dists_mat[i, start:end] = np.inf
            dists_mat[i, i] = np.inf

        nearest_neighbors = np.argmin(dists_mat, axis=1)
        # initial_dists = np.min(dists_mat, axis=1)

        # 3. Compute Divergence Rate for each point
        # Look ahead a short time (e.g., 5-10 steps) and see how much they separated
        lookahead = min(10, int(0.1 / self.dt)) if self.dt > 0 else 5
        divergence_rates = np.zeros(M - lookahead)

        # OPTIMIZATION: Vectorized loop calculation (approx. 45% faster)
        # Vectorized indices
        indices = np.arange(M - lookahead)
        nn_indices = nearest_neighbors[indices]

        # Filter valid neighbors (not too close to end)
        valid_mask = nn_indices < (M - lookahead)
        valid_i = indices[valid_mask]
        valid_nn = nn_indices[valid_mask]

        if len(valid_i) > 0:
            # Initial vector distances
            # (N_valid, dim)
            diff_0 = orbit[valid_i] - orbit[valid_nn]
            # OPTIMIZATION: Use squared distances to avoid sqrt calls
            dist_sq_0 = np.sum(diff_0**2, axis=1)

            # Final vector distances
            diff_t = orbit[valid_i + lookahead] - orbit[valid_nn + lookahead]
            dist_sq_t = np.sum(diff_t**2, axis=1)

            # Calculate rates where distances are non-zero (squared threshold 1e-18 corresponds to 1e-9)
            safe_mask = (dist_sq_0 > 1e-18) & (dist_sq_t > 1e-18)

            # Pre-calculate denominator
            denom = lookahead * self.dt

            # Assign valid rates
            # valid_i[safe_mask] maps back to original indices where both neighbor valid and dists valid
            # Optimization: log(sqrt(a)/sqrt(b)) = 0.5 * log(a/b)
            if denom > 0:
                divergence_rates[valid_i[safe_mask]] = (
                    0.5 * np.log(dist_sq_t[safe_mask] / dist_sq_0[safe_mask]) / denom
                )

        # Align times
        # Divergence rate at index i corresponds to time[i] (roughly)
        valid_times = self.times[: len(divergence_rates)]

        return valid_times, divergence_rates

    def compute_coupling_angles(
        self,
        joint_idx_1: int,
        joint_idx_2: int,
    ) -> np.ndarray:
        """Compute Vector Coding coupling angles between two joints.

        The coupling angle represents the direction of the vector between
        successive data points in an Angle-Angle diagram. It quantifies
        the coordination pattern between the two segments.

        Args:
            joint_idx_1: Index of the first joint (x-axis)
            joint_idx_2: Index of the second joint (y-axis)

        Returns:
            Array of coupling angles in degrees [0, 360)
        """
        if (
            joint_idx_1 >= self.joint_velocities.shape[1]
            or joint_idx_2 >= self.joint_velocities.shape[1]
        ):
            return np.array([])

        # Use velocities for instantaneous tangent
        # Gamma = arctan2(vel2, vel1)
        vel1 = self.joint_velocities[:, joint_idx_1]
        vel2 = self.joint_velocities[:, joint_idx_2]

        # Handle near-zero velocities to avoid noise
        # (Optional: thresholding, but raw calc is standard)

        gamma_rad: np.ndarray = np.arctan2(vel2, vel1)
        gamma_deg: np.ndarray = np.rad2deg(gamma_rad)

        # Normalize to [0, 360)
        gamma_deg = np.mod(gamma_deg, 360.0)

        return np.asarray(gamma_deg)

    def compute_phase_angle(self, joint_idx: int) -> np.ndarray:
        """Compute continuous phase angle for a joint.

        Calculated using the arctangent of normalized velocity vs normalized position.
        Phi = arctan2(norm_vel, norm_pos).

        Args:
            joint_idx: Index of the joint

        Returns:
            Array of phase angles in degrees (unwrapped)
        """
        if (
            joint_idx >= self.joint_positions.shape[1]
            or joint_idx >= self.joint_velocities.shape[1]
        ):
            return np.array([])

        pos = self.joint_positions[:, joint_idx]
        vel = self.joint_velocities[:, joint_idx]

        if len(pos) < 2:
            return np.array([])

        # Normalize Position to [-1, 1] using dynamic range
        # (centered around midpoint of ROM)
        min_p, max_p = np.min(pos), np.max(pos)
        range_p = max_p - min_p
        if range_p < 1e-6:
            norm_pos = np.zeros_like(pos)
        else:
            # Scale to [0, 1] then to [-1, 1]
            norm_pos = 2 * (pos - min_p) / range_p - 1.0

        # Normalize Velocity by max absolute velocity
        # (Preserves zero crossing)
        max_v = np.max(np.abs(vel))
        if max_v < 1e-6:
            norm_vel = np.zeros_like(vel)
        else:
            norm_vel = vel / max_v

        # Calculate phase angle
        phase = np.arctan2(norm_vel, norm_pos)

        # Unwrap to make continuous
        phase_unwrapped = np.unwrap(phase)

        return np.asarray(np.rad2deg(phase_unwrapped))

    def compute_continuous_relative_phase(
        self,
        joint_idx_1: int,
        joint_idx_2: int,
    ) -> np.ndarray:
        """Compute Continuous Relative Phase (CRP) between two joints.

        CRP = Phase1 - Phase2.
        Used to analyze coordination dynamics. Near 0 implies in-phase,
        near 180 implies anti-phase.

        Args:
            joint_idx_1: Proximal joint index
            joint_idx_2: Distal joint index

        Returns:
            Array of CRP values in degrees
        """
        phi1 = self.compute_phase_angle(joint_idx_1)
        phi2 = self.compute_phase_angle(joint_idx_2)

        if len(phi1) == 0 or len(phi2) == 0:
            return np.array([])

        # Calculate difference
        crp = phi1 - phi2

        # Optionally wrap to specific range, but typically analysis looks at
        # the continuous curve or variability.
        # For display, we often wrap to [-180, 180] or [0, 180] absolute
        # but returning raw difference preserves the most info.
        return np.asarray(crp)

    def compute_swing_profile(self) -> SwingProfileMetrics | None:
        """Compute Swing Profile scores (0-100) for radar chart visualization.

        Heuristic scoring based on typical professional golf swing data.
        Note: These are rough approximations for visualization purposes.

        Returns:
            SwingProfileMetrics object or None if insufficient data
        """
        # 1. Speed Score
        # Pro avg club speed ~113 mph. Let's say 120 mph = 100 score.
        speed_score = 0.0
        if self.club_head_speed is not None:
            peak_speed = float(np.max(self.club_head_speed))  # m/s
            peak_speed_mph = peak_speed * 2.23694
            speed_score = float(min(100.0, (peak_speed_mph / 120.0) * 100.0))

        # 2. Sequence Score
        # Based on kinematic sequence efficiency
        sequence_score = 0.0
        # Assume standard order indices if available
        # We need segment indices. We'll try to guess based on joint count
        # or just skip if we can't reliably determine.
        # Actually, let's use a simpler proxy if we don't have segment names:
        # Check peak timing of first 3 joints.
        if self.joint_velocities.shape[1] >= 3:
            peaks = []
            for i in range(3):
                idx = np.argmax(np.abs(self.joint_velocities[:, i]))
                peaks.append(idx)
            # Check if sorted ascending (0->1->2)
            if peaks == sorted(peaks, key=int):
                sequence_score = 100.0
            else:
                # Deduct points for out of order
                sequence_score = 50.0  # Baseline
        else:
            sequence_score = 0.0

        # 3. Stability Score
        # Based on CoP-CoM distance variance or inclination
        stability_score = 0.0
        stab_metrics = self.compute_stability_metrics()
        if stab_metrics:
            # Lower inclination is generally more stable/balanced?
            # Or consistent CoM-CoP margin.
            # Let's use inclination. < 10 deg is great, > 30 is bad.
            angle = stab_metrics.mean_inclination_angle
            stability_score = float(max(0.0, min(100.0, 100.0 - (angle - 5.0) * 4.0)))
        else:
            stability_score = 0.0

        # 4. Efficiency Score
        # Kinetic Energy transfer efficiency
        efficiency_score = 0.0
        if (
            self.club_head_speed is not None
            and self.joint_torques.shape[1] > 0
            and self.joint_velocities.shape[1] > 0
        ):
            # Energy metrics
            ke = 0.5 * 1.0 * (self.club_head_speed**2)  # Dummy mass 1kg
            # Calculate total work done (vectorized for all joints at once)
            n_joints = self.joint_torques.shape[1]
            n_samples = min(
                self.joint_torques.shape[0],
                self.joint_velocities.shape[0],
                len(self.times),
            )

            if n_samples >= 2:
                # Vectorized computation: calculate work for all joints at once
                torques = self.joint_torques[:n_samples, :n_joints]
                velocities = self.joint_velocities[:n_samples, :n_joints]
                power = torques * velocities
                pos_power = np.maximum(power, 0)

                # Integrate positive power across time for each joint
                if hasattr(np, "trapezoid"):
                    # NumPy 2.0+
                    total_work = float(
                        np.trapezoid(pos_power, dx=self.dt, axis=0).sum()
                    )
                else:
                    # Older NumPy
                    trapz_func = getattr(np, "trapz")  # noqa: B009
                    total_work = float(trapz_func(pos_power, dx=self.dt, axis=0).sum())
            else:
                total_work = 0.0

            if total_work > 0:
                peak_ke = float(np.max(ke))
                # Ratio of output KE to input Work
                # (This is rough, ignores body KE, but serves as proxy)
                eff = peak_ke / total_work
                efficiency_score = float(min(100.0, eff * 200.0))  # Scaling factor
        else:
            efficiency_score = 0.0

        # 5. Power Score (vectorized computation - 2-3× faster)
        # Peak total power
        power_score = 0.0
        if self.joint_torques.shape[1] > 0 and self.joint_velocities.shape[1] > 0:
            # Vectorized: compute all joint powers at once instead of loop
            n_joints = min(self.joint_torques.shape[1], self.joint_velocities.shape[1])
            # Element-wise multiplication across all joints, then sum along joint axis
            total_power = np.sum(
                self.joint_torques[:, :n_joints] * self.joint_velocities[:, :n_joints],
                axis=1,
            )
            peak_power = float(np.max(total_power))
            # Pro might generate > 3000 W?
            power_score = float(min(100.0, (peak_power / 3000.0) * 100.0))

        return SwingProfileMetrics(
            speed_score=float(speed_score),
            sequence_score=float(sequence_score),
            stability_score=float(stability_score),
            efficiency_score=float(efficiency_score),
            power_score=float(power_score),
        )

    def compute_work_metrics(self, joint_idx: int) -> dict[str, float] | None:
        """Compute mechanical work metrics for a joint.

        Work is calculated as the time integral of power (torque * angular velocity).
        Results are cached for performance (30-40% faster for repeated calls).

        Args:
            joint_idx: Index of the joint

        Returns:
            Dictionary with 'positive_work', 'negative_work', 'net_work' (Joules)
            or None if data unavailable.
        """
        # Check cache first (performance optimization)
        if joint_idx in self._work_metrics_cache:
            return self._work_metrics_cache[joint_idx]

        if (
            joint_idx >= self.joint_torques.shape[1]
            or joint_idx >= self.joint_velocities.shape[1]
        ):
            return None

        # Power = Torque * Velocity (Nm * rad/s = Watts)
        torque = self.joint_torques[:, joint_idx]
        velocity = self.joint_velocities[:, joint_idx]

        # Handle potentially different lengths if recorder had issues (unlikely but safe)
        n_samples = min(len(torque), len(velocity), len(self.times))
        if n_samples < 2:
            return None

        torque = torque[:n_samples]
        velocity = velocity[:n_samples]
        dt = self.dt

        power = torque * velocity

        # Calculate work via integration (Euler or Trapezoidal)
        # Positive work: Energy generation (concentric)
        # Negative work: Energy absorption (eccentric)
        pos_power = np.maximum(power, 0)
        neg_power = np.minimum(power, 0)

        # Integrate
        # Handle NumPy 2.0 deprecation of trapz
        if hasattr(np, "trapezoid"):
            positive_work = np.trapezoid(pos_power, dx=dt)
            negative_work = np.trapezoid(neg_power, dx=dt)
        else:
            # Fallback for NumPy versions that still provide trapz
            trapz_func = getattr(np, "trapz")  # noqa: B009
            positive_work = trapz_func(pos_power, dx=dt)
            negative_work = trapz_func(neg_power, dx=dt)

        net_work = positive_work + negative_work

        result = {
            "positive_work": float(positive_work),
            "negative_work": float(negative_work),
            "net_work": float(net_work),
        }

        # Cache the result
        self._work_metrics_cache[joint_idx] = result

        return result

    def compute_joint_power_metrics(self, joint_idx: int) -> JointPowerMetrics | None:
        """Compute detailed power metrics for a joint.

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

        # Generation (Power > 0)
        gen_indices = power > 0
        peak_gen = float(np.max(power)) if np.any(gen_indices) else 0.0
        avg_gen = float(np.mean(power[gen_indices])) if np.any(gen_indices) else 0.0
        gen_dur = float(np.sum(gen_indices) * dt)

        # Absorption (Power < 0)
        abs_indices = power < 0
        peak_abs = float(np.min(power)) if np.any(abs_indices) else 0.0
        avg_abs = float(np.mean(power[abs_indices])) if np.any(abs_indices) else 0.0
        abs_dur = float(np.sum(abs_indices) * dt)

        # Net work (integration)
        if hasattr(np, "trapezoid"):
            net_work = float(np.trapezoid(power, dx=dt))
        else:
            trapz_func = getattr(np, "trapz")  # noqa: B009
            net_work = float(trapz_func(power, dx=dt))

        return JointPowerMetrics(
            peak_generation=peak_gen,
            peak_absorption=peak_abs,
            avg_generation=avg_gen,
            avg_absorption=avg_abs,
            net_work=net_work,
            generation_duration=gen_dur,
            absorption_duration=abs_dur,
        )

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

        # Integrate
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

        This metric quantifies the excursion or complexity of the joint trajectory.
        Computed in normalized units (radians and radians/second).

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

        # Calculate Euclidean distance between consecutive points in phase space
        # OPTIMIZATION: Slicing is faster than np.diff
        d_pos = pos[1:] - pos[:-1]
        d_vel = vel[1:] - vel[:-1]

        dist = np.sqrt(d_pos**2 + d_vel**2)
        return float(np.sum(dist))

    def compute_recurrence_matrix(
        self,
        threshold_ratio: float = 0.1,
        metric: str = "euclidean",
    ) -> np.ndarray[tuple[int, int], np.dtype[np.int_]]:
        """Compute Recurrence Plot matrix.

        Constructs a phase space state vector from all joint positions and velocities,
        normalizes it, and calculates the binary recurrence matrix.

        Args:
            threshold_ratio: Threshold distance as ratio of max phase space diameter.
            metric: Distance metric (e.g., 'euclidean', 'cityblock').

        Returns:
            Binary recurrence matrix (N, N).
        """
        if (
            self.joint_positions.shape[1] == 0
            or self.joint_velocities.shape[1] == 0
            or len(self.times) < 2
        ):
            return np.array([], dtype=np.int_)

        # 1. Construct State Vector [positions, velocities]
        # (N, 2*nq)
        state_vec = np.hstack((self.joint_positions, self.joint_velocities))

        # 2. Normalize features (Z-score) to handle different units/scales
        mean = np.mean(state_vec, axis=0)
        std = np.std(state_vec, axis=0)
        # Avoid division by zero for constant features
        std[std < 1e-6] = 1.0
        normalized_state = (state_vec - mean) / std

        # 3. Compute Distance Matrix
        dists = pdist(normalized_state, metric=metric)
        dist_matrix = squareform(dists)

        # 4. Determine Threshold
        if threshold_ratio is None:
            threshold_ratio = 0.1
        threshold = threshold_ratio * np.max(dist_matrix)

        # 5. Thresholding
        recurrence_matrix = (dist_matrix < threshold).astype(np.int_)

        return cast(np.ndarray[tuple[int, int], np.dtype[np.int_]], recurrence_matrix)

    def compute_cross_recurrence_matrix(
        self,
        joint_idx_1: int,
        joint_idx_2: int,
        threshold_ratio: float = 0.1,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.int_]]:
        """Compute Cross Recurrence Plot matrix between two joints.

        Args:
            joint_idx_1: First joint index
            joint_idx_2: Second joint index
            threshold_ratio: Threshold distance as ratio of max distance

        Returns:
            Binary recurrence matrix (N, N)
        """
        # Construct state vectors for each joint (pos, vel)
        # N x 2
        s1 = np.column_stack(
            (
                self.joint_positions[:, joint_idx_1],
                self.joint_velocities[:, joint_idx_1],
            )
        )
        s2 = np.column_stack(
            (
                self.joint_positions[:, joint_idx_2],
                self.joint_velocities[:, joint_idx_2],
            )
        )

        # Normalize
        s1 = (s1 - np.mean(s1, axis=0)) / (np.std(s1, axis=0) + 1e-9)
        s2 = (s2 - np.mean(s2, axis=0)) / (np.std(s2, axis=0) + 1e-9)

        # Compute distance matrix between s1 and s2
        # cdist(s1, s2)
        from scipy.spatial.distance import cdist

        dist_matrix = cdist(s1, s2, metric="euclidean")

        threshold = threshold_ratio * np.max(dist_matrix)
        recurrence_matrix = (dist_matrix < threshold).astype(np.int_)

        return cast(np.ndarray[tuple[int, int], np.dtype[np.int_]], recurrence_matrix)

    def compute_rqa_metrics(
        self,
        recurrence_matrix: np.ndarray,
        min_line_length: int = 2,
    ) -> RQAMetrics | None:
        """Compute Recurrence Quantification Analysis (RQA) metrics.

        Args:
            recurrence_matrix: Binary recurrence matrix (N, N)
            min_line_length: Minimum length to consider a line (diagonal/vertical)

        Returns:
            RQAMetrics object or None
        """
        if recurrence_matrix.size == 0:
            return None

        N = recurrence_matrix.shape[0]
        if N < 2:
            return None

        # 1. Recurrence Rate (RR)
        # Exclude main diagonal? Often yes, RQA usually excludes LOI (Line of Identity).
        # But simple density includes it or excludes it. Standard is excluding.
        # But if matrix includes 1s on diagonal, we subtract N.
        n_recurrence_points = np.sum(recurrence_matrix) - N
        rr = n_recurrence_points / (N * N - N) if N > 1 else 0.0

        # 2. Diagonal Lines (Determinism)
        # Extract diagonals
        # We need to scan diagonals k=1 to N-1
        diagonal_lengths: list[int] = []
        for k in range(1, N):
            diag = np.diagonal(recurrence_matrix, offset=k)
            # Find lengths of consecutive 1s
            # Pad with 0 to find edges
            d = np.concatenate((np.array([0]), diag, np.array([0])))
            diffs = np.diff(d)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            lengths = ends - starts
            diagonal_lengths.extend(lengths[lengths >= min_line_length])

        n_diag_points = np.sum(diagonal_lengths)
        det = n_diag_points / n_recurrence_points if n_recurrence_points > 0 else 0.0
        l_max = np.max(diagonal_lengths) if len(diagonal_lengths) > 0 else 0

        # 3. Vertical Lines (Laminarity)
        # Scan columns
        vertical_lengths: list[int] = []
        for i in range(N):
            col = recurrence_matrix[:, i].copy()
            # To be consistent with Recurrence Rate (which excludes LOI),
            # we should exclude the main diagonal point (i, i) from vertical line analysis?
            # Standard RQA often includes LOI in RR, but my implementation of RR excludes it.
            # If we exclude LOI from denominator, we must exclude it from numerator (vertical points).
            # Let's zero out the diagonal point for vertical line detection to keep it consistent < 1.
            col[i] = 0

            c = np.concatenate((np.array([0]), col, np.array([0])))
            diffs = np.diff(c)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            lengths = ends - starts
            vertical_lengths.extend(lengths[lengths >= min_line_length])

        n_vert_points = np.sum(vertical_lengths)
        lam = n_vert_points / n_recurrence_points if n_recurrence_points > 0 else 0.0
        tt = float(np.mean(vertical_lengths)) if len(vertical_lengths) > 0 else 0.0

        return RQAMetrics(
            recurrence_rate=float(rr),
            determinism=float(det),
            laminarity=float(lam),
            longest_diagonal_line=int(l_max),
            trapping_time=tt,
        )

    def compute_correlation_dimension(
        self, data: np.ndarray, tau: int = 1, dim: int = 3
    ) -> float:
        """Estimate Correlation Dimension (D2) using Grassberger-Procaccia algorithm.

        Args:
            data: Time series
            tau: Time delay
            dim: Embedding dimension

        Returns:
            Estimated Correlation Dimension (slope of log C(r) vs log r)
        """
        N = len(data)
        M = N - (dim - 1) * tau
        if M < 20:
            return 0.0

        # Reconstruct phase space
        orbit = np.zeros((M, dim))
        for d in range(dim):
            orbit[:, d] = data[d * tau : d * tau + M]

        # Calculate pairwise distances (vectorized)
        from scipy.spatial.distance import pdist

        dists = pdist(orbit, metric="euclidean")

        # Compute Correlation Sum C(r) for various r
        # Use log-spaced radii
        # Avoid zero distance
        dists = dists[dists > 1e-9]
        if len(dists) == 0:
            return 0.0

        min_r, max_r = np.min(dists), np.max(dists)
        radii = np.geomspace(min_r * 2, max_r * 0.5, 20)
        c_r = []

        for r in radii:
            count = np.sum(dists < r)
            c_r.append(count / len(dists))

        # Fit line to log-log plot
        log_r = np.log(radii)
        log_c = np.log(c_r)

        # Select linear region (middle 50%)
        n_points = len(log_r)
        start = n_points // 4
        end = 3 * n_points // 4

        slope, _ = np.polyfit(log_r[start:end], log_c[start:end], 1)
        return float(slope)

    def compute_lyapunov_divergence(
        self,
        data: np.ndarray,
        tau: int = 1,
        dim: int = 3,
        window: int = 50,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute Lyapunov Divergence Curve (Rosenstein's algorithm).

        Args:
            data: 1D time series
            tau: Time delay
            dim: Embedding dimension
            window: Theiler window

        Returns:
            Tuple of (time_axis, log_divergence, estimated_slope)
        """
        N = len(data)
        if N < window:
            return np.array([]), np.array([]), 0.0

        # 1. Phase Space Reconstruction
        M = N - (dim - 1) * tau
        if M < 1:
            return np.array([]), np.array([]), 0.0

        orbit = np.zeros((M, dim))
        for d in range(dim):
            orbit[:, d] = data[d * tau : d * tau + M]

        # 2. Find Nearest Neighbors (Approx O(M log M) with KDTree or O(M^2) distance)
        from scipy.spatial.distance import cdist

        # Using simple cdist (okay for typical swing N ~ 500)
        dists_mat = cdist(orbit, orbit, metric="euclidean")

        # Apply Theiler window (exclude diagonal band)
        for i in range(M):
            start = max(0, i - window)
            end = min(M, i + window + 1)
            dists_mat[i, start:end] = np.inf
            dists_mat[i, i] = np.inf

        nearest_neighbors = np.argmin(dists_mat, axis=1)

        # 3. Track Divergence
        # How far do they separate over time?
        max_steps = min(M, int(1.0 / self.dt * 0.5)) if self.dt > 0 else 10
        divergence = np.zeros(max_steps)
        counts = np.zeros(max_steps)

        for i in range(max_steps):
            idx1_vec = np.arange(M) + i
            idx2_vec = nearest_neighbors + i

            valid_mask = (idx1_vec < M) & (idx2_vec < M)
            if not np.any(valid_mask):
                continue

            p1 = orbit[idx1_vec[valid_mask]]
            p2 = orbit[idx2_vec[valid_mask]]

            # Distances
            dists = np.sqrt(np.sum((p1 - p2) ** 2, axis=1))

            valid_dists_mask = dists > 1e-9
            valid_dists = dists[valid_dists_mask]

            if len(valid_dists) > 0:
                divergence[i] += np.sum(np.log(valid_dists))
                counts[i] += len(valid_dists)

        counts[counts == 0] = 1.0
        avg_log_dist = divergence / counts
        time_axis = np.arange(max_steps) * self.dt

        # Estimate Slope (LLE)
        slope = 0.0
        if len(time_axis) > 1:
            # Fit linear region (e.g. first 50% or all if short)
            limit = len(time_axis) // 2 if len(time_axis) > 10 else len(time_axis)
            if limit > 1:
                slope, _ = np.polyfit(time_axis[:limit], avg_log_dist[:limit], 1)

        return time_axis, avg_log_dist, float(slope)

    def estimate_lyapunov_exponent(
        self,
        data: np.ndarray,
        tau: int = 1,
        dim: int = 3,
        window: int = 50,
    ) -> float:
        """Estimate the Largest Lyapunov Exponent (LLE) using Rosenstein's algorithm.

        Args:
            data: 1D time series array
            tau: Time delay (lag)
            dim: Embedding dimension
            window: Minimum temporal separation for nearest neighbors (Theiler window)

        Returns:
            Estimated LLE (slope of divergence curve)
        """
        _, _, slope = self.compute_lyapunov_divergence(
            data, tau=tau, dim=dim, window=window
        )
        return slope

    def compute_principal_component_analysis(
        self,
        n_components: int | None = None,
        data_type: str = "position",
    ) -> PCAResult | None:
        """Compute Principal Component Analysis (PCA) on joint data.

        Also known as "Principal Movements" when applied to kinematic data.
        Identifies the main modes of variation in the movement.

        Args:
            n_components: Number of components to retain (default: all)
            data_type: 'position', 'velocity'

        Returns:
            PCAResult object or None
        """
        if data_type == "position":
            data = self.joint_positions
        else:
            data = self.joint_velocities

        if data.shape[1] == 0 or len(data) == 0:
            return None

        # Center data
        mean = np.mean(data, axis=0)
        centered_data = data - mean

        # SVD approach is numerically stable
        # X = U * S * Vt
        # Cov = X.T * X / (N-1)
        # Eigenvectors are rows of Vt (columns of V)
        # Eigenvalues are S^2 / (N-1)

        try:
            U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
        except np.linalg.LinAlgError:
            return None

        # Eigenvalues
        n_samples = data.shape[0]
        explained_variance = (S**2) / (n_samples - 1)
        total_var = np.sum(explained_variance)
        explained_variance_ratio = (
            explained_variance / total_var if total_var > 0 else np.zeros_like(S)
        )

        # Components (Eigenvectors)
        components = Vt

        # Project data (Scores)
        # Scores = X * V = U * S
        projected_data = U * S

        if n_components is not None:
            n_components = min(n_components, len(explained_variance))
            components = components[:n_components]
            explained_variance = explained_variance[:n_components]
            explained_variance_ratio = explained_variance_ratio[:n_components]
            projected_data = projected_data[:, :n_components]

        return PCAResult(
            components=components,
            explained_variance=explained_variance,
            explained_variance_ratio=explained_variance_ratio,
            projected_data=projected_data,
            mean=mean,
        )

    def compute_principal_movements(
        self, n_modes: int = 3
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute Principal Movements (PMs) from position data.

        Wrapper around PCA specifically for position data to analyze
        primary coordination modes.

        Args:
            n_modes: Number of PMs to return

        Returns:
            (eigenvectors, scores) or None
        """
        result = self.compute_principal_component_analysis(
            n_components=n_modes, data_type="position"
        )
        if result:
            return result.components, result.projected_data
        return None

    def compute_permutation_entropy(
        self,
        data: np.ndarray,
        order: int = 3,
        delay: int = 1,
    ) -> float:
        """Compute Permutation Entropy.

        Measures complexity of a time series based on ordinal patterns.

        Args:
            data: 1D time series
            order: Order of permutation (embedding dimension)
            delay: Time delay

        Returns:
            Entropy value (bits)
        """
        N = len(data)
        M = N - (order - 1) * delay
        if M < 1:
            return 0.0

        # Create embedding matrix
        # Shape (M, order)
        # For efficiency with simple loop or stride
        # patterns = []
        # for i in range(M):
        #     pattern = data[i : i + order * delay : delay]
        #     patterns.append(tuple(np.argsort(pattern)))

        # Vectorized approach
        # Create matrix of shape (M, order)
        matrix = np.zeros((M, order), dtype=data.dtype)
        for i in range(order):
            matrix[:, i] = data[i * delay : i * delay + M]

        # Get argsort (ranks)
        ranks = np.argsort(matrix, axis=1)

        # OPTIMIZATION: Use integer packing instead of axis=0 unique for small orders
        # np.unique(axis=0) is slow because it involves row-wise comparisons/sorts.
        # Packing into 1D integers (base-order encoding) allows 1D unique, which is >10x faster.
        # Ranks are digits 0..order-1.
        # Safe for order <= 12 (12^12 < 2^63). Default order is 3.
        if order <= 12:
            packed = np.zeros(M, dtype=np.int64)
            # Base-order encoding: rank[0]*order^0 + rank[1]*order^1 ...
            # Using order^i as weights ensures uniqueness for permutations
            # (actually base >= order is required, so base=order is sufficient)
            multiplier = 1
            for i in range(order):
                packed += ranks[:, i] * multiplier
                multiplier *= order

            _, counts = np.unique(packed, return_counts=True)
        else:
            # Fallback for very large orders (unlikely in PE context)
            _, counts = np.unique(ranks, axis=0, return_counts=True)

        # Probabilities
        probs = counts / M
        probs = probs[probs > 0]

        # Shannon Entropy
        pe = -np.sum(probs * np.log2(probs))

        # Normalize by log2(factorial(order))?
        # Standard PE is usually normalized.
        # factorials = [1, 1, 2, 6, 24, 120, 720...]
        # max_entropy = np.log2(np.math.factorial(order))
        # return pe / max_entropy

        # We return raw bits
        return float(pe)

    def compute_joint_stiffness(
        self,
        joint_idx: int,
        window: slice | None = None,
    ) -> JointStiffnessMetrics | None:
        """Compute Quasi-Stiffness from Moment-Angle relationship.

        Stiffness is estimated as the slope of the linear regression of
        Torque vs Angle. Also computes hysteresis (loop area).

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

        # Linear Regression (Torque = k * Angle + c)
        # k is stiffness (Nm/rad)
        # Use np.polyfit(deg=1)
        slope, intercept = np.polyfit(angles, torques, 1)

        # R-squared calculation
        predicted = slope * angles + intercept
        residuals = torques - predicted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((torques - np.mean(torques)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Hysteresis Area (Integration using Green's theorem or simple trapezoid)
        # Area = Integral(Torque dAngle)
        # Use simple trapezoidal rule on the polygon
        # Ideally, sort by time (which is implicit)
        # Area = 0.5 * sum((x_{i+1} + x_i) * (y_{i+1} - y_i)) - this is for y dx
        # For cyclic loop, this works.
        # We use numpy trapz
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

        # OPTIMIZATION: Use convolution for O(N) complexity instead of O(N*W)
        # Significantly faster and memory efficient compared to sliding_window_view
        kernel = np.ones(window_size)
        n = window_size

        s_x = np.convolve(angles, kernel, mode="valid")
        s_y = np.convolve(torques, kernel, mode="valid")
        s_xy = np.convolve(angles * torques, kernel, mode="valid")
        s_xx = np.convolve(angles**2, kernel, mode="valid")
        s_yy = np.convolve(torques**2, kernel, mode="valid")

        cov = s_xy - (s_x * s_y) / n
        var_x = s_xx - (s_x**2) / n
        var_y = s_yy - (s_y**2) / n

        # Clamp variances to be non-negative
        var_x = np.maximum(var_x, 0)
        var_y = np.maximum(var_y, 0)

        # Calculate Slope and R2
        # Use np.divide and where for safe division
        slope = np.zeros_like(cov)
        valid_var_x = var_x > 1e-9
        np.divide(cov, var_x, out=slope, where=valid_var_x)

        r2 = np.zeros_like(cov)
        valid_both = valid_var_x & (var_y > 1e-9)
        np.divide(cov**2, var_x * var_y, out=r2, where=valid_both)

        # Time points (center of window)
        # Indices in original array: window_size//2 to N - window_size + window_size//2
        valid_indices = np.arange(len(slope)) + window_size // 2
        time_points = self.times[valid_indices]

        return time_points, slope, r2

    def compute_fractal_dimension(
        self,
        data: np.ndarray,
        k_max: int = 10,
    ) -> float:
        """Compute Fractal Dimension using Higuchi's method.

        Measures the complexity/roughness of the time series.

        Args:
            data: 1D time series
            k_max: Maximum interval time (k)

        Returns:
            Fractal dimension (HFD) approx between 1.0 and 2.0
        """
        N = len(data)
        if N < k_max + 1:
            return 1.0

        L_k = []
        x_k = []

        for k in range(1, k_max + 1):
            L_m_k = 0.0
            # OPTIMIZATION: Compute all differences for lag k at once
            # abs_diffs[i] = |x(i+k) - x(i)|. This avoids creating indices/diffs in the inner loop.
            abs_diffs = np.abs(data[k:] - data[:-k])

            for m in range(k):
                # Standard Higuchi:
                # L_m(k) = (Sum |x(i+k)-x(i)| ) * (N-1) / ( floor((N-m-1)/k) * k )

                # Extract differences for this m
                # Indices in original data: m, m+k, m+2k...
                # Diffs are |x(m+k)-x(m)|, |x(m+2k)-x(m+k)|...
                # These correspond to abs_diffs indices: m, m+k, ...
                m_diffs = abs_diffs[m::k]
                n_intervals = len(m_diffs)

                if n_intervals > 0:
                    L_m_k += np.sum(m_diffs) * (N - 1) / (n_intervals * k)

            # Average over m (divide by k) AND apply Higuchi scaling (divide by k)
            # So divide by k^2
            L_k.append(L_m_k / (k * k))
            x_k.append(np.log(1.0 / k))

        # Slope of log(L(k)) vs log(1/k) is the dimension D
        y_val = np.log(L_k)
        slope, _ = np.polyfit(x_k, y_val, 1)

        return float(slope)

    def compute_sample_entropy(
        self,
        data: np.ndarray,
        m: int = 2,
        r: float = 0.2,
    ) -> float:
        """Compute Sample Entropy (SampEn).

        Measures the regularity of a time series. Lower values = more regular.

        Args:
            data: 1D time series
            m: Template length (embedding dimension)
            r: Tolerance (typically 0.2 * std)

        Returns:
            Sample Entropy value
        """
        N = len(data)
        if N < m + 1:
            return 0.0

        # Normalize r by standard deviation if it's relative
        # Typically r is passed as ratio, so we multiply
        tolerance = r * np.std(data)

        def count_matches(template_len: int) -> int:
            # Total possible vectors
            n_vectors = N - template_len

            # Construct matrix X of shape (n_vectors, template_len)
            X = np.zeros((n_vectors, template_len))
            for i in range(template_len):
                X[:, i] = data[i : i + n_vectors]

            # OPTIMIZATION: Use cKDTree for O(N log N) neighbor search instead of O(N^2) pdist
            # Chebychev distance corresponds to Minkowski p=inf
            tree = cKDTree(X)

            # count_neighbors(other, r) counts pairs (i, j) with dist <= r
            # Querying against itself returns count of pairs in X
            # This includes self-matches (dist=0) and counts (i, j) and (j, i) separately
            count = tree.count_neighbors(tree, r=tolerance, p=np.inf)

            # Subtract self-matches (n_vectors) and divide by 2 to get unique pairs i < j
            B = (count - n_vectors) // 2

            return int(B)

        # Count matches for m
        A = count_matches(m)
        # Count matches for m+1
        B = count_matches(m + 1)

        if A == 0 or B == 0:
            # Entropy undefined or infinite
            return 0.0  # Or return -log(2/((N-m-1)*(N-m)))?

        # SampEn = -log(B/A)
        return float(-np.log(B / A))

    def compute_multiscale_entropy(
        self,
        data: np.ndarray,
        max_scale: int = 10,
        m: int = 2,
        r: float = 0.15,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Multiscale Entropy (MSE).

        Calculates Sample Entropy at multiple time scales (coarse-graining).
        MSE accounts for complexity at different resolutions.

        Args:
            data: 1D time series
            max_scale: Maximum scale factor (tau)
            m: Template length
            r: Tolerance (ratio of std)

        Returns:
            Tuple of (scales, entropy_values)
        """
        mse_values = []
        scales = np.arange(1, max_scale + 1)

        for scale in scales:
            # Coarse-graining: average non-overlapping windows of length 'scale'
            if scale == 1:
                scaled_data = data
            else:
                n_windows = len(data) // scale
                if n_windows < m + 1:
                    mse_values.append(0.0)
                    continue
                # Reshape and mean
                # Truncate to multiple of scale
                truncated = data[: n_windows * scale]
                reshaped = truncated.reshape(n_windows, scale)
                scaled_data = np.mean(reshaped, axis=1)

            # Compute SampEn
            # Standard MSE uses fixed tolerance based on original SD
            # r_val = r * std_original
            # compute_sample_entropy uses r_ratio * std_current
            # So r_ratio = (r * std_original) / std_current

            std_current = np.std(scaled_data)
            std_original = np.std(data)

            if std_current < 1e-9:
                mse = 0.0
            else:
                r_ratio = (r * std_original) / std_current
                mse = self.compute_sample_entropy(scaled_data, m=m, r=r_ratio)

            mse_values.append(mse)

        return scales, np.array(mse_values)

    def compute_jerk_metrics(self, joint_idx: int) -> JerkMetrics | None:
        """Compute jerk metrics for a joint.

        Jerk is the rate of change of acceleration. High jerk implies less smooth
        movement.

        Args:
            joint_idx: Joint index

        Returns:
            JerkMetrics object or None
        """
        # We need acceleration. If not available, compute from velocity.
        if (
            hasattr(self, "joint_accelerations")
            and self.joint_accelerations is not None
        ):
            if joint_idx >= self.joint_accelerations.shape[1]:
                return None
            accel = self.joint_accelerations[:, joint_idx]
        elif (
            hasattr(self, "joint_velocities")
            and self.joint_velocities is not None
            and self.dt > 0
        ):
            if joint_idx >= self.joint_velocities.shape[1]:
                return None
            vel = self.joint_velocities[:, joint_idx]
            accel = np.gradient(vel, self.dt)
        else:
            return None

        fs = 1.0 / self.dt if self.dt > 0 else 0.0
        if fs <= 0:
            return None

        try:
            from shared.python import signal_processing

            jerk = signal_processing.compute_jerk(accel, fs)
        except ImportError:
            # Fallback
            jerk = np.gradient(accel, self.dt)

        peak_jerk = float(np.max(np.abs(jerk)))
        rms_jerk = float(np.sqrt(np.mean(jerk**2)))

        # Dimensionless Jerk (Log dimensionless jerk)
        # LDJ = - ln( integral(j^2 dt) * D^5 / A^2 )
        # Normalized by movement duration D and amplitude A (peak-to-peak pos or vel range)
        # Here we use a simpler dimensionless form: (RMS Jerk * Duration^2) / Peak Velocity
        # Or standard: Integral(j^2) * D^5 / L^2
        # Let's return RMS normalized by peak accel?
        # A common simple metric is Peak Jerk / Peak Accel
        peak_acc = float(np.max(np.abs(accel)))
        dim_jerk = peak_jerk / peak_acc if peak_acc > 1e-6 else 0.0

        return JerkMetrics(
            peak_jerk=peak_jerk,
            rms_jerk=rms_jerk,
            dimensionless_jerk=dim_jerk,
        )

    def compute_lag_matrix(
        self,
        data_type: str = "velocity",
        max_lag: float = 0.5,
    ) -> tuple[np.ndarray, list[str]]:
        """Compute time lag matrix between all pairs of joints.

        Args:
            data_type: 'position', 'velocity', 'torque'
            max_lag: Maximum lag to search (seconds)

        Returns:
            Tuple of (lag_matrix, labels).
            Matrix[i, j] > 0 means i leads j (j lags i).
            Actually compute_time_shift(x, y) returns tau where y(t) ~ x(t-tau).
            If tau > 0, y is delayed (lags) relative to x.
            So Matrix[i, j] = compute_time_shift(J_i, J_j).
            Positive value => J_j lags J_i (J_i leads).
        """
        if data_type == "position":
            data = self.joint_positions
        elif data_type == "torque":
            data = self.joint_torques
        else:
            data = self.joint_velocities

        n_joints = data.shape[1]
        if n_joints == 0:
            return np.array([]), []

        fs = 1.0 / self.dt if self.dt > 0 else 0.0
        if fs <= 0:
            return np.zeros((n_joints, n_joints)), []

        try:
            from shared.python import signal_processing
        except ImportError:
            return np.zeros((n_joints, n_joints)), []

        lag_matrix = np.zeros((n_joints, n_joints))

        # Parallelize? No, n_joints is small (10-30). O(N^2) loop is fine.
        # Matrix is antisymmetric: Lag(i, j) = -Lag(j, i)
        for i in range(n_joints):
            for j in range(i + 1, n_joints):
                lag = signal_processing.compute_time_shift(
                    data[:, i], data[:, j], fs, max_lag=max_lag
                )
                lag_matrix[i, j] = lag
                lag_matrix[j, i] = -lag

        labels = [f"J{i}" for i in range(n_joints)]
        return lag_matrix, labels

    def compute_poincare_map(
        self,
        dimensions: list[tuple[str, int]],
        section_condition: tuple[str, int, float],
        direction: str = "both",
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute Poincaré Map (Poincaré Section).

        Identify intersection points of the system trajectory with a defined
        lower-dimensional subspace (hyperplane).

        Args:
            dimensions: List of 3 (data_type, index) tuples for coordinates of intersection points.
                        data_type can be 'position', 'velocity', 'acceleration', 'torque'.
            section_condition: Tuple of (data_type, index, value) defining the section plane.
            direction: Crossing direction ('positive', 'negative', 'both').

        Returns:
            Tuple of (points, times) or None if inputs invalid.
            points is (N_crossings, len(dimensions))
            times is (N_crossings,)
        """

        # Helper to get data array
        def get_data(dtype: str, idx: int) -> np.ndarray | None:
            if dtype == "position":
                d = self.joint_positions
            elif dtype == "velocity":
                d = self.joint_velocities
            elif dtype == "acceleration":
                d = (
                    self.joint_accelerations
                    if self.joint_accelerations is not None
                    else np.zeros_like(self.joint_velocities)
                )
            elif dtype == "torque":
                d = self.joint_torques
            else:
                return None
            if d.ndim > 1 and idx < d.shape[1]:
                return d[:, idx]
            return None

        # Get condition variable
        cond_type, cond_idx, cond_val = section_condition
        cond_data = get_data(cond_type, cond_idx)

        if cond_data is None or len(cond_data) < 2:
            return None

        # Find crossings
        diff = cond_data - cond_val
        crossings = []

        for i in range(len(diff) - 1):
            if diff[i] * diff[i + 1] <= 0:  # Crossing detected
                # Check direction
                if diff[i] < diff[i + 1] and direction in ["positive", "both"]:
                    crossings.append(i)
                elif diff[i] > diff[i + 1] and direction in ["negative", "both"]:
                    crossings.append(i)

        if not crossings:
            return np.array([]), np.array([])

        # Interpolate state at crossings
        points = []
        point_times = []

        for i in crossings:
            # Linear interpolation factor
            # y = y0 + (y1-y0) * alpha
            # 0 = d0 + (d1-d0) * alpha => alpha = -d0 / (d1-d0)
            denom = diff[i + 1] - diff[i]
            if abs(denom) < 1e-9:
                alpha = 0.5
            else:
                alpha = -diff[i] / denom

            t_cross = self.times[i] + alpha * (self.times[i + 1] - self.times[i])
            point_times.append(t_cross)

            pt_coords = []
            for dim_type, dim_idx in dimensions:
                data = get_data(dim_type, dim_idx)
                if data is None:
                    pt_coords.append(0.0)
                else:
                    val = data[i] + alpha * (data[i + 1] - data[i])
                    # Keep native units here; conversion happens in plot
                    pt_coords.append(val)
            points.append(pt_coords)

        return np.array(points), np.array(point_times)

    def export_statistics_csv(
        self,
        filename: str,
        report: dict | None = None,
    ) -> None:
        """Export statistics to CSV file.

        Args:
            filename: Output filename
            report: Statistics report (if None, generates new one)
        """
        if report is None:
            report = self.generate_comprehensive_report()

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(["Golf Swing Statistical Analysis"])
            writer.writerow([])

            # Overall metrics
            writer.writerow(["Overall Metrics"])
            writer.writerow(["Metric", "Value", "Unit"])
            writer.writerow(["Duration", report["duration"], "s"])
            writer.writerow(["Sample Rate", report["sample_rate"], "Hz"])
            writer.writerow(["Samples", report["num_samples"], ""])
            writer.writerow([])

            # Stability Metrics (New)
            if "stability_metrics" in report:
                writer.writerow(["Stability Metrics"])
                writer.writerow(["Metric", "Value"])
                sm = report["stability_metrics"]
                for key, val in sm.items():
                    writer.writerow([key.replace("_", " ").title(), f"{val:.4f}"])
                writer.writerow([])

            # Club head speed
            if "club_head_speed" in report:
                writer.writerow(["Club Head Speed"])
                writer.writerow(["Metric", "Value", "Unit"])
                chs = report["club_head_speed"]
                writer.writerow(["Peak Speed", chs["peak_value"], "mph"])
                writer.writerow(["Peak Time", chs["peak_time"], "s"])
                writer.writerow([])

            # Tempo
            if "tempo" in report:
                writer.writerow(["Swing Tempo"])
                writer.writerow(["Metric", "Value", "Unit"])
                writer.writerow(
                    ["Backswing Duration", report["tempo"]["backswing_duration"], "s"],
                )
                writer.writerow(
                    ["Downswing Duration", report["tempo"]["downswing_duration"], "s"],
                )
                writer.writerow(["Tempo Ratio", report["tempo"]["ratio"], ""])
                writer.writerow([])

            # Phases
            if "phases" in report:
                writer.writerow(["Swing Phases"])
                writer.writerow(["Phase", "Start (s)", "End (s)", "Duration (s)"])
                for phase in report["phases"]:
                    writer.writerow(
                        [
                            phase["name"],
                            f"{phase['start_time']:.3f}",
                            f"{phase['end_time']:.3f}",
                            f"{phase['duration']:.3f}",
                        ],
                    )
                writer.writerow([])

            # GRF Metrics
            if "grf_metrics" in report:
                writer.writerow(["GRF & CoP Metrics"])
                writer.writerow(["Metric", "Value"])
                grf = report["grf_metrics"]
                for key, val in grf.items():
                    if val is not None:
                        writer.writerow([key.replace("_", " ").title(), f"{val:.4f}"])
                writer.writerow([])

            # Joint statistics
            writer.writerow(["Joint Statistics"])
            writer.writerow(
                [
                    "Joint",
                    "ROM (deg)",
                    "Min Angle",
                    "Max Angle",
                    "Max Velocity (deg/s)",
                    "Max Torque (Nm)",
                ],
            )

            for joint_name, joint_data in report["joints"].items():
                rom_data = joint_data["range_of_motion"]
                vel_stats = joint_data.get("velocity_stats", {})
                torque_stats = joint_data.get("torque_stats", {})

                writer.writerow(
                    [
                        joint_name,
                        f"{rom_data['rom_deg']:.1f}",
                        f"{rom_data['min_deg']:.1f}",
                        f"{rom_data['max_deg']:.1f}",
                        f"{vel_stats.get('max', 0.0):.1f}",
                        f"{torque_stats.get('max', 0.0):.1f}",
                    ],
                )
