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
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.signal import find_peaks, savgol_filter


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
    in_phase_pct: float       # Both segments rotating same direction
    anti_phase_pct: float     # Segments rotating opposite directions
    proximal_leading_pct: float # Proximal segment dominant
    distal_leading_pct: float   # Distal segment dominant

    # Mean coupling angle (if meaningful)
    mean_coupling_angle: float
    coordination_variability: float  # Std dev of coupling angle


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

        self.dt = float(np.mean(np.diff(times))) if len(times) > 1 else 0.0
        self.duration = times[-1] - times[0] if len(times) > 1 else 0.0

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
        cop_diff = np.diff(self.cop_position, axis=0)
        path_length = np.sum(np.linalg.norm(cop_diff, axis=1))

        # CoP Velocity
        cop_vel = cop_diff / self.dt
        max_vel = np.max(np.linalg.norm(cop_vel, axis=1))

        # CoP Range
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
                shear = np.linalg.norm(self.ground_forces[:, :2], axis=1)
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

        mag = np.linalg.norm(self.angular_momentum, axis=1)

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
            variability=variability
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

        dist = np.linalg.norm(cop_xy - com_xy, axis=1)

        # Inclination Angle (Angle between vertical and CoP-CoM vector)
        # Vector P = CoM - CoP
        # If CoP is 2D, assume Z=0
        if self.cop_position.shape[1] == 2:
            cop_z = np.zeros(len(self.cop_position))
        else:
            cop_z = self.cop_position[:, 2]

        vec = self.com_position - np.column_stack((cop_xy, cop_z))

        # Angle with vertical (Z-axis [0, 0, 1])
        # dot(v, k) = |v| * |k| * cos(theta)
        # theta = arccos( v_z / |v| )

        vec_norm = np.linalg.norm(vec, axis=1)
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
            mean_inclination_angle=float(np.mean(angles_deg))
        )

    def compute_coordination_metrics(
        self,
        joint_idx_1: int,
        joint_idx_2: int
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
        R = np.sqrt(np.sum(np.cos(angles_rad))**2 + np.sum(np.sin(angles_rad))**2) / total
        mean_angle_rad = np.arctan2(np.sum(np.sin(angles_rad)), np.sum(np.cos(angles_rad)))
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
            coordination_variability=float(circ_std_deg)
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

        return gamma_deg

    def compute_work_metrics(self, joint_idx: int) -> dict[str, float] | None:
        """Compute mechanical work metrics for a joint.

        Work is calculated as the time integral of power (torque * angular velocity).

        Args:
            joint_idx: Index of the joint

        Returns:
            Dictionary with 'positive_work', 'negative_work', 'net_work' (Joules)
            or None if data unavailable.
        """
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
            positive_work = np.trapz(pos_power, dx=dt)
            negative_work = np.trapz(neg_power, dx=dt)

        net_work = positive_work + negative_work

        return {
            "positive_work": float(positive_work),
            "negative_work": float(negative_work),
            "net_work": float(net_work),
        }

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
        d_pos = np.diff(pos)
        d_vel = np.diff(vel)

        dist = np.sqrt(d_pos**2 + d_vel**2)
        return float(np.sum(dist))

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
