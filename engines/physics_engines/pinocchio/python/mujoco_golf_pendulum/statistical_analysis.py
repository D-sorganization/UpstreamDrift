"""Statistical analysis module for golf swing biomechanics.

Provides comprehensive statistical analysis including:
- Peak detection
- Summary statistics
- Swing quality metrics
- Phase-specific analysis
"""

import csv
from dataclasses import dataclass
from pathlib import Path
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
    ) -> None:
        """Initialize analyzer with recorded data.

        Args:
            times: Time array (N,)
            joint_positions: Joint positions (N, nq)
            joint_velocities: Joint velocities (N, nv)
            joint_torques: Joint torques (N, nu)
            club_head_speed: Club head speed (N,) [optional]
            club_head_position: Club head 3D position (N, 3) [optional]
        """
        self.times = times
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.joint_torques = joint_torques
        self.club_head_speed = club_head_speed
        self.club_head_position = club_head_position

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

        return SummaryStatistics(
            mean=float(np.mean(data)),
            median=float(np.median(data)),
            std=float(np.std(data)),
            min=float(np.min(data)),
            max=float(np.max(data)),
            range=float(np.ptp(data)),
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

        return shoulder_rotation - hip_rotation

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
                    "statistics": self.compute_summary_stats(
                        self.club_head_speed,
                    ).__dict__,
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

        # Joint statistics
        report["joints"] = {}
        for i in range(self.joint_positions.shape[1]):
            min_angle, max_angle, rom = self.compute_range_of_motion(i)
            velocities = (
                self.joint_velocities[:, i]
                if i < self.joint_velocities.shape[1]
                else None
            )

            joint_stats = {
                "range_of_motion": {
                    "min_deg": min_angle,
                    "max_deg": max_angle,
                    "rom_deg": rom,
                },
                "position_stats": self.compute_summary_stats(
                    np.rad2deg(self.joint_positions[:, i]),
                ).__dict__,
            }

            if velocities is not None:
                joint_stats["velocity_stats"] = self.compute_summary_stats(
                    np.rad2deg(velocities),
                ).__dict__

            if i < self.joint_torques.shape[1]:
                joint_stats["torque_stats"] = self.compute_summary_stats(
                    self.joint_torques[:, i],
                ).__dict__

            report["joints"][f"joint_{i}"] = joint_stats

        return report

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

        with Path(filename).open("w", newline="") as f:
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
