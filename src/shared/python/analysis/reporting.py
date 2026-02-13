"""Reporting and export functionality for statistical analysis.

Includes comprehensive report generation, CSV export,
frequency analysis, smoothness metrics, jerk metrics,
and swing profile computation.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from typing import Any

import numpy as np

from src.shared.python.analysis.dataclasses import (
    JerkMetrics,
    SwingProfileMetrics,
)


class ReportingMixin:
    """Mixin for report generation, export, and high-level analysis.

    Expects the following attributes/methods to be available on the instance:
    - times: np.ndarray
    - joint_positions: np.ndarray
    - joint_velocities: np.ndarray
    - joint_torques: np.ndarray
    - club_head_speed: np.ndarray | None
    - joint_accelerations: np.ndarray | None
    - dt: float
    - duration: float
    - find_club_head_speed_peak()
    - compute_summary_stats()
    - compute_tempo()
    - detect_swing_phases()
    - compute_grf_metrics()
    - compute_angular_momentum_metrics()
    - compute_stability_metrics()
    """

    times: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    club_head_speed: np.ndarray | None
    joint_accelerations: np.ndarray | None
    dt: float
    duration: float

    def _report_club_head_speed(self) -> dict[str, Any] | None:
        if self.club_head_speed is None:
            return None
        peak_speed = self.find_club_head_speed_peak()  # type: ignore[attr-defined]
        if not peak_speed:
            return None
        return {
            "peak_value": peak_speed.value,
            "peak_time": peak_speed.time,
            "statistics": asdict(
                self.compute_summary_stats(  # type: ignore[attr-defined]
                    self.club_head_speed,
                )
            ),
        }

    def _report_tempo(self) -> dict[str, Any] | None:
        tempo_result = self.compute_tempo()  # type: ignore[attr-defined]
        if not tempo_result:
            return None
        return {
            "backswing_duration": tempo_result[0],
            "downswing_duration": tempo_result[1],
            "ratio": tempo_result[2],
        }

    def _report_phases(self) -> list[dict[str, Any]]:
        phases = self.detect_swing_phases()  # type: ignore[attr-defined]
        return [
            {
                "name": p.name,
                "start_time": p.start_time,
                "end_time": p.end_time,
                "duration": p.duration,
            }
            for p in phases
        ]

    def _report_joint_stats(self, joint_idx: int) -> dict[str, Any]:
        angles_deg = np.rad2deg(self.joint_positions[:, joint_idx])
        position_stats = self.compute_summary_stats(angles_deg)  # type: ignore[attr-defined]

        velocities = (
            self.joint_velocities[:, joint_idx]
            if joint_idx < self.joint_velocities.shape[1]
            else None
        )

        joint_stats: dict[str, Any] = {
            "range_of_motion": {
                "min_deg": position_stats.min,
                "max_deg": position_stats.max,
                "rom_deg": position_stats.range,
            },
            "position_stats": asdict(position_stats),
        }

        if velocities is not None:
            joint_stats["velocity_stats"] = asdict(
                self.compute_summary_stats(  # type: ignore[attr-defined]
                    np.rad2deg(velocities),
                )
            )

        if joint_idx < self.joint_torques.shape[1]:
            joint_stats["torque_stats"] = asdict(
                self.compute_summary_stats(  # type: ignore[attr-defined]
                    self.joint_torques[:, joint_idx],
                )
            )

        return joint_stats

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

        chs = self._report_club_head_speed()
        if chs:
            report["club_head_speed"] = chs

        tempo = self._report_tempo()
        if tempo:
            report["tempo"] = tempo

        report["phases"] = self._report_phases()

        grf_metrics = self.compute_grf_metrics()  # type: ignore[attr-defined]
        if grf_metrics:
            report["grf_metrics"] = asdict(grf_metrics)

        am_metrics = self.compute_angular_momentum_metrics()  # type: ignore[attr-defined]
        if am_metrics:
            report["angular_momentum_metrics"] = asdict(am_metrics)

        stability_metrics = self.compute_stability_metrics()  # type: ignore[attr-defined]
        if stability_metrics:
            report["stability_metrics"] = asdict(stability_metrics)

        report["joints"] = {}
        for i in range(self.joint_positions.shape[1]):
            report["joints"][f"joint_{i}"] = self._report_joint_stats(i)

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
            from shared.python.signal_toolkit import (
                signal_processing,  # type: ignore[attr-defined]
            )

            result = signal_processing.compute_psd(data, fs, window=window)
            return result  # type: ignore[no-any-return]
        except ImportError:
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
            from shared.python.signal_toolkit import (
                signal_processing,  # type: ignore[attr-defined]
            )

            result = signal_processing.compute_spectral_arc_length(data, fs)
            return result  # type: ignore[no-any-return]
        except ImportError:
            return 0.0

    def compute_swing_profile(self) -> SwingProfileMetrics | None:
        """Compute Swing Profile scores (0-100) for radar chart visualization.

        Returns:
            SwingProfileMetrics object or None if insufficient data
        """
        # 1. Speed Score
        speed_score = 0.0
        if self.club_head_speed is not None:
            peak_speed = float(np.max(self.club_head_speed))
            peak_speed_mph = peak_speed * 2.23694
            speed_score = float(min(100.0, (peak_speed_mph / 120.0) * 100.0))

        # 2. Sequence Score
        sequence_score = 0.0
        if self.joint_velocities.shape[1] >= 3:
            peaks = []
            for i in range(3):
                idx = np.argmax(np.abs(self.joint_velocities[:, i]))
                peaks.append(idx)
            if peaks == sorted(peaks, key=int):
                sequence_score = 100.0
            else:
                sequence_score = 50.0
        else:
            sequence_score = 0.0

        # 3. Stability Score
        stability_score = 0.0
        stab_metrics = self.compute_stability_metrics()  # type: ignore[attr-defined]
        if stab_metrics:
            angle = stab_metrics.mean_inclination_angle
            stability_score = float(max(0.0, min(100.0, 100.0 - (angle - 5.0) * 4.0)))
        else:
            stability_score = 0.0

        # 4. Efficiency Score
        efficiency_score = 0.0
        if (
            self.club_head_speed is not None
            and self.joint_torques.shape[1] > 0
            and self.joint_velocities.shape[1] > 0
        ):
            ke = 0.5 * 1.0 * (self.club_head_speed**2)
            n_joints = self.joint_torques.shape[1]
            n_samples = min(
                self.joint_torques.shape[0],
                self.joint_velocities.shape[0],
                len(self.times),
            )

            if n_samples >= 2:
                torques = self.joint_torques[:n_samples, :n_joints]
                velocities = self.joint_velocities[:n_samples, :n_joints]
                power = torques * velocities
                abs_power = np.abs(power)

                if hasattr(np, "trapezoid"):
                    total_work = float(
                        np.trapezoid(abs_power, dx=self.dt, axis=0).sum()
                    )
                else:
                    trapz_func = getattr(np, "trapz")  # noqa: B009
                    total_work = float(trapz_func(abs_power, dx=self.dt, axis=0).sum())
            else:
                total_work = 0.0

            if total_work > 0:
                peak_ke = float(np.max(ke))
                eff = peak_ke / total_work
                efficiency_score = float(min(100.0, eff * 200.0))
        else:
            efficiency_score = 0.0

        # 5. Power Score
        power_score = 0.0
        if self.joint_torques.shape[1] > 0 and self.joint_velocities.shape[1] > 0:
            n_joints = min(self.joint_torques.shape[1], self.joint_velocities.shape[1])
            total_power = np.sum(
                self.joint_torques[:, :n_joints] * self.joint_velocities[:, :n_joints],
                axis=1,
            )
            peak_power = float(np.max(total_power))
            power_score = float(min(100.0, (peak_power / 3000.0) * 100.0))

        return SwingProfileMetrics(
            speed_score=float(speed_score),
            sequence_score=float(sequence_score),
            stability_score=float(stability_score),
            efficiency_score=float(efficiency_score),
            power_score=float(power_score),
        )

    def compute_jerk_metrics(self, joint_idx: int) -> JerkMetrics | None:
        """Compute jerk metrics for a joint.

        Args:
            joint_idx: Joint index

        Returns:
            JerkMetrics object or None
        """
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
            from shared.python.signal_toolkit import (
                signal_processing,  # type: ignore[attr-defined]
            )

            jerk = signal_processing.compute_jerk(accel, fs)
        except ImportError:
            jerk = np.gradient(accel, self.dt)

        peak_jerk = float(np.max(np.abs(jerk)))
        rms_jerk = float(np.sqrt(np.mean(jerk**2)))

        if (
            hasattr(self, "joint_velocities")
            and self.joint_velocities is not None
            and joint_idx < self.joint_velocities.shape[1]
        ):
            vel = self.joint_velocities[:, joint_idx]
            peak_vel = float(np.max(np.abs(vel)))
        else:
            peak_vel = 0.0

        if peak_vel > 1e-6 and self.duration > 0:
            dim_jerk = (rms_jerk * (self.duration**2)) / peak_vel
        else:
            dim_jerk = 0.0

        return JerkMetrics(
            peak_jerk=peak_jerk,
            rms_jerk=rms_jerk,
            dimensionless_jerk=dim_jerk,
        )

    def _write_csv_overall_metrics(self, writer: csv.writer, report: dict) -> None:
        writer.writerow(["Golf Swing Statistical Analysis"])
        writer.writerow([])
        writer.writerow(["Overall Metrics"])
        writer.writerow(["Metric", "Value", "Unit"])
        writer.writerow(["Duration", report["duration"], "s"])
        writer.writerow(["Sample Rate", report["sample_rate"], "Hz"])
        writer.writerow(["Samples", report["num_samples"], ""])
        writer.writerow([])

    def _write_csv_stability_metrics(self, writer: csv.writer, report: dict) -> None:
        if "stability_metrics" not in report:
            return
        writer.writerow(["Stability Metrics"])
        writer.writerow(["Metric", "Value"])
        for key, val in report["stability_metrics"].items():
            writer.writerow([key.replace("_", " ").title(), f"{val:.4f}"])
        writer.writerow([])

    def _write_csv_club_head_speed(self, writer: csv.writer, report: dict) -> None:
        if "club_head_speed" not in report:
            return
        writer.writerow(["Club Head Speed"])
        writer.writerow(["Metric", "Value", "Unit"])
        chs = report["club_head_speed"]
        writer.writerow(["Peak Speed", chs["peak_value"], "mph"])
        writer.writerow(["Peak Time", chs["peak_time"], "s"])
        writer.writerow([])

    def _write_csv_tempo(self, writer: csv.writer, report: dict) -> None:
        if "tempo" not in report:
            return
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

    def _write_csv_phases(self, writer: csv.writer, report: dict) -> None:
        if "phases" not in report:
            return
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

    def _write_csv_grf_metrics(self, writer: csv.writer, report: dict) -> None:
        if "grf_metrics" not in report:
            return
        writer.writerow(["GRF & CoP Metrics"])
        writer.writerow(["Metric", "Value"])
        for key, val in report["grf_metrics"].items():
            if val is not None:
                writer.writerow([key.replace("_", " ").title(), f"{val:.4f}"])
        writer.writerow([])

    def _write_csv_joint_statistics(self, writer: csv.writer, report: dict) -> None:
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
            self._write_csv_overall_metrics(writer, report)
            self._write_csv_stability_metrics(writer, report)
            self._write_csv_club_head_speed(writer, report)
            self._write_csv_tempo(writer, report)
            self._write_csv_phases(writer, report)
            self._write_csv_grf_metrics(writer, report)
            self._write_csv_joint_statistics(writer, report)
