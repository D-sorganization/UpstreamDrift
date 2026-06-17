# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Reusable analysis helpers for optimization results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .trajectory import OptimizationResult

JOINT_LABELS: tuple[str, ...] = ("Ankle", "Knee", "Hip")
HIGH_PEAK_TORQUE_NM = 500.0
HIGH_COM_RANGE_CM = 15.0


@dataclass(frozen=True)
class JointTorqueStatistics:
    """Summary statistics for one joint torque trajectory."""

    joint: str
    mean_nm: float
    std_nm: float
    min_nm: float
    max_nm: float
    peak_abs_nm: float
    rms_nm: float


class ResultAnalyzer:
    """Compute professional summary metrics for an optimization result.

    Preconditions:
        ``result.torques`` is a 2D array with one column per joint.
        ``result.com`` is a 2D array with horizontal COM in column 0.
    """

    def __init__(self, result: OptimizationResult) -> None:
        if result.torques.ndim != 2:
            raise ValueError("result.torques must be a 2D array")
        if result.com.ndim != 2 or result.com.shape[1] < 1:
            raise ValueError("result.com must be a 2D array with at least one column")
        self.result = result

    def peak_torques(self) -> dict[str, float]:
        """Return peak absolute torque per joint in N*m."""
        peaks = np.max(np.abs(self.result.torques), axis=0)
        return {self._joint_label(idx): float(value) for idx, value in enumerate(peaks)}

    def torque_statistics(self) -> list[JointTorqueStatistics]:
        """Return mean, standard deviation, range, peak, and RMS per joint."""
        stats: list[JointTorqueStatistics] = []
        for idx in range(self.result.torques.shape[1]):
            col = self.result.torques[:, idx]
            stats.append(
                JointTorqueStatistics(
                    joint=self._joint_label(idx),
                    mean_nm=float(np.mean(col)),
                    std_nm=float(np.std(col)),
                    min_nm=float(np.min(col)),
                    max_nm=float(np.max(col)),
                    peak_abs_nm=float(np.max(np.abs(col))),
                    rms_nm=float(np.sqrt(np.mean(col**2))),
                )
            )
        return stats

    def balance_assessment(self) -> str:
        """Return a qualitative assessment of horizontal COM stability."""
        horizontal_range_cm = float(self.result.com_horizontal_range_cm)
        if horizontal_range_cm <= 5.0:
            return "Excellent - COM very stable"
        if horizontal_range_cm <= HIGH_COM_RANGE_CM:
            return "Good - COM well controlled"
        return "Needs review - excessive COM movement"

    def recommendations(self) -> list[str]:
        """Return result-driven recommendations for the exported report."""
        recommendations: list[str] = []
        if not self.result.success:
            recommendations.append("Review optimization settings; the solver did not converge.")
        if self.result.n_joint_limit_violations > 0:
            recommendations.append(
                "Review joint limits; the trajectory exceeded configured bounds."
            )
        if self.result.com_horizontal_range_cm > HIGH_COM_RANGE_CM:
            recommendations.append(
                "Consider increasing duration or adjusting stance to reduce COM travel."
            )

        peak_torques = self.peak_torques()
        high_torque_joints = [
            joint for joint, peak in peak_torques.items() if peak > HIGH_PEAK_TORQUE_NM
        ]
        if high_torque_joints:
            joined = ", ".join(high_torque_joints)
            recommendations.append(f"Review load selection; peak torque is high at: {joined}.")

        if not recommendations:
            recommendations.append("No immediate issues detected in the exported result.")
        return recommendations

    def com_range_cm(self) -> float:
        """Return horizontal COM range in centimeters."""
        if self.result.com.shape[0] == 0:
            return 0.0
        horizontal_com: NDArray[np.float64] = self.result.com[:, 0]
        return float((np.max(horizontal_com) - np.min(horizontal_com)) * 100.0)

    @staticmethod
    def _joint_label(idx: int) -> str:
        if idx < len(JOINT_LABELS):
            return JOINT_LABELS[idx]
        return f"Joint {idx + 1}"
