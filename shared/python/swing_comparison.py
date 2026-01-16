"""Swing comparison module for biomechanical analysis.

This module provides tools to compare two swing datasets (e.g., Student vs Pro),
quantifying similarities and differences in kinematics, timing, and coordination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from shared.python import common_utils, signal_processing
from shared.python.statistical_analysis import StatisticalAnalyzer

# Constants for scoring
EPSILON = 1e-9
SIMILARITY_SCORE_CONSTANT = 100.0


@dataclass
class ComparisonMetric:
    """Result of a single metric comparison."""

    name: str
    reference_value: float
    student_value: float
    difference: float
    percent_diff: float
    score: float  # 0.0 to 100.0 (similarity)


@dataclass
class DTWResult:
    """Result of Dynamic Time Warping alignment."""

    distance: float
    path: list[tuple[int, int]]
    normalized_distance: float  # Distance per sample
    similarity_score: float  # 0-100 score


class SwingComparator:
    """Comparator for two golf swing datasets."""

    def __init__(
        self,
        reference_data: dict[str, Any] | StatisticalAnalyzer,
        student_data: dict[str, Any] | StatisticalAnalyzer,
    ) -> None:
        """Initialize comparator.

        Args:
            reference_data: Reference swing data (Pro/Model)
            student_data: Student swing data (User)
        """
        self.ref = self._ensure_analyzer(reference_data)
        self.student = self._ensure_analyzer(student_data)

    def _ensure_analyzer(
        self, data: dict[str, Any] | StatisticalAnalyzer
    ) -> StatisticalAnalyzer:
        if isinstance(data, StatisticalAnalyzer):
            return data
        # Assume dict
        return StatisticalAnalyzer(**data)

    def compare_tempo(self) -> ComparisonMetric | None:
        """Compare swing tempo."""
        ref_tempo = self.ref.compute_tempo()
        stu_tempo = self.student.compute_tempo()

        if ref_tempo is None or stu_tempo is None:
            return None

        ref_ratio = ref_tempo[2]
        stu_ratio = stu_tempo[2]
        diff = stu_ratio - ref_ratio
        pct_diff = (diff / ref_ratio) * 100 if ref_ratio != 0 else 0.0

        # Score: Perfect match = 100. Decay based on difference.
        # Acceptable range +/- 10%?
        score = max(0.0, 100.0 - abs(pct_diff) * 2.0)

        return ComparisonMetric(
            name="Tempo Ratio",
            reference_value=ref_ratio,
            student_value=stu_ratio,
            difference=diff,
            percent_diff=pct_diff,
            score=score,
        )

    def compute_kinematic_similarity(
        self,
        joint_idx: int,
        feature: str = "velocity",
    ) -> DTWResult | None:
        """Compute similarity score for a specific joint kinematic trajectory using DTW.

        Args:
            joint_idx: Joint index
            feature: 'position' or 'velocity'

        Returns:
            DTWResult object
        """
        if feature == "position":
            ref_data = self.ref.joint_positions
            stu_data = self.student.joint_positions
        else:
            ref_data = self.ref.joint_velocities
            stu_data = self.student.joint_velocities

        if (
            joint_idx >= ref_data.shape[1]
            or joint_idx >= stu_data.shape[1]
            or len(ref_data) == 0
            or len(stu_data) == 0
        ):
            return None

        # Extract series
        s1 = ref_data[:, joint_idx]
        s2 = stu_data[:, joint_idx]

        # Normalize (Z-score) to compare shape independent of magnitude offset/scale?
        # Or Min-Max?
        # Z-score is standard for shape comparison
        s1_norm = common_utils.normalize_z_score(s1, EPSILON)
        s2_norm = common_utils.normalize_z_score(s2, EPSILON)

        # Compute DTW
        dist, path = signal_processing.compute_dtw_path(s1_norm, s2_norm)

        # Normalize distance by path length
        norm_dist = dist / len(path)

        # Similarity Score (0-100)
        # Empirical: norm_dist 0 -> 100.
        # norm_dist 1.0 (1 std dev avg error) -> maybe 50?
        # Score = 100 / (1 + dist)
        score = SIMILARITY_SCORE_CONSTANT / (1.0 + norm_dist)

        return DTWResult(
            distance=dist,
            path=path,
            normalized_distance=norm_dist,
            similarity_score=score,
        )

    def compare_peak_speeds(
        self,
        segment_indices: dict[str, int],
    ) -> dict[str, ComparisonMetric]:
        """Compare peak speeds of key segments.

        Args:
            segment_indices: Map of segment name to joint index

        Returns:
            Dictionary of comparison metrics
        """
        metrics = {}

        for name, idx in segment_indices.items():
            # Get max velocities
            if (
                idx < self.ref.joint_velocities.shape[1]
                and idx < self.student.joint_velocities.shape[1]
            ):
                ref_peak = float(np.max(np.abs(self.ref.joint_velocities[:, idx])))
                stu_peak = float(np.max(np.abs(self.student.joint_velocities[:, idx])))

                diff = stu_peak - ref_peak
                pct = (diff / ref_peak * 100) if ref_peak > 0 else 0.0

                # Score: 100 if >= ref (for speed). Or match?
                # Usually more speed is good, but let's assume matching target is goal.
                # Let's say matching within 5% is 100.
                error = abs(pct)
                score = max(0.0, 100.0 - error)

                metrics[name] = ComparisonMetric(
                    name=f"Peak {name} Speed",
                    reference_value=ref_peak,
                    student_value=stu_peak,
                    difference=diff,
                    percent_diff=pct,
                    score=score,
                )

        # Club head speed
        if (
            self.ref.club_head_speed is not None
            and self.student.club_head_speed is not None
        ):
            ref_chs = float(np.max(self.ref.club_head_speed))
            stu_chs = float(np.max(self.student.club_head_speed))
            diff = stu_chs - ref_chs
            pct = (diff / ref_chs * 100) if ref_chs > 0 else 0.0
            metrics["Club Head"] = ComparisonMetric(
                name="Peak Club Head Speed",
                reference_value=ref_chs,
                student_value=stu_chs,
                difference=diff,
                percent_diff=pct,
                score=max(0.0, 100.0 - abs(pct)),
            )

        return metrics

    def generate_comparison_report(
        self, segment_indices: dict[str, int] | None = None
    ) -> dict[str, Any]:
        """Generate a full comparison report.

        Args:
            segment_indices: Map of segment names to joint indices

        Returns:
            Dictionary report
        """
        report: dict[str, Any] = {"metrics": {}}

        # Tempo
        tempo_metric = self.compare_tempo()
        if tempo_metric:
            report["metrics"]["tempo"] = tempo_metric

        # Speeds
        if segment_indices:
            speed_metrics = self.compare_peak_speeds(segment_indices)
            report["metrics"].update(speed_metrics)

            # Sequence similarity (DTW)
            report["sequence_similarity"] = {}
            total_sim = 0.0
            count = 0
            for name, idx in segment_indices.items():
                res = self.compute_kinematic_similarity(idx)
                if res:
                    report["sequence_similarity"][name] = res.similarity_score
                    total_sim += res.similarity_score
                    count += 1

            if count > 0:
                report["overall_similarity"] = total_sim / count

        return report
