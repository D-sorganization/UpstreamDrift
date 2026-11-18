"""Swing comparison module for biomechanical analysis.

This module provides tools to compare two swing datasets (e.g., Student vs Pro),
quantifying similarities and differences in kinematics, timing, and coordination.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.core.contracts import ensure, require
from src.shared.python.data_io import common_utils
from src.shared.python.signal_toolkit import signal_processing
from src.shared.python.validation_pkg.statistical_analysis import StatisticalAnalyzer

# Constants for scoring
EPSILON = 1e-9
SIMILARITY_SCORE_CONSTANT = 100.0


def _percent_difference(reference: float, student: float) -> float:
    """Signed difference of ``student`` from ``reference`` in percent of ``reference``.

    A zero reference has no relative scale: equal values give ``0.0`` and any
    other student value gives ``nan`` (undefined), never ``0.0``, so a student
    who moves against a reference that does not is not reported as a match.
    """
    diff = student - reference
    if reference == 0:
        return 0.0 if diff == 0 else float("nan")
    return diff / reference * 100.0


def _score_from_percent(percent_diff: float, penalty_per_percent: float) -> float:
    """Similarity score in [0, 100]; an undefined (NaN) percent difference scores 0."""
    if math.isnan(percent_diff):
        return 0.0
    return max(0.0, 100.0 - abs(percent_diff) * penalty_per_percent)


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
        if reference_data is None:
            raise ValueError("reference_data must be provided")
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
        pct_diff = _percent_difference(ref_ratio, stu_ratio)

        # Score: Perfect match = 100. Decay based on difference.
        # Acceptable range +/- 10%?
        score = _score_from_percent(pct_diff, 2.0)

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
        if joint_idx is None:
            raise ValueError("joint_idx must be provided")
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
        # Using exponential decay to avoid patent conflicts with specific algebraic forms.
        # norm_dist is Z-score error. 1.0 means ~1 std dev avg error.
        score = SIMILARITY_SCORE_CONSTANT * np.exp(-norm_dist)

        result = DTWResult(
            distance=dist,
            path=path,
            normalized_distance=norm_dist,
            similarity_score=score,
        )
        ensure(
            result.distance >= 0, "DTW distance must be non-negative", result.distance
        )
        ensure(
            0.0 <= result.similarity_score <= 100.0,
            "similarity_score must be in [0, 100]",
            result.similarity_score,
        )
        return result

    def compare_peak_speeds(
        self,
        segment_indices: dict[str, int],
    ) -> dict[str, ComparisonMetric]:
        """Compare peak speeds of key segments.

        Design by Contract:
            Preconditions:
                - segment_indices must be non-empty
            Postconditions:
                - each metric score is in [0, 100]

        Args:
            segment_indices: Map of segment name to joint index

        Returns:
            Dictionary of comparison metrics
        """
        if segment_indices is None:
            raise ValueError("segment_indices must be provided")
        require(
            len(segment_indices) > 0,
            "segment_indices must be non-empty",
        )
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
                pct = _percent_difference(ref_peak, stu_peak)

                # Score: 100 if >= ref (for speed). Or match?
                # Usually more speed is good, but let's assume matching target is goal.
                # Let's say matching within 5% is 100.
                score = _score_from_percent(pct, 1.0)

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
            pct = _percent_difference(ref_chs, stu_chs)
            metrics["Club Head"] = ComparisonMetric(
                name="Peak Club Head Speed",
                reference_value=ref_chs,
                student_value=stu_chs,
                difference=diff,
                percent_diff=pct,
                score=_score_from_percent(pct, 1.0),
            )

        return metrics

    def generate_comparison_report(
        self, segment_indices: dict[str, int] | None = None
    ) -> dict[str, Any]:
        """Generate a full comparison report.

        Design by Contract:
            Postconditions:
                - overall_similarity (if present) is in [0, 100]

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

        if "overall_similarity" in report:
            ensure(
                0.0 <= report["overall_similarity"] <= 100.0,
                "overall_similarity must be in [0, 100]",
                report["overall_similarity"],
            )

        return report
