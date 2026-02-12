"""Segment Timing Analysis Module.

This module provides tools to analyze the timing sequence of body segment
peak velocities during dynamic movements. It identifies peak velocities and
evaluates the timing and order of these peaks against a user-supplied
expected order.

Note: This module does NOT implement any proprietary or patented analysis
methodology. The expected segment order must be supplied by the caller
and is treated as a neutral, user-defined parameter.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.analysis.dataclasses import (
    CITATION_SEGMENT_TIMING,
    MethodCitation,
)

if TYPE_CHECKING:
    from typing import Any


@dataclass
class SegmentPeak:
    """Information about a specific segment's peak velocity."""

    name: str
    peak_velocity: float  # Magnitude
    time: float
    index: int
    normalized_velocity: float = 0.0  # 0.0 to 1.0 relative to max in set
    speed_gain: float | None = None  # Ratio to proximal peak
    deceleration_rate: float | None = None  # Deceleration immediately after peak


@dataclass
class SegmentTimingResult:
    """Result of a segment timing analysis."""

    peaks: list[SegmentPeak]
    sequence_order: list[str]  # Names in order of peak time
    expected_order: list[str] | None
    sequence_consistency: float  # 0.0 to 1.0 (adherence to expected order)
    timing_gaps: dict[str, float] = field(
        default_factory=dict
    )  # Time diff between peaks
    is_valid_sequence: bool = False  # True if order matches expected
    methodology: MethodCitation | None = None


# Backward-compatible alias
KinematicSequenceResult = SegmentTimingResult


class SegmentTimingAnalyzer:
    """Analyzer for segment timing sequence.

    Evaluates the temporal ordering of peak velocities across body segments
    during a dynamic movement. The expected order is fully user-defined and
    not tied to any specific proprietary methodology.
    """

    def __init__(self, expected_order: list[str] | None = None) -> None:
        """Initialize analyzer.

        Args:
            expected_order: List of segment names in expected order (Proximal -> Distal).
                            Must be supplied by the caller. If None, only peak detection
                            is performed (no sequence scoring).
        """
        self.expected_order = expected_order

    def analyze(
        self,
        segment_velocities: dict[str, np.ndarray],
        times: np.ndarray,
    ) -> SegmentTimingResult:
        """Analyze the kinematic sequence from velocity data.

        Args:
            segment_velocities: Dict mapping segment name to velocity array (1D)
            times: Time array corresponding to velocities

        Returns:
            KinematicSequenceResult object
        """
        # 1. Detect peaks for each segment
        peaks = self._detect_peaks(segment_velocities, times)

        # 2. Calculate extended metrics (Speed Gain, Deceleration)
        self._compute_extended_metrics(peaks, segment_velocities, times)

        # 3. Sort by time to determine actual sequence
        peaks.sort(key=lambda x: x.time)
        actual_order = [p.name for p in peaks]

        # 4. Calculate timing gaps
        timing_gaps = self._compute_timing_gaps(peaks)

        # 5. Evaluate against expected order
        sequence_consistency, is_valid = self._evaluate_sequence_order(
            peaks, segment_velocities
        )

        return SegmentTimingResult(
            peaks=peaks,
            sequence_order=actual_order,
            expected_order=self.expected_order,
            sequence_consistency=sequence_consistency,
            timing_gaps=timing_gaps,
            is_valid_sequence=is_valid,
            methodology=CITATION_SEGMENT_TIMING,
        )

    @staticmethod
    def _detect_peaks(
        segment_velocities: dict[str, np.ndarray],
        times: np.ndarray,
    ) -> list[SegmentPeak]:
        """Detect peak velocity for each segment and normalize.

        Args:
            segment_velocities: Dict mapping segment name to velocity array (1D)
            times: Time array corresponding to velocities

        Returns:
            List of SegmentPeak objects with normalized velocities.
        """
        peaks: list[SegmentPeak] = []
        max_overall_velocity = 0.0

        for name, velocity_data in segment_velocities.items():
            if len(velocity_data) == 0:
                continue

            abs_vel = np.abs(velocity_data)
            max_idx = np.argmax(abs_vel)
            peak_val = float(abs_vel[max_idx])
            peak_time = float(times[max_idx])

            max_overall_velocity = max(max_overall_velocity, peak_val)

            peaks.append(
                SegmentPeak(
                    name=name,
                    peak_velocity=peak_val,
                    time=peak_time,
                    index=int(max_idx),
                )
            )

        # Normalize velocities
        if max_overall_velocity > 0:
            for peak in peaks:
                peak.normalized_velocity = peak.peak_velocity / max_overall_velocity

        return peaks

    def _compute_extended_metrics(
        self,
        peaks: list[SegmentPeak],
        segment_velocities: dict[str, np.ndarray],
        times: np.ndarray,
    ) -> None:
        """Compute speed gain and deceleration rate for each peak.

        Args:
            peaks: List of detected segment peaks.
            segment_velocities: Dict mapping segment name to velocity array (1D)
            times: Time array corresponding to velocities
        """
        peak_map = {p.name: p for p in peaks}

        # Speed Gain -- requires expected_order to identify proximal segments
        if self.expected_order:
            for i, name in enumerate(self.expected_order):
                if name not in peak_map:
                    continue
                current_peak = peak_map[name]
                if i > 0:
                    proximal_name = self.expected_order[i - 1]
                    if proximal_name in peak_map:
                        proximal_peak = peak_map[proximal_name]
                        if proximal_peak.peak_velocity > 1e-6:
                            current_peak.speed_gain = (
                                current_peak.peak_velocity / proximal_peak.peak_velocity
                            )

        # Deceleration Rate -- computed for ALL segments (independent of expected_order)
        window_duration = 0.03  # 30ms post-peak window
        for name, peak_info in peak_map.items():
            if name in segment_velocities:
                vel_data = np.abs(segment_velocities[name])
                start_idx = peak_info.index
                start_time = times[start_idx]
                target_time = start_time + window_duration

                end_idx = int(np.searchsorted(times, target_time))
                end_idx = min(end_idx, len(times) - 1)

                if end_idx > start_idx:
                    v_start = vel_data[start_idx]
                    v_end = vel_data[end_idx]
                    dt = times[end_idx] - times[start_idx]
                    if dt > 1e-6:
                        slope = (v_end - v_start) / dt
                        peak_info.deceleration_rate = -slope

    @staticmethod
    def _compute_timing_gaps(peaks: list[SegmentPeak]) -> dict[str, float]:
        """Compute time gaps between consecutive peaks.

        Args:
            peaks: Sorted list of segment peaks.

        Returns:
            Dict mapping gap names to time differences.
        """
        timing_gaps: dict[str, float] = {}
        for i in range(len(peaks) - 1):
            current = peaks[i]
            next_peak = peaks[i + 1]
            gap_name = f"{current.name}->{next_peak.name}"
            timing_gaps[gap_name] = next_peak.time - current.time
        return timing_gaps

    def _evaluate_sequence_order(
        self,
        peaks: list[SegmentPeak],
        segment_velocities: dict[str, np.ndarray],
    ) -> tuple[float, bool]:
        """Evaluate actual peak order against expected order.

        Args:
            peaks: Sorted list of segment peaks.
            segment_velocities: Dict mapping segment name to velocity array.

        Returns:
            Tuple of (sequence_consistency, is_valid_sequence).
        """
        if not self.expected_order:
            return 0.0, False

        # Filter expected order to only include segments present in data
        relevant_expected = [
            name for name in self.expected_order if name in segment_velocities
        ]

        if not relevant_expected:
            return 0.0, False

        # Pairwise ordering check (more robust than absolute index)
        total_pairs = 0
        correct_pairs = 0

        peak_times = {p.name: p.time for p in peaks}

        for s1, s2 in itertools.combinations(relevant_expected, 2):
            if s1 in peak_times and s2 in peak_times:
                total_pairs += 1
                if peak_times[s1] < peak_times[s2]:
                    correct_pairs += 1

        if total_pairs > 0:
            sequence_consistency = correct_pairs / total_pairs
        else:
            # If no pairs could be compared (e.g. not enough peaks),
            # we can't determine consistency.
            # If len(peaks) == 0, consistency should be 0.0 (no sequence).
            sequence_consistency = 1.0 if len(peaks) == 1 else 0.0

        is_valid = sequence_consistency == 1.0 and len(peaks) >= 2
        return sequence_consistency, is_valid

    def extract_velocities_from_recorder(
        self,
        recorder: Any,  # Protocol: RecorderInterface
        segment_indices: dict[str, int],
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Helper to extract relevant data from a recorder.

        Args:
            recorder: Object with get_time_series method
            segment_indices: Map of segment names to joint indices

        Returns:
            (segment_velocities, times)
        """
        times, joint_velocities = recorder.get_time_series("joint_velocities")
        joint_velocities = np.asarray(joint_velocities)

        segment_data = {}

        if len(times) > 0 and joint_velocities.size > 0:
            for name, idx in segment_indices.items():
                if idx < joint_velocities.shape[1]:
                    segment_data[name] = joint_velocities[:, idx]

        return segment_data, np.asarray(times)


# Backward-compatible alias
KinematicSequenceAnalyzer = SegmentTimingAnalyzer
