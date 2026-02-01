"""Kinematic Sequence Analysis Module.

This module provides tools to analyze the kinematic sequence (proximal-to-distal sequencing)
of a golf swing. It identifies peak velocities of body segments and evaluates the
timing and order of these peaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

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
class KinematicSequenceResult:
    """Result of a kinematic sequence analysis."""

    peaks: list[SegmentPeak]
    sequence_order: list[str]  # Names in order of peak time
    expected_order: list[str] | None
    sequence_consistency: float  # 0.0 to 1.0 (adherence to expected order)
    timing_gaps: dict[str, float] = field(
        default_factory=dict
    )  # Time diff between peaks
    is_valid_sequence: bool = False  # True if order matches expected


class KinematicSequenceAnalyzer:
    """Analyzer for kinematic sequencing."""

    def __init__(self, expected_order: list[str] | None = None) -> None:
        """Initialize analyzer.

        Args:
            expected_order: List of segment names in expected order (Proximal -> Distal).
                            e.g. ['Pelvis', 'Thorax', 'Arm', 'Club']
        """
        self.expected_order = expected_order or [
            "Pelvis",
            "Torso",
            "Arm",
            "Club",
        ]

    def analyze(
        self,
        segment_velocities: dict[str, np.ndarray],
        times: np.ndarray,
    ) -> KinematicSequenceResult:
        """Analyze the kinematic sequence from velocity data.

        Args:
            segment_velocities: Dict mapping segment name to velocity array (1D)
            times: Time array corresponding to velocities

        Returns:
            KinematicSequenceResult object
        """
        peaks: list[SegmentPeak] = []
        max_overall_velocity = 0.0

        # 1. Detect peaks for each segment
        for name, velocity_data in segment_velocities.items():
            if len(velocity_data) == 0:
                continue

            # Use absolute velocity for peak detection
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

        # 2. Normalize velocities
        if max_overall_velocity > 0:
            for peak in peaks:
                peak.normalized_velocity = peak.peak_velocity / max_overall_velocity

        # 3. Calculate extended metrics (Speed Gain, Deceleration)
        # Create map for name -> SegmentPeak
        peak_map = {p.name: p for p in peaks}

        if self.expected_order:
            for i, name in enumerate(self.expected_order):
                if name not in peak_map:
                    continue

                current_peak = peak_map[name]

                # Speed Gain (Distal / Proximal)
                # Proximal is expected_order[i-1]
                if i > 0:
                    proximal_name = self.expected_order[i - 1]
                    if proximal_name in peak_map:
                        proximal_peak = peak_map[proximal_name]
                        if proximal_peak.peak_velocity > 1e-6:
                            current_peak.speed_gain = current_peak.peak_velocity / proximal_peak.peak_velocity

                # Deceleration Rate (Slope post-peak)
                # Calculate slope over next 20-50ms or until velocity drops significantly
                # Using 30ms window (approx 0.03s)
                window_duration = 0.03

                # Get velocity data for this segment
                if name in segment_velocities:
                    vel_data = np.abs(segment_velocities[name])

                    # Find indices corresponding to window
                    start_idx = current_peak.index
                    start_time = times[start_idx]
                    target_time = start_time + window_duration

                    # Find end index (closest to target time)
                    # Assuming times are sorted
                    # Using searchsorted for efficiency or just simple search
                    # times is np.ndarray
                    end_idx = np.searchsorted(times, target_time)
                    end_idx = min(end_idx, len(times) - 1)

                    if end_idx > start_idx:
                        v_start = vel_data[start_idx]
                        v_end = vel_data[end_idx]
                        t_start = times[start_idx]
                        t_end = times[end_idx]

                        dt = t_end - t_start
                        if dt > 1e-6:
                            # Deceleration is rate of decrease.
                            # Since we expect v_end < v_start, (v_end - v_start) is negative.
                            # We report positive deceleration rate (magnitude of negative slope).
                            slope = (v_end - v_start) / dt
                            # If slope is negative (decelerating), store magnitude.
                            # If slope is positive (accelerating), store 0 or negative?
                            # Usually deceleration is expected.
                            current_peak.deceleration_rate = -slope

        # 4. Sort by time to determine actual sequence
        peaks.sort(key=lambda x: x.time)
        actual_order = [p.name for p in peaks]

        # 5. Calculate timing gaps
        timing_gaps: dict[str, float] = {}
        for i in range(len(peaks) - 1):
            current = peaks[i]
            next_peak = peaks[i + 1]
            gap_name = f"{current.name}->{next_peak.name}"
            timing_gaps[gap_name] = next_peak.time - current.time

        # 6. Evaluate against expected order
        sequence_consistency = 0.0
        is_valid = False

        if self.expected_order:
            # Filter expected order to only include segments present in data
            relevant_expected = [
                name for name in self.expected_order if name in segment_velocities
            ]

            if not relevant_expected:
                sequence_consistency = 0.0
                is_valid = False
            else:
                # Calculate simple match score

                # Check direct position matches
                # Note: This is strict. If one is missing, it shifts everything.
                # Better: Check relative ordering pairs.

                # Pairwise ordering check (more robust than absolute index)
                total_pairs = 0
                correct_pairs = 0

                # Create a map of name -> time for easy lookup
                peak_map = {p.name: p.time for p in peaks}

                import itertools

                for s1, s2 in itertools.combinations(relevant_expected, 2):
                    if s1 in peak_map and s2 in peak_map:
                        total_pairs += 1
                        if peak_map[s1] < peak_map[s2]:
                            correct_pairs += 1

                if total_pairs > 0:
                    sequence_consistency = correct_pairs / total_pairs
                else:
                    # If no pairs could be compared (e.g. not enough peaks),
                    # we can't determine consistency.
                    # If we have 0 or 1 peak, it's technically "ordered", but implies missing data.
                    # For strict analysis, if we expected a sequence but found nothing, score should be low.
                    # However, if we just have 1 segment, it is perfectly ordered with itself.
                    # Let's keep 1.0 for "no violations" but mark as invalid if < 2 peaks.
                    # Wait, test expects 0.0 for empty data.
                    # If len(peaks) == 0, consistency should be 0.0 (no sequence).
                    sequence_consistency = 1.0 if len(peaks) == 1 else 0.0

                is_valid = sequence_consistency == 1.0 and len(peaks) >= 2

        return KinematicSequenceResult(
            peaks=peaks,
            sequence_order=actual_order,
            expected_order=self.expected_order,
            sequence_consistency=sequence_consistency,
            timing_gaps=timing_gaps,
            is_valid_sequence=is_valid,
        )

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
