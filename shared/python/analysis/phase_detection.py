"""Swing phase detection analysis module."""

import numpy as np
from scipy.signal import savgol_filter

from typing import cast
from shared.python.analysis.dataclasses import SummaryStatistics, SwingPhase


class PhaseDetectionMixin:
    """Mixin for detecting and analyzing swing phases."""

    def detect_swing_phases(self) -> list[SwingPhase]:
        """Automatically detect swing phases.

        Uses heuristics based on club head speed and position.

        Returns:
            List of SwingPhase objects
        """
        phases = []
        club_head_speed = getattr(self, "club_head_speed", None)
        times = getattr(self, "times", None)
        duration = getattr(self, "duration", 0.0)

        if club_head_speed is None or len(club_head_speed) < 20:
            # If no club head data, return single phase
            t_start = float(times[0]) if times is not None and len(times) > 0 else 0.0
            t_end = float(times[-1]) if times is not None and len(times) > 0 else 0.0
            idx_end = len(times) - 1 if times is not None and len(times) > 0 else 0
            
            return [
                SwingPhase(
                    name="Complete Swing",
                    start_time=t_start,
                    end_time=t_end,
                    start_index=0,
                    end_index=idx_end,
                    duration=float(duration),
                ),
            ]

        # Smooth speed for phase detection
        window_len = min(11, len(club_head_speed))
        if window_len % 2 == 0:
            window_len -= 1

        if window_len <= 3:
            smoothed_speed = club_head_speed
        else:
            smoothed_speed = savgol_filter(club_head_speed, window_len, 3)

        # Key events
        impact_idx = np.argmax(smoothed_speed)  # Peak speed = impact

        # Find transition (top of backswing) - minimum speed before impact
        search_end = int(impact_idx * 0.7)
        if search_end > 5:
            # Avoid searching in very beginning noise
            transition_idx = 5 + np.argmin(smoothed_speed[5:search_end])
        else:
            transition_idx = impact_idx // 2

        # Find takeaway start (first significant movement)
        speed_threshold = 0.1 * smoothed_speed[transition_idx]
        takeaway_idx = 0
        # OPTIMIZATION: Use vectorized search instead of loop
        search_region = smoothed_speed[1:transition_idx]
        mask = search_region > speed_threshold
        if np.any(mask):
            takeaway_idx = 1 + int(np.argmax(mask))

        # Find finish (speed drops after impact)
        finish_threshold = 0.3 * smoothed_speed[impact_idx]
        finish_idx = len(smoothed_speed) - 1

        # OPTIMIZATION: Use vectorized search instead of loop
        search_region_post = smoothed_speed[impact_idx + 1 :]
        mask_post = search_region_post < finish_threshold
        if np.any(mask_post):
            finish_idx = int(impact_idx + 1 + np.argmax(mask_post))

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

        if times is None:
             # Should not happen if other logic holds, but safe fallback
             times = np.zeros(0)

        for name, start_idx_raw, end_idx_raw in phase_definitions:
            # Type cast to handle tuple unpacking
            start_idx_val = int(cast(int, start_idx_raw))
            end_idx_val = int(cast(int, end_idx_raw))
            
            # Ensure indices are within bounds
            max_idx = len(times) - 1
            start_idx = int(max(0, min(start_idx_val, max_idx)))
            end_idx = int(max(start_idx, min(end_idx_val, max_idx)))

            phases.append(
                SwingPhase(
                    name=name,
                    start_time=float(times[start_idx]) if max_idx >= 0 else 0.0,
                    end_time=float(times[end_idx]) if max_idx >= 0 else 0.0,
                    start_index=int(start_idx),
                    end_index=int(end_idx),
                    duration=float(times[end_idx] - times[start_idx]) if max_idx >= 0 else 0.0,
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
        # Assuming compute_summary_stats is available (from BasicStatsMixin via MRO)
        compute_summary_stats = getattr(self, "compute_summary_stats", None)

        if not compute_summary_stats:
            # Fallback or raise error? For now, we assume it's mixed in.
            return {}

        original_times = getattr(self, "times", None)

        for phase in phases:
            phase_data = data[phase.start_index : phase.end_index + 1]
            if len(phase_data) > 0:
                # Temporarily override times for this phase if needed by compute_summary_stats
                # Note: compute_summary_stats in shared/python/analysis/basic_stats.py
                # doesn't seem to rely on self.times, it takes data as input.
                # However, the original code did this swap.
                # Let's check basic_stats.py...
                # It uses np.diff(times) if available or assumes uniform dt.
                # If we want phase-specific stats including derivatives, we might need context.
                # But looking at BasicStatsMixin, it takes 'data' as arg.

                # To be safe and mimic original behavior:
                if original_times is not None:
                    self.times = original_times[phase.start_index : phase.end_index + 1]

                phase_stats[phase.name] = compute_summary_stats(phase_data)

                if original_times is not None:
                    self.times = original_times

        return phase_stats
