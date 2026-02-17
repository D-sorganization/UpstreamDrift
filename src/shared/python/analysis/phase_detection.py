"""Swing phase detection analysis module."""

from typing import cast

import numpy as np
from scipy.signal import savgol_filter

from src.shared.python.analysis.dataclasses import SummaryStatistics, SwingPhase
from src.shared.python.core.contracts import ensure


class PhaseDetectionMixin:
    """Mixin for detecting and analyzing swing phases."""

    def detect_swing_phases(self) -> list[SwingPhase]:
        """Automatically detect swing phases.

        Uses heuristics based on club head speed and position.

        Design by Contract:
            Postconditions:
                - all phases have duration >= 0
                - all phases have start_time <= end_time
                - all phases have start_index <= end_index

        Returns:
            List of SwingPhase objects
        """
        club_head_speed = getattr(self, "club_head_speed", None)
        times = getattr(self, "times", None)
        duration = getattr(self, "duration", 0.0)

        if club_head_speed is None or len(club_head_speed) < 20:
            return self._fallback_single_phase(times, duration)

        smoothed_speed = self._smooth_speed(club_head_speed)
        impact_idx, transition_idx, takeaway_idx, finish_idx = self._find_key_events(
            smoothed_speed
        )
        phase_defs = self._build_phase_definitions(
            smoothed_speed,
            impact_idx,
            transition_idx,
            takeaway_idx,
            finish_idx,
        )

        if times is None:
            times = np.zeros(0)

        phases = self._create_phases_from_definitions(phase_defs, times)

        # Postconditions
        for phase in phases:
            ensure(
                phase.duration >= 0,
                f"phase '{phase.name}' duration must be non-negative",
                phase.duration,
            )
            ensure(
                phase.start_time <= phase.end_time,
                f"phase '{phase.name}' start_time must be <= end_time",
            )
            ensure(
                phase.start_index <= phase.end_index,
                f"phase '{phase.name}' start_index must be <= end_index",
            )

        return phases

    @staticmethod
    def _fallback_single_phase(
        times: np.ndarray | None,
        duration: float,
    ) -> list[SwingPhase]:
        """Return a single 'Complete Swing' phase when data is insufficient."""
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

    @staticmethod
    def _smooth_speed(club_head_speed: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay smoothing to club head speed."""
        window_len = min(11, len(club_head_speed))
        if window_len % 2 == 0:
            window_len -= 1

        if window_len <= 3:
            return club_head_speed
        return savgol_filter(club_head_speed, window_len, 3)

    @staticmethod
    def _find_key_events(
        smoothed_speed: np.ndarray,
    ) -> tuple[int, int, int, int]:
        """Locate impact, transition, takeaway, and finish indices."""
        impact_idx = int(np.argmax(smoothed_speed))

        # Transition (top of backswing) - minimum speed before impact
        search_end = int(impact_idx * 0.7)
        if search_end > 5:
            transition_idx = 5 + int(np.argmin(smoothed_speed[5:search_end]))
        else:
            transition_idx = impact_idx // 2

        # Takeaway start (first significant movement)
        speed_threshold = 0.1 * smoothed_speed[transition_idx]
        takeaway_idx = 0
        search_region = smoothed_speed[1:transition_idx]
        mask = search_region > speed_threshold
        if np.any(mask):
            takeaway_idx = 1 + int(np.argmax(mask))

        # Finish (speed drops after impact)
        finish_threshold = 0.3 * smoothed_speed[impact_idx]
        finish_idx = len(smoothed_speed) - 1
        search_region_post = smoothed_speed[impact_idx + 1 :]
        mask_post = search_region_post < finish_threshold
        if np.any(mask_post):
            finish_idx = int(impact_idx + 1 + np.argmax(mask_post))

        return impact_idx, transition_idx, takeaway_idx, finish_idx

    @staticmethod
    def _build_phase_definitions(
        smoothed_speed: np.ndarray,
        impact_idx: int,
        transition_idx: int,
        takeaway_idx: int,
        finish_idx: int,
    ) -> list[tuple[str, int, int]]:
        """Build the list of (name, start_idx, end_idx) phase boundaries."""
        return [
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

    @staticmethod
    def _create_phases_from_definitions(
        phase_definitions: list[tuple[str, int, int]],
        times: np.ndarray,
    ) -> list[SwingPhase]:
        """Convert raw phase definitions into bounded SwingPhase objects."""
        phases: list[SwingPhase] = []
        for name, start_idx_raw, end_idx_raw in phase_definitions:
            start_idx_val = int(cast(int, start_idx_raw))
            end_idx_val = int(cast(int, end_idx_raw))

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
                    duration=(
                        float(times[end_idx] - times[start_idx])
                        if max_idx >= 0
                        else 0.0
                    ),
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
