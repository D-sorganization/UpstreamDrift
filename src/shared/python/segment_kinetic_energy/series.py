"""Time-indexed segment kinetic-energy series (#15389)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import (
    SegmentKineticEnergyFrame,
    SegmentKineticEnergyPeak,
    SegmentKineticEnergyProvider,
)

__all__ = ["SegmentKineticEnergySeries"]


@dataclass(frozen=True)
class SegmentKineticEnergySeries:
    """Per-frame segment KE samples over a swing trajectory."""

    times_s: tuple[float, ...]
    engine: str
    frames: tuple[SegmentKineticEnergyFrame, ...]

    def __post_init__(self) -> None:
        if len(self.times_s) != len(self.frames):
            raise ValueError("times_s and frames must have the same length")
        if self.frames and self.frames[0].engine != self.engine:
            raise ValueError("engine must match every frame")

    @classmethod
    def from_trajectory(
        cls,
        provider: SegmentKineticEnergyProvider,
        times: np.ndarray,
        qpos_traj: np.ndarray,
        qvel_traj: np.ndarray,
    ) -> SegmentKineticEnergySeries:
        if len(times) == 0:
            raise ValueError("times must be non-empty")
        n = len(times)
        if not (len(qpos_traj) == len(qvel_traj) == n):
            raise ValueError("trajectory arrays must align with times")
        frames = tuple(
            provider.frame_at(qpos_traj[i], qvel_traj[i], time_s=float(times[i]))
            for i in range(n)
        )
        return cls(
            times_s=tuple(float(t) for t in times),
            engine=provider.ENGINE,
            frames=frames,
        )

    def peak_totals(self) -> tuple[SegmentKineticEnergyPeak, ...]:
        peaks: dict[str, SegmentKineticEnergyPeak] = {}
        for frame in self.frames:
            for seg in frame.segments:
                prev = peaks.get(seg.body)
                if prev is None or seg.total_j > prev.peak_total_j:
                    peaks[seg.body] = SegmentKineticEnergyPeak(
                        body=seg.body,
                        peak_total_j=float(seg.total_j),
                        peak_time_s=float(frame.time_s),
                    )
        return tuple(peaks[k] for k in sorted(peaks))
