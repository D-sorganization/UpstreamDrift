"""Adversarial: resource bounds.

Synthetic large input must complete within a generous wall-clock budget
and not run away with memory.
"""

from __future__ import annotations

import time

import pytest

from src.shared.python.motion_pipeline.contracts import (
    Keypoint,
    KeypointFrame,
    KeypointSequence,
)


@pytest.mark.slow
def test_large_keypoint_sequence_construction_under_60s() -> None:
    """60 seconds @ 1000 Hz = 60000 frames. Building the CIR object alone
    must finish well under 60s wall-clock."""
    n_frames = 60_000
    start = time.perf_counter()
    frames = [
        KeypointFrame(
            timestamp=i * 0.001,
            schema_name="MediaPipe_33",
            keypoints=[Keypoint(x=0.0, y=0.0, confidence=0.9)],
        )
        for i in range(n_frames)
    ]
    seq = KeypointSequence(
        id="big",
        frames=frames,
        fps=1000.0,
        schema_name="MediaPipe_33",
    )
    elapsed = time.perf_counter() - start
    assert len(seq.frames) == n_frames
    assert elapsed < 60.0, f"Construction took {elapsed:.1f}s (budget 60s)"
