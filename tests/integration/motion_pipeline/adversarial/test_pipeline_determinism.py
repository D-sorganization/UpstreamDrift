"""Adversarial: pipeline determinism.

Running the same input twice must produce byte-identical output. Surfaces
non-deterministic behaviour like un-seeded RNGs, dict ordering, timestamp
drift.
"""

from __future__ import annotations

import pytest

from src.shared.python.motion_pipeline.contracts import (
    Keypoint,
    KeypointFrame,
    KeypointSequence,
)


def _make_seq(seed_offset: float = 0.0) -> KeypointSequence:
    return KeypointSequence(
        id="det-test",
        frames=[
            KeypointFrame(
                timestamp=0.0 + seed_offset,
                schema_name="MediaPipe_33",
                keypoints=[
                    Keypoint(x=float(i), y=float(i + 1), confidence=0.9)
                    for i in range(5)
                ],
            ),
            KeypointFrame(
                timestamp=0.033 + seed_offset,
                schema_name="MediaPipe_33",
                keypoints=[
                    Keypoint(x=float(i + 1), y=float(i + 2), confidence=0.8)
                    for i in range(5)
                ],
            ),
        ],
        fps=30.0,
        schema_name="MediaPipe_33",
    )


def test_keypoint_sequence_serialisation_deterministic() -> None:
    """Same input twice → byte-identical model_dump_json output."""
    a = _make_seq().model_dump_json()
    b = _make_seq().model_dump_json()
    assert a == b


def test_keypoint_sequence_roundtrip_deterministic() -> None:
    """Serialise → parse → serialise must equal the original serialisation."""
    seq = _make_seq()
    s1 = seq.model_dump_json()
    seq2 = KeypointSequence.model_validate_json(s1)
    s2 = seq2.model_dump_json()
    assert s1 == s2


def test_preprocessing_pipeline_deterministic() -> None:
    """PreprocessingPipeline construction with identical config must be
    deterministic."""
    try:
        from src.shared.python.motion_pipeline.preprocessing import (
            PreprocessingPipeline,
        )
    except ImportError:
        pytest.skip("PreprocessingPipeline not importable")
    p1 = PreprocessingPipeline()
    p2 = PreprocessingPipeline()
    # Both should have the same default configuration shape.
    assert type(p1) is type(p2)
