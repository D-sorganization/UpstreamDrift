"""Unit tests for motion_pipeline.preprocessing.pipeline composition."""

from __future__ import annotations

import pytest

from src.shared.python.motion_pipeline.contracts import (
    KeypointSequence,
    MarkerTrajectory,
)
from src.shared.python.motion_pipeline.preprocessing.filter import FilterType
from src.shared.python.motion_pipeline.preprocessing.gap_fill import GapFillStrategy
from src.shared.python.motion_pipeline.preprocessing.normalize import UpAxis
from src.shared.python.motion_pipeline.preprocessing.pipeline import (
    FilterStep,
    GapFillStep,
    NormalizeStep,
    PreprocessingPipeline,
    ResampleStep,
)

from ._local_fixtures import make_keypoint_sequence, make_marker_trajectory


def test_empty_pipeline_is_identity() -> None:
    pipeline = PreprocessingPipeline()
    seq = make_keypoint_sequence(num_frames=10, num_kp=2)
    out = pipeline.apply(seq)
    assert out is seq


def test_pipeline_len_and_iter() -> None:
    p = PreprocessingPipeline()
    p.add_step(GapFillStep())
    p.add_step(FilterStep())
    assert len(p) == 2
    assert len(list(iter(p))) == 2


def test_gapfill_step_default_strategy() -> None:
    step = GapFillStep()
    assert step.strategy == GapFillStrategy.LINEAR


def test_filter_step_default_filter_type() -> None:
    step = FilterStep()
    assert step.filter_type == FilterType.BUTTERWORTH


def test_normalize_step_default_target_up() -> None:
    step = NormalizeStep()
    assert step.target_up == UpAxis.Y_UP


def test_pipeline_chain_preserves_cir_types() -> None:
    """Each step takes and returns a KeypointSequence (LoD)."""
    p = PreprocessingPipeline(steps=[GapFillStep(), FilterStep(fps=30.0)])
    seq = make_keypoint_sequence(num_frames=20, num_kp=2, fps=30.0)
    out = p.apply(seq)
    assert isinstance(out, KeypointSequence)
    assert out.num_frames == seq.num_frames


def test_pipeline_with_resample_changes_frame_count() -> None:
    p = PreprocessingPipeline(
        steps=[
            FilterStep(fps=30.0),
            ResampleStep(target_fps=120.0, source_fps=30.0),
        ]
    )
    seq = make_keypoint_sequence(num_frames=15, num_kp=1, fps=30.0)
    out = p.apply(seq)
    # Resample upsamples
    assert out.num_frames > seq.num_frames


def test_pipeline_with_normalize_step_works_on_marker_traj() -> None:
    p = PreprocessingPipeline(
        steps=[
            NormalizeStep(
                target_up=UpAxis.Y_UP, source_up=UpAxis.Y_UP, center_origin=True
            ),
        ]
    )
    traj = make_marker_trajectory(num_frames=10)
    out = p.apply(traj)
    assert isinstance(out, MarkerTrajectory)
    assert out.num_frames == traj.num_frames


def test_pipeline_add_step_appends_in_order() -> None:
    p = PreprocessingPipeline()
    s1 = GapFillStep()
    s2 = FilterStep()
    p.add_step(s1)
    p.add_step(s2)
    assert p.steps == [s1, s2]


def test_pipeline_full_chain_compiles_and_runs() -> None:
    p = PreprocessingPipeline(
        steps=[
            GapFillStep(strategy=GapFillStrategy.LINEAR),
            FilterStep(filter_type=FilterType.BUTTERWORTH, cutoff=6.0, fps=30.0),
            NormalizeStep(target_up=UpAxis.Y_UP, source_up=UpAxis.Y_UP),
        ]
    )
    seq = make_keypoint_sequence(num_frames=20, num_kp=2, fps=30.0)
    out = p.apply(seq)
    assert isinstance(out, KeypointSequence)
    # Each successive step recorded its metadata
    assert out.metadata.get("normalized") is True
