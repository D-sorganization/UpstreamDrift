"""Comparison drawing media preserves the existing video compositor recipe."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.coaching import Drawing, DrawingLayer, render_layer
from src.motion_capture.reference.comparison import ComparisonLayer
from src.tools.capture_rig.clips import _rendered
from src.tools.capture_rig.comparison_frame_source import ComparisonFrameSource
from src.tools.capture_rig.model_frame_source import ModelFrameSource
from src.tools.capture_rig.player import VideoReader
from src.tools.capture_rig.overlay import PoseTrack
from src.tools.capture_rig.reference_rendering import (
    ComparisonRenderContext,
    ComparisonRenderer,
)
from tests.motion_capture.rig.test_ingest import _bundle
from tests.tools.capture_rig.test_reference_comparison_ui import synthetic_motion
from src.tools.capture_rig.session import load_session

pytestmark = pytest.mark.unit


def test_editable_comparison_matches_original_grid_export_recipe(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)
    motion = synthetic_motion()
    virtual = ModelFrameSource.from_motion(motion, size=(64, 48))
    recipe = virtual.recipe
    virtual.close()
    drawings = DrawingLayer(view="a", width=64, height=48, frames=6).with_shapes(
        (Drawing(kind="line", start=(0, 24), end=(63, 24), stroke=4),)
    )
    track = PoseTrack(
        fps=30,
        names=("left", "right"),
        edges=((0, 1),),
        estimator=None,
        frames={0: (np.array([[10, 24], [50, 24]]), np.array([1.0, 0.1]))},
    )
    context = ComparisonRenderContext(
        "a",
        motion,
        recipe.registration,
        ComparisonLayer(opacity=0.6),
        drawings=drawings,
        track=track,
    )
    path = load_session(root).view("a").recording
    camera = recipe.camera.record()
    source = ComparisonFrameSource(VideoReader(path), context, camera)
    assert (source.width, source.height, source.frame_count) == (64, 48, 6)
    with VideoReader(path) as reader, ComparisonRenderer(motion) as renderer:
        rendering = replace(
            renderer.rendering(context, camera, reader.fps), clock=False
        )
        for index in (0, 3, 5):
            expected = _rendered(
                reader, index, track, min_confidence=0.5, label="a", rendering=rendering
            )
            actual = source.finalize(
                render_layer(source.read(index), drawings, index), index
            )
            np.testing.assert_array_equal(actual, expected)
    source.close()
    with pytest.raises(ValueError, match="closed"):
        source.read(0)
