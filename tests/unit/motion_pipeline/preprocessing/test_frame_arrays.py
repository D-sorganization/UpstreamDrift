"""Characterization tests for preprocessing frame array conversion."""

from __future__ import annotations

from importlib import import_module
from collections.abc import Callable

import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import (
    Keypoint,
    KeypointFrame,
    Marker,
    MarkerFrame,
)
from src.shared.python.motion_pipeline.preprocessing._frame_arrays import (
    keypoints_to_array,
    markers_to_array,
)

filter_module = import_module("src.shared.python.motion_pipeline.preprocessing.filter")
resample_module = import_module(
    "src.shared.python.motion_pipeline.preprocessing.resample"
)
pure_filter_module = import_module(
    "src.shared.python.motion_pipeline.preprocessing._filter_pure_python"
)
pure_resample_module = import_module(
    "src.shared.python.motion_pipeline.preprocessing._resample_pure_python"
)


KEYPOINT_HELPERS: tuple[Callable[[list[KeypointFrame]], np.ndarray], ...] = (
    keypoints_to_array,
    filter_module._keypoints_to_array,
    resample_module._keypoints_to_array,
    pure_filter_module._keypoints_to_array,
    pure_resample_module._keypoints_to_array,
)

MARKER_HELPERS: tuple[Callable[[list[MarkerFrame]], np.ndarray], ...] = (
    markers_to_array,
    filter_module._markers_to_array,
    resample_module._markers_to_array,
    pure_filter_module._markers_to_array,
    pure_resample_module._markers_to_array,
)


@pytest.mark.parametrize("helper", KEYPOINT_HELPERS)
def test_keypoints_to_array_matches_legacy_shape_and_missing_depth(
    helper: Callable[[list[KeypointFrame]], np.ndarray],
) -> None:
    frames = [
        KeypointFrame(
            timestamp=0.0,
            keypoints=[
                Keypoint(x=1.0, y=2.0, z=None, confidence=0.9, name="hip"),
                Keypoint(x=3.0, y=4.0, z=5.0, confidence=0.8, name="knee"),
            ],
            schema_name="custom",
            frame_index=0,
        ),
        KeypointFrame(
            timestamp=0.1,
            keypoints=[
                Keypoint(x=6.0, y=7.0, z=8.0, confidence=0.7, name="hip"),
                Keypoint(x=9.0, y=10.0, z=None, confidence=0.6, name="knee"),
            ],
            schema_name="custom",
            frame_index=1,
        ),
    ]

    np.testing.assert_allclose(
        helper(frames),
        np.array(
            [
                [[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]],
                [[6.0, 7.0, 8.0], [9.0, 10.0, 0.0]],
            ]
        ),
    )
    assert helper([]).size == 0


@pytest.mark.parametrize("helper", MARKER_HELPERS)
def test_markers_to_array_uses_first_frame_order_and_zero_fills_missing_markers(
    helper: Callable[[list[MarkerFrame]], np.ndarray],
) -> None:
    frames = [
        MarkerFrame(
            timestamp=0.0,
            markers={
                "LASI": Marker(name="LASI", x=1.0, y=2.0, z=3.0),
                "RASI": Marker(name="RASI", x=4.0, y=5.0, z=6.0),
            },
            frame_index=0,
        ),
        MarkerFrame(
            timestamp=0.1,
            markers={
                "LASI": Marker(name="LASI", x=7.0, y=8.0, z=9.0),
                "EXTRA": Marker(name="EXTRA", x=10.0, y=11.0, z=12.0),
            },
            frame_index=1,
        ),
    ]

    np.testing.assert_allclose(
        helper(frames),
        np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [0.0, 0.0, 0.0]],
            ]
        ),
    )
    assert helper([]).size == 0
