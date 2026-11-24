"""Frame-times sidecar of exported clips (#15722)."""

from __future__ import annotations

import pytest

from src.tools.native_viewer_export.frame_times import (
    frame_at_time,
    frame_times_path,
    load_frame_times,
    write_frame_times,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_times_round_trip_beside_each_clip(tmp_path) -> None:
    clips = {"face_on": tmp_path / "a_face_on_1x.mp4", "dtl": tmp_path / "a_dtl.mp4"}
    write_frame_times(clips, [0.0, 0.5, 0.5, 1.0])
    for clip in clips.values():
        assert frame_times_path(clip).name == clip.stem + ".times.json"
        assert load_frame_times(clip) == (0.0, 0.5, 0.5, 1.0)
    with pytest.raises(ValueError, match="finite"):
        write_frame_times(clips, [float("nan")])


def test_the_first_nearest_frame_is_chosen_for_a_held_frame() -> None:
    times = (0.0, 0.5, 0.5, 0.5, 1.0)
    assert frame_at_time(times, 0.52) == 1
    assert frame_at_time(times, 0.9) == 4
    with pytest.raises(ValueError, match="outside"):
        frame_at_time(times, 2.0)
    with pytest.raises(ValueError, match="no frames"):
        frame_at_time((), 0.0)
