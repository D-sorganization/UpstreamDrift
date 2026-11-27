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


def test_the_hud_says_why_the_shading_is_withheld_at_ball_contact() -> None:
    """Section loads are blanked within a step of contact (the impact impulse
    is not modelled); unexplained, the unshaded body reads as a glitch."""
    from types import SimpleNamespace

    from src.tools.native_viewer_export.overlay_feed import (
        CONTACT_BLACKOUT_NOTE,
        OverlayFeed,
    )

    def frame(value):
        loads = SimpleNamespace(values_n={"trunk": value})
        return SimpleNamespace(axial_loads=loads, wrenches=())

    blank = OverlayFeed(lambda _i: frame(None))
    assert CONTACT_BLACKOUT_NOTE in blank.hud_lines_at(0)
    assert "impulse" in CONTACT_BLACKOUT_NOTE
    shaded = OverlayFeed(lambda _i: frame(5.0))
    assert CONTACT_BLACKOUT_NOTE not in shaded.hud_lines_at(0)
