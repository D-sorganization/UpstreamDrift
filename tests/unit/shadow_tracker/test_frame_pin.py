"""Source-frame mapping preserves repeated-frame lineage."""

import pytest

from shadow_tracker.frame_pin import map_frames

pytestmark = pytest.mark.unit


def test_repeats_map_to_first_consecutive_source_frame() -> None:
    """Consecutive identical digests share one output index."""
    rows = map_frames([(0, 0, "a"), (1, 1001, "a"), (2, 2002, "b"), (3, 3003, "a")])
    assert [row["output_index"] for row in rows] == [0, 0, 1, 2]
    assert [row["source_index"] for row in rows] == [0, 1, 2, 3]
    assert [row["pts"] for row in rows] == [0, 1001, 2002, 3003]
    assert [row["frame_sha256"] for row in rows] == ["a", "a", "b", "a"]


def test_rejects_nonmonotonic_source_identity() -> None:
    """Non-increasing PTS is rejected."""
    with pytest.raises(ValueError):
        map_frames([(0, 0, "a"), (1, 0, "b")])


def test_source_pin_maps_decoded_frames_and_preserves_timebase(tmp_path) -> None:
    """A decoded source maps frames, keeps its timebase and writes unique PNGs."""
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from shadow_tracker.frame_pin import pin_source

    source = tmp_path / "source.mkv"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("ffv1", rate=10)
        stream.width = stream.height = 16
        stream.pix_fmt = "bgr0"
        for value in (0, 0, 255):
            frame = av.VideoFrame.from_ndarray(
                np.full((16, 16, 3), value, dtype=np.uint8), format="bgr24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    output = tmp_path / "frames"
    pin = pin_source(source, output=output)
    assert len(pin["source_sha256"]) == 64
    assert [row["output_index"] for row in pin["frames"]] == [0, 0, 1]
    assert pin["timebase"] == [1, 1000]
    assert pin["physical_time_verified"] is False
    assert sorted(path.name for path in output.iterdir()) == [
        "000000.png",
        "000001.png",
    ]
