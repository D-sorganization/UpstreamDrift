"""Compact annotations retain qualifications inside measured source bounds."""

from fractions import Fraction
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_compact_layout_bounds_qualifications_and_exact_clock() -> None:
    from src.shared.python.workspace.necromatcher_caption import (
        CaptionFrame,
        CaptionOverlayOptions,
        caption_layout,
        draw_caption,
    )

    frame = CaptionFrame(200, Fraction(350, 3), 10.65, 13, True, 0.35)
    layout = caption_layout((320, 240), frame, CaptionOverlayOptions())
    assert layout.rectangle[3] <= 48
    joined = " ".join(line.text for line in layout.lines)
    for required in (
        "RESEARCH",
        "Camera/Anatomy Unqualified",
        "Physical Time Unknown",
        "F200",
        "350/3s",
        "Surfaces",
        "Uncalibrated",
    ):
        assert required in joined
    boxes = [line.bounds for line in layout.lines]
    assert all(0 <= x < x + w <= 320 and 0 <= y < y + h <= 240 for x, y, w, h in boxes)
    assert all(a[1] + a[3] <= b[1] for a, b in zip(boxes, boxes[1:], strict=False))
    image = np.full((240, 320, 3), 91, dtype=np.uint8)
    before = image.copy()
    draw_caption(image, layout)
    np.testing.assert_array_equal(
        image[: layout.rectangle[1]], before[: layout.rectangle[1]]
    )
    assert np.any(image != before)


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"style": "legacy"},
        {"style": True},
        {"style": "compact_research_v1", "opacity": 0.35},
    ],
)
def test_caption_options_reject_unknown_or_ambiguous_records(record: Any) -> None:
    from src.shared.python.workspace.necromatcher_caption import CaptionOverlayOptions

    with pytest.raises((TypeError, ValueError)):
        CaptionOverlayOptions.from_record(record)


def test_caption_options_round_trip_and_small_frame_fails_closed() -> None:
    from src.shared.python.workspace.necromatcher_caption import (
        CaptionFrame,
        CaptionOverlayOptions,
        caption_layout,
    )

    options = CaptionOverlayOptions()
    assert CaptionOverlayOptions.from_record(options.to_record()) == options
    with pytest.raises(ValueError, match="fit|dimensions"):
        caption_layout(
            (80, 60), CaptionFrame(0, Fraction(0), None, 0, True, 0.35), options
        )


@pytest.mark.parametrize("size", [(320, 240), (1280, 720)])
def test_unavailable_metric_and_long_identity_never_silently_truncate(
    size: tuple[int, int],
) -> None:
    from src.shared.python.workspace.necromatcher_caption import (
        CaptionFrame,
        CaptionOverlayOptions,
        caption_layout,
    )

    result = caption_layout(
        size,
        CaptionFrame(123456, Fraction(1234567, 30000), None, 0),
        CaptionOverlayOptions(),
    )
    assert "1234567/30000s" in " ".join(line.text for line in result.lines)
    assert "RMS unavailable" in " ".join(line.text for line in result.lines)


def test_caption_source_changes_invalidate_canonical_execution_stamp(
    tmp_path: Any, monkeypatch: Any
) -> None:
    from src.shared.python.workspace import necromatcher_fit_jobs as module

    relative = "src/shared/python/workspace/necromatcher_caption.py"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"# Reviewed caption fixture\n")
    monkeypatch.setattr(module, "get_repo_root", lambda: tmp_path)
    monkeypatch.setattr(module, "read_git_commit", lambda root: "a" * 40)
    monkeypatch.setattr(module, "version", lambda package: "fixture")
    before = module.fit_execution_stamp()
    path.write_bytes(b"# Changed caption semantics\n")
    after = module.fit_execution_stamp()
    assert before["source_files"][relative] != after["source_files"][relative]
    assert before["source_sha256"] != after["source_sha256"]
    assert before["runtime_sha256"] == after["runtime_sha256"]


@pytest.mark.parametrize(
    "image",
    [np.zeros((120, 160, 3), dtype=np.uint8), np.zeros((240, 320, 3), dtype=float)],
)
def test_caption_rejects_wrong_raster_before_drawing(image: np.ndarray) -> None:
    from src.shared.python.workspace.necromatcher_caption import (
        CaptionFrame,
        CaptionOverlayOptions,
        caption_layout,
        draw_caption,
    )

    layout = caption_layout(
        (320, 240), CaptionFrame(0, Fraction(0), None, 0), CaptionOverlayOptions()
    )
    before = image.copy()
    with pytest.raises(ValueError, match="raster"):
        draw_caption(image, layout)
    np.testing.assert_array_equal(image, before)
