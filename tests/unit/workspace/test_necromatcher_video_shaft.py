"""Native source-sized shaft overlays preserve sparse evidence and legacy output."""

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.unit.workspace import test_necromatcher_video as video_fixtures

video_case = video_fixtures.video_case

pytestmark = pytest.mark.unit


@pytest.fixture
def shaft_case(video_case: tuple[Any, Any, dict[str, Any]]) -> tuple[Any, Any]:
    from src.shared.python.motion_matching.historical_fit import (
        ShaftAxisEvidence,
        ShaftAxisSegment,
        SourceBoundShaftFrame,
    )
    from src.shared.python.shadow_tracker.source_records import FrameIdentity
    from src.shared.python.workspace import CaptureReview

    library, capture, _ = video_case
    items = []
    with CaptureReview(library, capture.dataset_id) as review:
        for index in range(2):
            segment = (
                ShaftAxisSegment(
                    "observed",
                    ((140, 60), (180, 60)),
                    "test",
                    "Synthetic interior fragment",
                    0.8,
                    None,
                    3.0,
                )
                if index == 0
                else ShaftAxisSegment(
                    "ambiguous", None, "test", "Abstain", None, None, None
                )
            )
            items.append(
                SourceBoundShaftFrame(
                    index,
                    FrameIdentity.from_dict(review.frame(index)["frame"]),
                    "sha256:" + hashlib.sha256(review.image(index)).hexdigest(),
                    segment,
                )
            )
    evidence = ShaftAxisEvidence(
        capture.dataset_id,
        capture.metadata["hash"],
        "sha256:" + "a" * 64,
        (320, 240),
        tuple(items),
    )
    return library, evidence


def test_enabled_overlay_records_raw_shaft_metrics_and_original_bindings(
    shaft_case: tuple[Any, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cv2
    from src.shared.python.workspace import necromatcher_video as module

    library, evidence = shaft_case
    source_before = Path(library.load_asset(evidence.capture_id).path).read_bytes()
    calls = []
    original = module.load_native_fit_binding

    def load(*args: Any) -> Any:
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(module, "load_native_fit_binding", load)
    output = tmp_path / "shaft"
    manifest = module.export_fit_video(
        library, "video-fit", output, selected_frames=(0, 1), shaft_evidence=evidence
    )
    assert len(calls) == 1
    shaft = manifest["shaft_overlay"]
    assert shaft["evidence_sha256"] == evidence.sha256
    assert shaft["axis"]["semantic"] == "infinite_authored_shaft_axis"
    assert shaft["uncertainty_calibrated"] is False
    assert shaft["physical_geometry_qualified"] is False
    assert manifest["physical_time_qualified"] is False
    assert manifest["frames"][0]["shaft_overlay"]["raw_rms_pixels"] is not None
    assert manifest["frames"][1]["shaft_overlay"]["status"] == "ambiguous"
    assert manifest["frames"][1]["shaft_overlay"]["raw_rms_pixels"] is None
    image = cv2.imread(str(output / "frame-000000.png"))
    assert image.shape == (240, 320, 3)
    assert np.array_equal(image[60, 160], np.array([230, 60, 230]))
    assert (
        Path(library.load_asset(evidence.capture_id).path).read_bytes() == source_before
    )
    import json

    assert (
        json.loads((output / "manifest.json").read_text(encoding="utf-8")) == manifest
    )


def test_disabled_overlay_preserves_legacy_frames_and_manifest(
    video_case: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    from src.shared.python.workspace.necromatcher_video import export_fit_video

    first = export_fit_video(
        video_case[0], "video-fit", tmp_path / "default", selected_frames=(0,)
    )
    explicit = export_fit_video(
        video_case[0],
        "video-fit",
        tmp_path / "none",
        selected_frames=(0,),
        shaft_evidence=None,
    )
    assert first == explicit
    assert "shaft_overlay" not in first
    assert (tmp_path / "default/frame-000000.png").read_bytes() == (
        tmp_path / "none/frame-000000.png"
    ).read_bytes()


@pytest.mark.parametrize("fault", ["png", "pts", "camera"])
def test_changed_source_evidence_cannot_publish(
    shaft_case: tuple[Any, Any], tmp_path: Path, fault: str
) -> None:
    from src.shared.python.workspace.necromatcher_video import export_fit_video

    library, evidence = shaft_case
    first = evidence.frames[0]
    if fault == "png":
        first = replace(first, png_sha256="sha256:" + "f" * 64)
    elif fault == "pts":
        first = replace(first, frame=replace(first.frame, pts_ticks=-1))
    else:
        first = replace(first, frame=replace(first.frame, camera_id="foreign-camera"))
        evidence = replace(
            evidence,
            frames=(
                first,
                replace(
                    evidence.frames[1],
                    frame=replace(evidence.frames[1].frame, camera_id="foreign-camera"),
                ),
            ),
        )
    if fault != "camera":
        evidence = replace(evidence, frames=(first, evidence.frames[1]))
    with pytest.raises(ValueError, match="PNG|identity|camera"):
        export_fit_video(
            library, "video-fit", tmp_path / "bad", shaft_evidence=evidence
        )
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "points",
    [np.array([[4.0, 4.0], [4.0, 4.0]]), np.array([[np.nan, 1.0], [1.0, 2.0]])],
)
def test_degenerate_or_nonfinite_axis_rejected(points: np.ndarray) -> None:
    from src.shared.python.workspace.necromatcher_video import clip_infinite_axis

    with pytest.raises(ValueError, match="finite|degenerate"):
        clip_infinite_axis(points, (320, 240))


def test_infinite_axis_clips_to_image_not_authored_endpoints() -> None:
    from src.shared.python.workspace.necromatcher_video import clip_infinite_axis

    start, end = clip_infinite_axis(
        np.array([[150.0, 40.0], [150.0, 80.0]]), (320, 240)
    )
    assert np.array_equal(start, [150.0, 0.0])
    assert np.array_equal(end, [150.0, 239.0])
    assert (
        clip_infinite_axis(np.array([[-10.0, 1.0], [-10.0, 40.0]]), (320, 240)) is None
    )


def test_changed_original_png_during_render_fails_closed(
    shaft_case: tuple[Any, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cv2
    from src.shared.python.workspace import CaptureReview
    from src.shared.python.workspace.necromatcher_video import export_fit_video

    original = CaptureReview.image
    reads = 0

    def image(review: Any, index: int) -> bytes:
        nonlocal reads
        if index == 0:
            reads += 1
            if reads == 2:
                ok, encoded = cv2.imencode(".png", np.full((240, 320, 3), 30, np.uint8))
                assert ok
                return encoded.tobytes()
        return original(review, index)

    monkeypatch.setattr(CaptureReview, "image", image)
    with pytest.raises(ValueError, match="PNG changed"):
        export_fit_video(
            shaft_case[0],
            "video-fit",
            tmp_path / "changed",
            shaft_evidence=shaft_case[1],
        )
    assert not (tmp_path / "changed").exists()


def test_unreviewed_frames_remain_unreviewed(
    shaft_case: tuple[Any, Any], tmp_path: Path
) -> None:
    from src.shared.python.workspace.necromatcher_video import export_fit_video

    library, evidence = shaft_case
    sparse = replace(evidence, frames=(evidence.frames[0],))
    manifest = export_fit_video(
        library, "video-fit", tmp_path / "sparse", shaft_evidence=sparse
    )
    assert manifest["frames"][1]["shaft_overlay"]["status"] == "unreviewed"
    assert manifest["frames"][1]["shaft_overlay"]["raw_rms_pixels"] is None
    assert manifest["shaft_overlay"]["assessment"]["observed_segment_count"] == 1


@pytest.mark.parametrize("reverse", [False, True])
def test_axis_clipping_retains_diagonal_orientation(reverse: bool) -> None:
    from src.shared.python.workspace.necromatcher_video import clip_infinite_axis

    points = np.array([[2.0, 2.0], [4.0, 4.0]])
    clipped = clip_infinite_axis(points[::-1] if reverse else points, (20, 10))
    assert clipped is not None
    assert {tuple(point) for point in clipped} == {(0.0, 0.0), (9.0, 9.0)}
