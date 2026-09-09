"""Adverse rendering and publication contracts for reference comparison (#9882)."""

from pathlib import Path
import json

import cv2
import numpy as np
import pytest

from src.motion_capture.coaching.storage import layer_path
from src.motion_capture.reference.comparison import ComparisonLayer
from src.motion_capture.reference.model import ReferenceSource, ReferenceVideo
from src.motion_capture.reference.registration import ReferenceRegistration
from src.motion_capture.provenance import sha256_of
from src.tools.capture_rig.clips import ClipRendering
from src.tools.capture_rig.player import VideoReader
from src.tools.capture_rig.reference_export import (
    ComparisonVideoExportOptions,
    draw_reference_overlay,
    export_comparison_video,
)
from tests.motion_capture.rig.test_ingest import _bundle
from tests.motion_capture.test_reference_registration import two_camera_rig
from tests.tools.capture_rig.test_reference_comparison_ui import synthetic_motion
from src.motion_capture.coaching import Drawing, DrawingLayer
from src.motion_capture.coaching.storage import save_layer
from src.motion_capture.rig.edits import CropRect, SessionEdits, ViewEdit, save_edits
from src.motion_capture.reference.storage import ReferenceLibrary
from src.motion_capture.reference.evidence import CameraSnapshot, ViewClock
from src.tools.capture_rig.reference_comparison import ReferenceComparisonDialog
from tests.tools.capture_rig.test_pane_layout import _app

pytestmark = pytest.mark.unit


def video_asset(root: Path) -> ReferenceVideo:
    path = root / "b_2.avi"
    return ReferenceVideo(
        title="Expert",
        width=64,
        height=48,
        fps=30,
        frames=6,
        source=ReferenceSource(path=str(path), sha256=sha256_of(path), format="video"),
    )


def registration(asset: ReferenceVideo) -> ReferenceRegistration:
    return ReferenceRegistration(reference_id=asset.id, calibration_id="manual")


def test_motion_opacity_blends_projected_pixels() -> None:
    asset = synthetic_motion()
    reg = ReferenceRegistration(reference_id=asset.id, calibration_id="manual")
    camera = two_camera_rig()[0]
    frame = np.full((720, 1280, 3), 80, dtype=np.uint8)
    full = draw_reference_overlay(
        frame.copy(), asset, 0.0, reg, camera, ComparisonLayer()
    )
    half = draw_reference_overlay(
        frame.copy(), asset, 0.0, reg, camera, ComparisonLayer(opacity=0.5)
    )
    assert np.any(full != frame), "Fixture must project visible reference geometry"
    expected = cv2.addWeighted(full, 0.5, frame, 0.5, 0)
    np.testing.assert_array_equal(half, expected)


def test_video_homography_preserves_uncovered_player_pixels(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    asset = video_asset(root)
    reg = registration(asset).model_copy(
        update={
            "image_transform_2d": ((1.0, 0.0, 20.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        }
    )
    frame = np.full((48, 64, 3), 100, dtype=np.uint8)
    result = draw_reference_overlay(
        frame.copy(), asset, 0, reg, None, ComparisonLayer(opacity=0.5)
    )
    np.testing.assert_array_equal(result[:, :20], frame[:, :20])
    assert np.all(result[:, 22:] < 80)


def test_uncropped_odd_pixels_are_padded_without_loss() -> None:
    rendering = ClipRendering()
    frame = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    assert rendering.size(7, 5) == (8, 6)
    image = rendering.image(frame)
    assert image.shape == (6, 8, 3)
    np.testing.assert_array_equal(image[:5, :7], frame)
    np.testing.assert_array_equal(image[-1, :7], frame[-1])


@pytest.mark.parametrize(
    "failure", ["decode", "drawings", "changed-source", "changed-reference"]
)
def test_failed_export_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    root = _bundle(tmp_path)
    asset = video_asset(root)
    if failure == "decode":
        original_read = VideoReader.read

        def broken_read(self: VideoReader, index: int) -> np.ndarray | None:
            if self.path.name == "a_1.avi" and index == 2:
                return None
            return original_read(self, index)

        monkeypatch.setattr(VideoReader, "read", broken_read)
    elif failure == "drawings":
        path = layer_path(root, "a")
        path.parent.mkdir()
        path.write_text("{broken", encoding="utf-8")
    changed = False

    def progress(done: int, total: int) -> None:
        nonlocal changed
        if failure.startswith("changed-") and not changed:
            path = root / ("a_1.avi" if failure == "changed-source" else "b_2.avi")
            with path.open("ab") as stream:
                stream.write(b"changed")
            changed = True

    out = tmp_path / "comparison.avi"
    with pytest.raises((ValueError, OSError), match="."):
        export_comparison_video(
            root,
            "a",
            asset,
            registration(asset),
            ComparisonLayer(),
            out,
            ComparisonVideoExportOptions(progress=progress),
        )
    assert not out.exists()
    assert not out.with_suffix(".json").exists()


@pytest.mark.parametrize("speed", [0.0, -1.0, float("nan"), float("inf"), 2.0])
def test_invalid_export_speed_is_rejected(tmp_path: Path, speed: float) -> None:
    root = _bundle(tmp_path)
    asset = video_asset(root)
    with pytest.raises(ValueError, match="speed"):
        export_comparison_video(
            root,
            "a",
            asset,
            registration(asset),
            ComparisonLayer(),
            tmp_path / "bad.avi",
            ComparisonVideoExportOptions(speed=speed),
        )


def test_visible_motion_export_requires_camera(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    asset = synthetic_motion()
    reg = ReferenceRegistration(reference_id=asset.id, calibration_id="manual")
    with pytest.raises(ValueError, match="camera"):
        export_comparison_video(
            root, "a", asset, reg, ComparisonLayer(), tmp_path / "bad.avi"
        )


@pytest.mark.parametrize("when", ["before", "during"])
def test_changed_camera_evidence_never_publishes(tmp_path: Path, when: str) -> None:
    root = _bundle(tmp_path)
    asset = synthetic_motion()
    snapshot = CameraSnapshot.from_calibration(
        two_camera_rig()[0].to_calibration(),
        provenance="Session reconstruction fixture",
    )
    snapshot = snapshot.model_validate(
        snapshot.model_dump()
        | {
            "camera_id": "a",
            "image_size_px": (64, 48),
            "matrix": ((40, 0, 32), (0, 40, 24), (0, 0, 1)),
        }
    )
    reg = ReferenceRegistration(reference_id=asset.id, calibration_id="manual").bound(
        asset, snapshot, ViewClock(view="a")
    )
    path = root / "reconstruct" / "reconstruction.json"
    path.parent.mkdir()
    record = snapshot.record().to_dict()
    path.write_text(json.dumps({"cameras": [record]}), encoding="utf-8")

    def change(done: int, total: int) -> None:
        record["intrinsics"]["matrix"][0][0] = 41
        path.write_text(json.dumps({"cameras": [record]}), encoding="utf-8")

    if when == "before":
        change(0, 0)
    out = tmp_path / "stale.avi"
    with pytest.raises(ValueError, match="camera changed"):
        export_comparison_video(
            root,
            "a",
            asset,
            reg,
            ComparisonLayer(),
            out,
            ComparisonVideoExportOptions(camera=snapshot.record(), progress=change),
        )
    assert not out.exists() and not out.with_suffix(".json").exists()


def test_failed_encoded_verification_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _bundle(tmp_path)
    asset = video_asset(root)
    original_read = VideoReader.read

    def broken_encoded(self: VideoReader, index: int) -> np.ndarray | None:
        if self.path.name == "encoded.avi" and index == 3:
            return None
        return original_read(self, index)

    monkeypatch.setattr(VideoReader, "read", broken_encoded)
    out = tmp_path / "encoded.avi"
    with pytest.raises(ValueError, match="encoded frame"):
        export_comparison_video(
            root, "a", asset, registration(asset), ComparisonLayer(), out
        )
    assert not out.exists() and not out.with_suffix(".json").exists()


@pytest.mark.parametrize(
    "matrix",
    [
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, -32.0)),
    ],
)
def test_invalid_homography_rejects_export(tmp_path: Path, matrix: tuple) -> None:
    root = _bundle(tmp_path)
    asset = video_asset(root)
    reg = registration(asset).model_copy(update={"image_transform_2d": matrix})
    out = tmp_path / "invalid.avi"
    with pytest.raises(ValueError, match="alignment"):
        export_comparison_video(root, "a", asset, reg, ComparisonLayer(), out)
    assert not out.exists()


def test_cancellation_during_render_publishes_nothing(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    asset = video_asset(root)
    cancelled = False

    def progress(done: int, total: int) -> None:
        nonlocal cancelled
        cancelled = done >= 2

    out = tmp_path / "cancelled.avi"
    with pytest.raises(InterruptedError):
        export_comparison_video(
            root,
            "a",
            asset,
            registration(asset),
            ComparisonLayer(),
            out,
            ComparisonVideoExportOptions(
                cancelled=lambda: cancelled, progress=progress
            ),
        )
    assert not out.exists() and not out.with_suffix(".json").exists()


@pytest.mark.ui
def test_preview_export_pixels_recipe_and_decoder_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app()
    root = _bundle(tmp_path)
    asset = video_asset(root)
    library = ReferenceLibrary(tmp_path / "references")
    library.save(asset)
    save_edits(
        root,
        SessionEdits(
            views={
                "a": ViewEdit(
                    first=1, last=4, crop=CropRect(x=4, y=2, width=55, height=43)
                )
            }
        ),
    )
    drawings = DrawingLayer(
        view="a",
        width=64,
        height=48,
        frames=6,
        shapes=(Drawing(kind="line", start=(8, 32), end=(53, 35), colour="#ff0000"),),
    )
    save_layer(root, drawings)
    # Real session loader discovers these source-pixel detector observations.
    (root / "observations").mkdir()
    observations = {
        "fps": 30,
        "detector_layout": {"keypoint_names": ["wrist"]},
        "frames": [
            {"time_s": i / 30, "keypoints_px": [[22, 32]], "confidence": [1.0]}
            for i in range(6)
        ],
    }
    (root / "observations" / "a.json").write_text(
        json.dumps(observations), encoding="utf-8"
    )
    (root / "observations" / "observations.json").write_text(
        json.dumps(
            {
                "views": [{"view": "a", "status": "available", "file": "a.json"}],
            }
        ),
        encoding="utf-8",
    )
    opens = []
    original_init = VideoReader.__init__

    def opened(self: VideoReader, path: Path) -> None:
        opens.append(path)
        original_init(self, path)

    monkeypatch.setattr(VideoReader, "__init__", opened)
    dialog = ReferenceComparisonDialog(root, "a", library)
    written = []
    from src.tools.capture_rig.clips import _writer

    class PixelWriter:
        def __init__(self, delegate: cv2.VideoWriter) -> None:
            self.delegate = delegate

        def write(self, frame: np.ndarray) -> None:
            written.append(frame.copy())
            self.delegate.write(frame)

        def release(self) -> None:
            self.delegate.release()

    def writer(path: Path, fps: float, size: tuple[int, int]) -> PixelWriter:
        assert fps == 15 and size == (56, 44)
        return PixelWriter(_writer(path, fps, size))

    try:
        assert dialog._track is not None
        dialog.opacity_spin.setValue(0.5)
        assert dialog.slider.minimum() == 1 and dialog.slider.maximum() == 4
        preview = []
        for index in range(1, 5):
            dialog._show_frame(index)
            preview.append(dialog.canvas._image.copy())
        assert sum(path.name == "b_2.avi" for path in opens) == 1
        assert dialog.save()
        monkeypatch.setattr("src.tools.capture_rig.clips._writer", writer)
        sidecar = export_comparison_video(
            root,
            "a",
            asset,
            dialog._session.registration,
            dialog._session.layer,
            tmp_path / "parity.avi",
            ComparisonVideoExportOptions(speed=0.5),
        )
        assert len(written) == len(preview) == 4
        for actual, expected in zip(written, preview, strict=True):
            np.testing.assert_array_equal(actual, expected)
        assert (
            sum(path.name == "b_2.avi" for path in opens) == 2
        )  # one per owner, not per frame
        assert sidecar["reference_asset"] == asset.model_dump(mode="json")
        assert sidecar["drawings"] == drawings.model_dump(mode="json")
        assert sidecar["selection"] == {"first": 1, "last": 4}
        assert sidecar["padding"] == {"right": 1, "bottom": 1}
        assert sidecar["playback"]["output_frame_times"] == [
            i / 30 for i in range(1, 5)
        ]
        reader = dialog._renderer.reader
        dialog.close()
        assert reader._cached_frame is None
    finally:
        dialog.close()
        app.processEvents()
