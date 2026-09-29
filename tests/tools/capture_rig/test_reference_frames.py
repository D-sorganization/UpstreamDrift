"""Reference marks retain their exact original image and capture ownership."""

import numpy as np
import pytest

from src.tools.capture_rig.reference_calibration.frames import archive_frame, load_frame
from src.tools.capture_rig.reference_calibration.frames import verify_frame

pytestmark = pytest.mark.unit


def test_archived_frame_is_independent_of_later_edits(tmp_path) -> None:
    original = np.full((40, 60, 3), 110, dtype=np.uint8)
    saved = archive_frame(
        tmp_path,
        original,
        capture_id="swing-1",
        view="front",
        frame_index=12,
        timestamp_s=0.4,
        source_label="Original Recording",
    )
    original[:] = 0
    restored = load_frame(tmp_path, saved.path, saved.sha256, capture_id="swing-1")
    assert restored.shape == (40, 60, 3)
    assert np.all(restored == 110)
    second = archive_frame(
        tmp_path,
        original,
        capture_id="swing-1",
        view="front",
        frame_index=12,
        timestamp_s=0.4,
        source_label="Original Recording",
    )
    assert second.path != saved.path
    assert np.all(
        load_frame(tmp_path, saved.path, saved.sha256, capture_id="swing-1") == 110
    )


def test_changed_frame_and_wrong_capture_are_rejected(tmp_path) -> None:
    saved = archive_frame(
        tmp_path,
        np.zeros((20, 30, 3), dtype=np.uint8),
        capture_id="one",
        view="front",
        frame_index=0,
        timestamp_s=0,
        source_label="Live Camera",
    )
    with pytest.raises(ValueError, match="another capture"):
        load_frame(tmp_path, saved.path, saved.sha256, capture_id="two")
    (tmp_path / saved.path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_frame(tmp_path, saved.path, saved.sha256, capture_id="one")


@pytest.mark.parametrize("path", ["../outside.png", "C:/outside.png", "frame.png"])
def test_only_owned_frame_paths_can_be_loaded(tmp_path, path) -> None:
    with pytest.raises(ValueError, match="frame path"):
        load_frame(tmp_path, path, "0" * 64, capture_id="one")


def test_saved_evidence_rechecks_bytes_without_requiring_image_decompression(
    tmp_path, monkeypatch
):
    import cv2

    saved = archive_frame(
        tmp_path,
        np.zeros((20, 30, 3), dtype=np.uint8),
        capture_id="one",
        view="front",
        frame_index=0,
        timestamp_s=0,
        source_label="Original Frame",
    )

    def unavailable_decoder(*_args):
        raise AssertionError("Evidence verification does not require a decoder")

    monkeypatch.setattr(cv2, "imdecode", unavailable_decoder)
    assert verify_frame(tmp_path, saved.path, saved.sha256, capture_id="one") == saved
    (tmp_path / saved.path).write_bytes(b"changed after inspection")
    with pytest.raises(ValueError, match="changed"):
        verify_frame(tmp_path, saved.path, saved.sha256, capture_id="one")
