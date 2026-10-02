"""Actual archive review, independent of GUI and estimation dependencies."""

import json
from pathlib import Path
from zipfile import ZipFile
import pytest
from src.shared.python.workspace import NecromatcherLibrary, compute_file_sha256
from src.shared.python.workspace.necromatcher_review import CaptureReview

pytestmark = pytest.mark.unit


def test_review_keeps_missingness_pts_and_original_png(tmp_path):
    library = NecromatcherLibrary.create(tmp_path / "library")
    library.add_player("hogan", "Ben Hogan")
    library.add_swing("practice", "hogan", "Practice")
    image = b"synthetic-test-only-png-bytes"
    row = {
        "image": "frame-7.png",
        "frame": {
            "pts_ticks": 7,
            "timebase_numerator": 1,
            "timebase_denominator": 30,
            "physical_time_s": None,
        },
        "observation": {"status": "missing", "landmarks": {}},
    }
    archive_path = tmp_path / "fixture.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "receipt.json",
            json.dumps(
                {
                    "source": {"width_px": 320, "height_px": 240},
                    "frame_count": 1,
                    "qualification": "image_observations_only",
                }
            ),
        )
        archive.writestr("observations.jsonl", json.dumps(row) + "\n")
        archive.writestr("frame-7.png", image)
    # Deliberately inject a declared unit fixture through the shared public store:
    # this exercises read/recall, not capture-image qualification.
    from src.shared.python.workspace import SessionProjectStore

    SessionProjectStore(library.root).register_dataset(
        "capture",
        "practice",
        archive_path,
        "image_capture",
        metadata={
            "hash": compute_file_sha256(archive_path),
            "schema": "necromatcher/image-capture/1",
        },
    )
    with CaptureReview(library, "capture") as review:
        preview = review.frame(0)
        assert preview["frame"]["pts_ticks"] == 7
        assert preview["frame"]["physical_time_s"] is None
        assert preview["observation"]["landmarks"] == {}
        assert preview["image_width"] == 320
        assert review.image(0) == image
        with pytest.raises(IndexError):
            review.frame(1)
        with pytest.raises(IndexError):
            review.image(-1)
    archive_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        CaptureReview(library, "capture")


def test_default_library_uses_configured_root_and_reopens(tmp_path, monkeypatch):
    from src.shared.python.workspace.necromatcher import default_necromatcher_library

    root = tmp_path / "configured"
    monkeypatch.setenv("NECROMATCHER_LIBRARY_ROOT", str(root))
    default_necromatcher_library.cache_clear()
    library = default_necromatcher_library()
    library.add_player("hogan", "Ben Hogan")
    default_necromatcher_library.cache_clear()
    assert default_necromatcher_library().players()[0].display_name == "Ben Hogan"
    assert default_necromatcher_library().root == root.resolve()
    default_necromatcher_library.cache_clear()
