"""Unit tests for capture-O video acquisition receipt (COV-1, #11269).

Tests adhere strictly to TDD, DbC, and verify fail-closed acquisition contracts
using synthetic video fixtures and mock ffprobe probes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from src.motion_capture.acquisition_receipt import (
    AcquisitionEntry,
    AcquisitionError,
    AcquisitionReceipt,
    EmptyDirectoryError,
    FFProbeError,
    HashMismatchError,
    NonVideoFileError,
    VideoAcquisitionRejection,
    build_acquisition_receipt,
    verify_acquisition_receipt,
)

pytestmark = pytest.mark.unit


def _mock_probe_fn(path: Path) -> dict[str, Any]:
    """Synthetic ffprobe result for testing without external binaries."""
    if "corrupt" in path.name:
        raise FFProbeError(f"Mock ffprobe failed on {path.name}", filename=path.name)
    if path.suffix not in {".mp4", ".mov", ".m4v"}:
        return {
            "streams": [{"codec_type": "audio"}],
            "format": {"filename": path.name, "format_name": "mp3"},
        }
    return {
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30/1",
                "avg_frame_rate": "30/1",
                "tags": {
                    "creation_time": "2019-05-18T14:22:10.000000Z",
                },
            }
        ],
        "format": {
            "filename": str(
                path.resolve()
            ),  # Simulate ffprobe returning an absolute path!
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": "10.000000",
            "size": str(path.stat().st_size),
            "tags": {
                "creation_time": "2019-05-18T14:22:10.000000Z",
            },
        },
    }


class TestCaptureOVideoAcquisition:
    """TDD behavioral unit tests for capture-O video acquisition receipt."""

    def test_empty_directory_raises_value_error(self, tmp_path: Path) -> None:
        """An empty directory raises ValueError (specifically EmptyDirectoryError)."""
        empty_dir = tmp_path / "empty_originals"
        empty_dir.mkdir()
        with pytest.raises(ValueError) as exc_info:
            build_acquisition_receipt(empty_dir, probe_fn=_mock_probe_fn)
        assert isinstance(exc_info.value, EmptyDirectoryError)

    def test_non_video_file_raises_typed_rejection_naming_file(
        self, tmp_path: Path
    ) -> None:
        """A non-video file raises a typed rejection naming the file and is never skipped."""
        originals_dir = tmp_path / "originals"
        originals_dir.mkdir()
        (originals_dir / "valid_clip.mp4").write_bytes(b"dummy video 1 bytes")
        bad_file = originals_dir / "notes.txt"
        bad_file.write_text("not a video")

        with pytest.raises(VideoAcquisitionRejection) as exc_info:
            build_acquisition_receipt(originals_dir, probe_fn=_mock_probe_fn)
        assert isinstance(exc_info.value, NonVideoFileError)
        assert exc_info.value.filename == "notes.txt"
        assert "notes.txt" in str(exc_info.value)

    def test_ffprobe_failure_raises_typed_rejection_naming_file(
        self, tmp_path: Path
    ) -> None:
        """An ffprobe failure raises a typed rejection naming the file and is never skipped."""
        originals_dir = tmp_path / "originals"
        originals_dir.mkdir()
        (originals_dir / "clip_corrupt.mp4").write_bytes(b"corrupt header bytes")

        with pytest.raises(VideoAcquisitionRejection) as exc_info:
            build_acquisition_receipt(originals_dir, probe_fn=_mock_probe_fn)
        assert isinstance(exc_info.value, FFProbeError)
        assert exc_info.value.filename == "clip_corrupt.mp4"
        assert "clip_corrupt.mp4" in str(exc_info.value)

    def test_receipt_contains_no_absolute_local_paths(self, tmp_path: Path) -> None:
        """The receipt contains no absolute local paths anywhere in its content."""
        originals_dir = tmp_path / "originals"
        originals_dir.mkdir()
        (originals_dir / "20190518_142210.mp4").write_bytes(b"synthetic video payload")
        receipt_path = tmp_path / "acquisition_receipt.json"

        receipt = build_acquisition_receipt(
            originals_dir,
            output_path=receipt_path,
            probe_fn=_mock_probe_fn,
        )

        receipt_raw = receipt_path.read_text(encoding="utf-8")
        # Ensure no Windows or Unix absolute drive / root paths are embedded
        assert str(tmp_path) not in receipt_raw
        assert str(originals_dir) not in receipt_raw
        assert ":\\" not in receipt_raw
        assert ":/" not in receipt_raw

        # Also inspect structured entry
        entry = receipt.entries[0]
        assert entry.original_filename == "20190518_142210.mp4"
        format_dict = entry.ffprobe.get("format", {})
        assert format_dict.get("filename") == "20190518_142210.mp4"

    def test_determinism_rerun_produces_byte_identical_receipt(
        self, tmp_path: Path
    ) -> None:
        """Re-running on an unchanged directory produces a byte-identical receipt."""
        originals_dir = tmp_path / "originals"
        originals_dir.mkdir()
        (originals_dir / "clip_a.mp4").write_bytes(b"video a bytes")
        (originals_dir / "clip_b.mp4").write_bytes(b"video b bytes")
        receipt_path = tmp_path / "acquisition_receipt.json"

        build_acquisition_receipt(
            originals_dir,
            output_path=receipt_path,
            downloaded_at="2026-10-02T12:00:00Z",
            probe_fn=_mock_probe_fn,
        )
        first_bytes = receipt_path.read_bytes()

        # Re-run without specifying downloaded_at; existing metadata should be preserved
        build_acquisition_receipt(
            originals_dir,
            output_path=receipt_path,
            probe_fn=_mock_probe_fn,
        )
        second_bytes = receipt_path.read_bytes()

        assert first_bytes == second_bytes

    def test_file_bytes_change_after_first_receipt_raises_hash_mismatch(
        self, tmp_path: Path
    ) -> None:
        """A file whose bytes change after the first receipt raises a hash mismatch error."""
        originals_dir = tmp_path / "originals"
        originals_dir.mkdir()
        clip = originals_dir / "clip_tamper.mp4"
        clip.write_bytes(b"original pristine bytes")
        receipt_path = tmp_path / "acquisition_receipt.json"

        build_acquisition_receipt(
            originals_dir,
            output_path=receipt_path,
            probe_fn=_mock_probe_fn,
        )

        # Alter bytes in place
        clip.write_bytes(b"tampered modified bytes")

        with pytest.raises(HashMismatchError) as exc_info:
            build_acquisition_receipt(
                originals_dir,
                output_path=receipt_path,
                probe_fn=_mock_probe_fn,
            )
        assert exc_info.value.filename == "clip_tamper.mp4"
        assert isinstance(exc_info.value, ValueError)

        # verify_acquisition_receipt also raises HashMismatchError
        with pytest.raises(HashMismatchError):
            verify_acquisition_receipt(receipt_path, originals_dir)

    def test_album_copy_and_phone_original_linked_by_lineage(
        self, tmp_path: Path
    ) -> None:
        """An album copy and a phone original of the same clip both stay present, linked by lineage."""
        originals_dir = tmp_path / "originals"
        originals_dir.mkdir()
        (originals_dir / "clip_01_phone.mp4").write_bytes(b"phone original bytes")
        (originals_dir / "clip_01_album.mp4").write_bytes(b"album copy bytes")
        receipt_path = tmp_path / "acquisition_receipt.json"

        receipt = build_acquisition_receipt(
            originals_dir,
            output_path=receipt_path,
            lineage_map={"clip_01_album.mp4": "album-copy-of cov-02"},
            owner_recollections={"clip_01_phone.mp4": "Session driver swing 1"},
            excluded_by_owner={"clip_01_album.mp4"},
            probe_fn=_mock_probe_fn,
        )

        assert len(receipt.entries) == 2
        # Sorted order: clip_01_album.mp4 -> cov-01, clip_01_phone.mp4 -> cov-02
        album_entry = receipt.entries[0]
        phone_entry = receipt.entries[1]

        assert album_entry.original_filename == "clip_01_album.mp4"
        assert album_entry.cov_id == "cov-01"
        assert album_entry.lineage == "album-copy-of cov-02"
        assert album_entry.excluded_by_owner is True

        assert phone_entry.original_filename == "clip_01_phone.mp4"
        assert phone_entry.cov_id == "cov-02"
        assert phone_entry.lineage == "original"
        assert phone_entry.owner_recollection == "Session driver swing 1"
        assert phone_entry.excluded_by_owner is False

    def test_sidecar_ffprobe_loaded_without_probe_fn(self, tmp_path: Path) -> None:
        """Sidecar ffprobe JSON files are discovered and embedded cleanly."""
        originals_dir = tmp_path / "capture-O-video" / "originals"
        ffprobe_dir = tmp_path / "capture-O-video" / "ffprobe"
        originals_dir.mkdir(parents=True)
        ffprobe_dir.mkdir(parents=True)

        clip_name = "test_clip.mp4"
        (originals_dir / clip_name).write_bytes(b"sample video bytes")
        sidecar_data = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 3840,
                    "height": 2160,
                    "r_frame_rate": "60/1",
                }
            ],
            "format": {
                "filename": clip_name,
                "format_name": "mov,mp4",
                "tags": {"creation_time": "2019-05-18T15:00:00Z"},
            },
        }
        (ffprobe_dir / f"{clip_name}.json").write_text(
            json.dumps(sidecar_data), encoding="utf-8"
        )

        receipt = build_acquisition_receipt(
            originals_dir,
            output_path=tmp_path / "capture-O-video" / "acquisition_receipt.json",
        )
        assert len(receipt) == 1
        entry = receipt[0]
        assert entry.cov_id == "cov-01"
        assert entry.container_creation_time == "2019-05-18T15:00:00Z"
        assert entry.ffprobe["streams"][0]["width"] == 3840

    def test_receipt_roundtrip_serialization_and_indexing(self, tmp_path: Path) -> None:
        """AcquisitionReceipt supports round-trip serialization and dict/index access."""
        originals_dir = tmp_path / "originals"
        originals_dir.mkdir()
        (originals_dir / "vid.mp4").write_bytes(b"data")
        receipt_path = tmp_path / "receipt.json"

        receipt = build_acquisition_receipt(
            originals_dir,
            output_path=receipt_path,
            probe_fn=_mock_probe_fn,
        )

        # Indexing checks
        assert receipt[0].cov_id == "cov-01"
        assert receipt["cov-01"].original_filename == "vid.mp4"
        assert receipt["vid.mp4"].cov_id == "cov-01"
        assert len(receipt["entries"]) == 1

        # Roundtrip from file
        loaded = AcquisitionReceipt.from_file(receipt_path)
        assert loaded.video_count == 1
        assert loaded.entries[0] == receipt[0]

        # Roundtrip from list of dicts
        list_loaded = AcquisitionReceipt.from_dict([receipt[0].to_dict()])
        assert list_loaded.video_count == 1
        assert list_loaded.entries[0] == receipt[0]

    def test_verify_acquisition_receipt_missing_file_raises(
        self, tmp_path: Path
    ) -> None:
        """verify_acquisition_receipt raises FileNotFoundError when a video is missing."""
        originals_dir = tmp_path / "originals"
        originals_dir.mkdir()
        video_file = originals_dir / "temp.mp4"
        video_file.write_bytes(b"video bytes")
        receipt_path = tmp_path / "receipt.json"

        receipt = build_acquisition_receipt(
            originals_dir,
            output_path=receipt_path,
            probe_fn=_mock_probe_fn,
        )
        # Delete the file
        video_file.unlink()

        with pytest.raises(FileNotFoundError):
            verify_acquisition_receipt(receipt, originals_dir)

    def test_contract_validations_fail_closed(self) -> None:
        """AcquisitionEntry rejects invalid fields under DbC."""
        with pytest.raises(ValueError, match="cov_id"):
            AcquisitionEntry(
                cov_id="invalid-id",
                original_filename="clip.mp4",
                sha256="a" * 64,
                byte_size=100,
                downloaded_at="2026-10-02T12:00:00Z",
                ffprobe={"streams": [{"codec_type": "video"}]},
            )

        with pytest.raises(ValueError, match="lineage"):
            AcquisitionEntry(
                cov_id="cov-01",
                original_filename="clip.mp4",
                sha256="a" * 64,
                byte_size=100,
                downloaded_at="2026-10-02T12:00:00Z",
                lineage="invalid-lineage",
                ffprobe={"streams": [{"codec_type": "video"}]},
            )

        with pytest.raises(ValueError, match="sha256"):
            AcquisitionEntry(
                cov_id="cov-01",
                original_filename="clip.mp4",
                sha256="not-a-sha256",
                byte_size=100,
                downloaded_at="2026-10-02T12:00:00Z",
                ffprobe={"streams": [{"codec_type": "video"}]},
            )
