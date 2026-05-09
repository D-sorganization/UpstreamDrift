"""Regression tests for streamed upload enforcement.

These tests lock in the body-size contract for upload handling:
- the shared helper enforces the byte ceiling while reading in chunks
- partial temp files are removed when a limit violation aborts streaming
- the video and data explorer routes reject oversized bodies even when
  Content-Length is absent from the call path
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import BackgroundTasks, HTTPException
from src.api.middleware import upload_limits

pytestmark = pytest.mark.anyio


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    """Use asyncio for async route/helper tests."""
    return "asyncio"


class FakeUploadFile:
    """Minimal async upload stub with chunk-aware reads."""

    def __init__(
        self,
        chunks: Sequence[bytes],
        *,
        filename: str = "upload.bin",
        content_type: str = "application/octet-stream",
    ) -> None:
        self.filename = filename
        self.content_type = content_type
        self._chunks = list(chunks)
        self.read_sizes: list[int] = []
        self.size = None

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if not self._chunks:
            return b""

        if size < 0:
            data = b"".join(self._chunks)
            self._chunks.clear()
            return data

        remaining = size
        output = bytearray()
        while self._chunks and remaining > 0:
            chunk = self._chunks[0]
            if len(chunk) <= remaining:
                output.extend(self._chunks.pop(0))
                remaining -= len(chunk)
            else:
                output.extend(chunk[:remaining])
                self._chunks[0] = chunk[remaining:]
                remaining = 0
        return bytes(output)


class TestUploadStreamingHelper:
    """Helper-level tests for bounded upload streaming."""

    async def test_read_upload_file_bytes_returns_bytes(self) -> None:
        """Chunks under the limit are concatenated in order."""
        upload = FakeUploadFile([b"abc", b"def"])

        content = await upload_limits.read_upload_file_bytes(
            upload, max_bytes=6, chunk_size=3
        )

        assert content == b"abcdef"
        assert upload.read_sizes == [3, 3, 3]

    async def test_iter_upload_file_chunks_rejects_oversized_body(self) -> None:
        """The helper aborts once the accumulated bytes exceed the cap."""
        upload = FakeUploadFile([b"abc", b"def"])

        with pytest.raises(HTTPException) as excinfo:
            async for _ in upload_limits.iter_upload_file_chunks(
                upload,
                max_bytes=5,
                chunk_size=3,
            ):
                pass

        assert excinfo.value.status_code == 413

    async def test_write_upload_file_to_path_removes_partial_file_on_error(
        self, tmp_path: Path
    ) -> None:
        """Oversized uploads do not leave a partial temp file behind."""
        upload = FakeUploadFile([b"abc", b"def"])
        destination = tmp_path / "upload.bin"

        with pytest.raises(HTTPException) as excinfo:
            await upload_limits.write_upload_file_to_path(
                upload, destination, max_bytes=5, chunk_size=3
            )

        assert excinfo.value.status_code == 413
        assert not destination.exists()


class TestUploadRouteContracts:
    """Route-level tests for the upload-bounded read helpers."""

    async def test_import_dataset_rejects_oversized_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Data explorer import must reject a streamed body over the limit."""
        from src.api.routes import data_explorer

        monkeypatch.setattr(upload_limits, "MAX_UPLOAD_SIZE_BYTES", 5)
        data_explorer._loaded_datasets.clear()

        upload = FakeUploadFile(
            [b"a,b\n1,2\n", b"3,4\n"],
            filename="sample.csv",
            content_type="text/csv",
        )

        with pytest.raises(HTTPException) as excinfo:
            await data_explorer.import_dataset(upload)

        assert excinfo.value.status_code == 413
        assert "sample.csv" not in data_explorer._loaded_datasets

    async def test_analyze_video_rejects_oversized_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Video analysis must stop on streamed body size even without Content-Length."""
        from src.api.routes import video as video_module

        monkeypatch.setattr(upload_limits, "MAX_UPLOAD_SIZE_BYTES", 5)
        monkeypatch.setattr(
            video_module,
            "_load_video_pipeline_classes",
            lambda: (_ for _ in ()).throw(
                AssertionError("pipeline load should not run")
            ),
        )

        upload = FakeUploadFile(
            [b"\x00\x00\x00\x20ftypisom", b"more-bytes"],
            filename="swing.mp4",
            content_type="video/mp4",
        )

        with pytest.raises(HTTPException) as excinfo:
            await video_module.analyze_video(
                file=upload,
                video_pipeline=MagicMock(),
                logger=MagicMock(),
            )

        assert excinfo.value.status_code == 413

    async def test_analyze_video_async_rejects_oversized_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Async video analysis must also enforce the stream cap before queuing work."""
        from src.api.routes import video as video_module

        monkeypatch.setattr(upload_limits, "MAX_UPLOAD_SIZE_BYTES", 5)
        monkeypatch.setattr(
            video_module,
            "_load_video_pipeline_classes",
            lambda: (_ for _ in ()).throw(
                AssertionError("pipeline load should not run")
            ),
        )

        upload = FakeUploadFile(
            [b"\x00\x00\x00\x20ftypisom", b"more-bytes"],
            filename="swing.mp4",
            content_type="video/mp4",
        )

        with pytest.raises(HTTPException) as excinfo:
            await video_module.analyze_video_async(
                background_tasks=BackgroundTasks(),
                file=upload,
                video_pipeline=MagicMock(),
                task_manager={},
            )

        assert excinfo.value.status_code == 413
