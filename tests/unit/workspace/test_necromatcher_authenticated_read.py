"""Real encoded/decoded capture authentication is bounded to one read operation."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import FrozenInstanceError
import os
from pathlib import Path
from typing import Any

import pytest

from hypothesis_fixture import imported_capture
from src.shared.python.workspace import NecromatcherLibrary
from src.shared.python.workspace.necromatcher_capture_identity import capture_identity

pytestmark = pytest.mark.unit


@pytest.fixture
def captured(fit_case: Any, tmp_path: Path) -> tuple[Any, str]:
    library, _, _ = fit_case
    asset = imported_capture(library, tmp_path / "real-source")
    return library, asset.dataset_id


def mutate_same_stat(path: Path) -> None:
    before = path.stat()
    raw = bytearray(path.read_bytes())
    raw[-10] ^= 1
    path.write_bytes(raw)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def test_full_png_decode_once_then_fresh_after_close(
    captured: Any, monkeypatch: Any
) -> None:
    import cv2

    library, capture_id = captured
    original = cv2.imdecode
    calls = []

    def decode(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, "imdecode", decode)
    expected = capture_identity(library, capture_id).to_record()
    calls.clear()
    with library.authenticated_read():
        first = capture_identity(library, capture_id)
        second = capture_identity(library, capture_id)
        assert first.to_record() == second.to_record() == expected
        detached = first.to_record()
        detached["frames"][0]["frame_id"] = "not-original"
        assert first.to_record() == expected
        with pytest.raises(FrozenInstanceError):
            first.capture_id = "changed"
    assert len(calls) == 3
    capture_identity(library, capture_id)
    assert len(calls) == 6


@pytest.mark.parametrize("reuse", [False, True])
def test_same_stat_capture_mutation_rejects_reuse_or_close(
    captured: Any, reuse: bool
) -> None:
    library, capture_id = captured
    with pytest.raises(ValueError, match="hash|changed"):
        with library.authenticated_read():
            capture_identity(library, capture_id)
            mutate_same_stat(Path(library.load_asset(capture_id).path))
            if reuse:
                capture_identity(library, capture_id)


def test_fresh_registered_asset_metadata_rejects_same_stat_mutation(
    captured: Any,
) -> None:
    library, capture_id = captured
    with pytest.raises(ValueError, match="changed"):
        with library.authenticated_read():
            capture_identity(library, capture_id)
            path = library.root / "project.json"
            before = path.stat()
            raw = path.read_text(encoding="utf-8")
            assert '"source_sha256"' in raw
            # Valid JSON, same byte size, and unchanged capture file/hash.
            raw = raw.replace('"source_sha256"', '"source_sha257"')
            path.write_text(raw, encoding="utf-8")
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            capture_identity(library, capture_id)


def test_nested_context_and_same_root_foreign_library_reject(captured: Any) -> None:
    library, capture_id = captured
    other = NecromatcherLibrary(library.root)
    with library.authenticated_read():
        expected = capture_identity(library, capture_id)
        with pytest.raises(ValueError, match="nested"):
            with library.authenticated_read():
                pass
        with pytest.raises(ValueError, match="library"):
            capture_identity(other, capture_id)
        assert capture_identity(library, capture_id) == expected


def test_copied_thread_context_cannot_use_owner(captured: Any) -> None:
    library, capture_id = captured
    with library.authenticated_read():
        capture_identity(library, capture_id)
        context = copy_context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(context.run, capture_identity, library, capture_id)
            with pytest.raises(ValueError, match="thread|task"):
                future.result()


def test_copied_async_task_cannot_use_owner(captured: Any) -> None:
    library, capture_id = captured

    async def child() -> None:
        capture_identity(library, capture_id)

    async def parent() -> None:
        with library.authenticated_read():
            capture_identity(library, capture_id)
            with pytest.raises(ValueError, match="thread|task"):
                await asyncio.create_task(child())

    asyncio.run(parent())


def test_exception_resets_and_preserves_primary_on_closing_mutation(
    captured: Any,
) -> None:
    library, capture_id = captured
    with pytest.raises(RuntimeError, match="primary") as error:
        with library.authenticated_read():
            capture_identity(library, capture_id)
            mutate_same_stat(Path(library.load_asset(capture_id).path))
            raise RuntimeError("primary")
    assert any("authentication" in note for note in error.value.__notes__)
    with pytest.raises(ValueError, match="hash"):
        capture_identity(library, capture_id)


def test_suppressed_failure_cannot_close_as_success(captured: Any) -> None:
    library, capture_id = captured
    with pytest.raises(ValueError, match="hash|failed"):
        with library.authenticated_read():
            capture_identity(library, capture_id)
            mutate_same_stat(Path(library.load_asset(capture_id).path))
            with pytest.raises(ValueError):
                capture_identity(library, capture_id)


def test_context_cannot_be_reentered_after_close(captured: Any) -> None:
    library, capture_id = captured
    context = library.authenticated_read()
    with context:
        capture_identity(library, capture_id)
    with pytest.raises(ValueError, match="closed|entered"):
        with context:
            pass
