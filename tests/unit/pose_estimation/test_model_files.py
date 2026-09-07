"""Verified model downloads and the OpenPose file specs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.shared.python.core.error_utils import ModelError
from src.shared.python.pose_estimation import openpose_models
from src.shared.python.pose_estimation.model_files import (
    fetch_verified,
    https_opener,
    verify_file,
)

pytestmark = pytest.mark.unit

PAYLOAD = b"weights" * 100
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def _opener(payload: bytes):
    calls: list[str] = []

    def open_url(url: str):
        calls.append(url)
        yield payload[: len(payload) // 2]
        yield payload[len(payload) // 2 :]

    open_url.calls = calls  # type: ignore[attr-defined]
    return open_url


def test_fetch_verified_installs_only_after_size_and_digest_match(
    tmp_path: Path,
) -> None:
    target = tmp_path / "m.bin"
    out = fetch_verified(
        "https://x/m",
        target,
        sha256=DIGEST,
        size_bytes=len(PAYLOAD),
        opener=_opener(PAYLOAD),
    )
    assert out == target and verify_file(target, sha256=DIGEST, size_bytes=len(PAYLOAD))
    assert not list(tmp_path.glob("tmp*"))


def test_fetch_verified_rejects_bad_size_and_bad_digest(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="size"):
        fetch_verified(
            "https://x/m",
            tmp_path / "a",
            sha256=None,
            size_bytes=1,
            opener=_opener(PAYLOAD),
        )
    with pytest.raises(ModelError, match="sha256"):
        fetch_verified(
            "https://x/m",
            tmp_path / "b",
            sha256="0" * 64,
            size_bytes=len(PAYLOAD),
            opener=_opener(PAYLOAD),
        )
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_fetch_verified_is_idempotent_and_unpinned_downloads_are_allowed(
    tmp_path: Path,
) -> None:
    target = tmp_path / "m.bin"
    opener = _opener(PAYLOAD)
    fetch_verified(
        "https://x/m", target, sha256=None, size_bytes=len(PAYLOAD), opener=opener
    )
    fetch_verified(
        "https://x/m", target, sha256=None, size_bytes=len(PAYLOAD), opener=opener
    )
    assert len(opener.calls) == 1  # type: ignore[attr-defined]


def test_https_opener_refuses_other_origins() -> None:
    open_url = https_opener(("https://good.example/",))
    with pytest.raises(Exception, match="pinned origin"):
        open_url("https://evil.example/x")
    with pytest.raises(Exception, match="https"):
        https_opener(("http://plain.example/",))


def test_body25_download_and_resolve_round_trip(tmp_path: Path) -> None:
    files = openpose_models.BODY25_FILES

    def opener(url: str):
        spec = next(s for s in files.values() if s.url == url)
        yield b"\0" * spec.size_bytes

    prototxt, weights = openpose_models.download_body25(tmp_path, opener=opener)
    assert prototxt.name.endswith(".prototxt") and weights.name.endswith(".caffemodel")
    assert openpose_models.resolve_body25(tmp_path) == (prototxt, weights)
    with pytest.raises(ModelError, match="openpose_models"):
        openpose_models.resolve_body25(tmp_path / "empty")


def test_body25_urls_are_on_the_allowed_origins() -> None:
    for spec in openpose_models.BODY25_FILES.values():
        assert spec.url.startswith(openpose_models.ALLOWED_ORIGINS)
        assert spec.size_bytes > 0
