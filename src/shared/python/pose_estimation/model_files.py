"""Verified downloads of pinned model files (shared by the estimator resolvers).

A model file is fetched from a pinned origin into a temporary file, checked
against its expected size and SHA-256, and only then moved into place; a
mismatch leaves nothing behind and raises :class:`ModelError`. Estimator
modules describe *what* to fetch; this module owns *how*.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

from src.shared.python.core.contracts import require
from src.shared.python.core.error_utils import ModelError
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

_CHUNK_BYTES = 1 << 20

Opener = Callable[[str], Iterable[bytes]]


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_cache_dir() -> Path:
    """``~/.cache/upstreamdrift/models`` (created on demand by callers)."""
    return Path.home() / ".cache" / "upstreamdrift" / "models"


def https_opener(allowed_origins: tuple[str, ...]) -> Opener:
    """An opener that streams only from the given HTTPS origins."""
    require(
        all(o.startswith("https://") for o in allowed_origins),
        "allowed origins must be https",
        allowed_origins,
    )

    def open_url(url: str) -> Iterable[bytes]:
        require(
            url.startswith(allowed_origins), "model URL must be on a pinned origin", url
        )
        import requests

        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()
        return response.iter_content(chunk_size=_CHUNK_BYTES)

    return open_url


def verify_file(path: Path, *, sha256: str | None, size_bytes: int) -> bool:
    """True when ``path`` exists with the expected size and (if pinned) digest."""
    if not path.is_file() or path.stat().st_size != size_bytes:
        return False
    return sha256 is None or sha256_of(path) == sha256


def fetch_verified(
    url: str,
    target: Path,
    *,
    sha256: str | None,
    size_bytes: int,
    opener: Opener,
) -> Path:
    """Download ``url`` to ``target`` and verify it before installing.

    ``sha256=None`` means the digest is not yet pinned: the size is still
    enforced and the actual digest is logged so it can be pinned. Postcondition:
    the returned path satisfies :func:`verify_file`.
    """
    require(size_bytes > 0, "size_bytes must be positive", size_bytes)
    if verify_file(target, sha256=sha256, size_bytes=size_bytes):
        logger.info("model file already present at %s", target)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s from %s", target.name, url)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        for chunk in opener(url):
            tmp.write(chunk)
    actual_size = tmp_path.stat().st_size
    actual = sha256_of(tmp_path)
    problem = None
    if actual_size != size_bytes:
        problem = f"size {actual_size} != {size_bytes}"
    elif sha256 is not None and actual != sha256:
        problem = f"sha256 {actual} != {sha256}"
    if problem:
        tmp_path.unlink(missing_ok=True)
        raise ModelError(
            target.name, "download", details=f"failed verification: {problem}"
        )
    if sha256 is None:
        logger.warning("%s downloaded unpinned; pin sha256 %s", target.name, actual)
    tmp_path.replace(target)
    return target
