"""File-level guards shared by every open-fp reader: path, size, digest.

Each guard raises ``ValueError`` naming the offending file.  Bytes are read once
and both hashed and parsed from that single buffer, so a recorded digest always
describes exactly the content that was parsed.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

from src.shared.python.force_plates.open_fp_format import format as fmt

__all__ = ["as_path", "checked_file", "read_limited", "sha256_hex"]


def as_path(value: object, name: str) -> Path:
    """Return ``value`` as a ``Path``; ``TypeError`` for anything but str/PathLike."""
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{name} must be a str or path, got {type(value).__name__}")
    return Path(value)


def checked_file(path: Path, suffix: str, *, swing_id_stem: bool) -> Path:
    """Validate ``path`` as a regular, non-symlink file with ``suffix``.

    With ``swing_id_stem`` the stem must also be a swing id, which excludes path
    tricks such as ``../x.json`` without resolving anything.  The size cap is
    enforced by :func:`read_limited`.
    """
    if path.suffix != suffix:
        raise ValueError(f"{path}: expected a {suffix} file")
    if swing_id_stem and not fmt.ID_PATTERN.match(path.stem):
        raise ValueError(f"{path}: file stem {path.stem!r} is not a swing id")
    if path.is_symlink():
        raise ValueError(f"{path}: symlinks are not followed")
    if not path.is_file():
        raise ValueError(f"{path}: not a regular file")
    return path


def read_limited(path: Path, max_bytes: int) -> bytes:
    """Read the regular, non-symlink file ``path``, at most ``max_bytes`` long.

    The file is opened with ``O_NOFOLLOW`` and the regular-file and size checks
    use that same descriptor, so a swap between check and read cannot redirect
    the read.  ``ValueError`` if it is not a regular file or is too large.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{path}: cannot open as a regular file ({exc})") from exc
    with os.fdopen(descriptor, "rb") as handle:
        status = os.fstat(handle.fileno())
        if not stat.S_ISREG(status.st_mode):
            raise ValueError(f"{path}: not a regular file")
        if status.st_size > max_bytes:
            raise ValueError(f"{path}: file too large (limit {max_bytes} bytes)")
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"{path}: file too large (limit {max_bytes} bytes)")
    return data


def sha256_hex(data: bytes) -> str:
    """Lower-case hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()
