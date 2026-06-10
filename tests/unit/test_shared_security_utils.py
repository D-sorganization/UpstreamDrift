"""Unit tests for shared security utilities."""

import socket
import zipfile
from pathlib import Path

import pytest
from src.shared.python.security.security_utils import (
    DOWNLOAD_TIMEOUT_SECONDS,
    download_to_file,
    safe_extract_zip,
    validate_path,
)


def test_validate_path_with_tmp_path(tmp_path: Path) -> None:
    """Test path validation using temporary directories."""
    root = tmp_path / "root"
    root.mkdir()

    # Create a safe file
    safe_file = root / "safe.txt"
    safe_file.touch()

    # Create a file outside
    outside_file = tmp_path / "outside.txt"
    outside_file.touch()

    # Test allowed
    result = validate_path(safe_file, [root])
    assert result == safe_file.resolve()

    # Test disallowed strict
    with pytest.raises(ValueError, match="Path traversal blocked"):
        validate_path(outside_file, [root], strict=True)

    # Test disallowed non-strict
    result_outside = validate_path(outside_file, [root], strict=False)
    assert result_outside == outside_file.resolve()


def test_validate_path_traversal(tmp_path: Path) -> None:
    """Test that path traversal attempts are caught."""
    root = tmp_path / "root"
    root.mkdir()

    # subdir
    subdir = root / "subdir"
    subdir.mkdir()

    # traversal attempt: root/subdir/../../outside.txt
    # This resolves to root/../outside.txt -> tmp_path/outside.txt (which is outside root)

    outside_file = tmp_path / "outside.txt"
    outside_file.touch()

    traversal_path = subdir / ".." / ".." / "outside.txt"

    with pytest.raises(ValueError, match="Path traversal blocked"):
        validate_path(traversal_path, [root], strict=True)


# ---------------------------------------------------------------------------
# safe_extract_zip — Zip Slip regression (issue #7183)
# ---------------------------------------------------------------------------


def _make_zip(path: Path, members: dict[str, str]) -> None:
    """Write *members* (arcname -> content) into a zip at *path*."""
    with zipfile.ZipFile(path, "w") as zf:
        for arcname, content in members.items():
            zf.writestr(arcname, content)


def test_safe_extract_zip_happy_path(tmp_path: Path) -> None:
    """A benign archive extracts normally into the destination."""
    archive = tmp_path / "good.zip"
    _make_zip(archive, {"a.txt": "hello", "sub/b.txt": "world"})
    dest = tmp_path / "out"

    with zipfile.ZipFile(archive, "r") as zf:
        safe_extract_zip(zf, dest)

    assert (dest / "a.txt").read_text() == "hello"
    assert (dest / "sub" / "b.txt").read_text() == "world"


def test_safe_extract_zip_rejects_parent_traversal(tmp_path: Path) -> None:
    """A member named ``../evil.txt`` is rejected and nothing escapes."""
    archive = tmp_path / "evil.zip"
    _make_zip(archive, {"../evil.txt": "pwned"})
    dest = tmp_path / "out"
    dest.mkdir()

    with (
        zipfile.ZipFile(archive, "r") as zf,
        pytest.raises(ValueError, match="Unsafe path in archive"),
    ):
        safe_extract_zip(zf, dest)

    # The traversal target (tmp_path/evil.txt) must NOT have been written.
    assert not (tmp_path / "evil.txt").exists()


def test_safe_extract_zip_rejects_absolute_member(tmp_path: Path) -> None:
    """An absolute member path is rejected."""
    archive = tmp_path / "abs.zip"
    # zipfile normalises away a leading '/', so craft an explicit absolute-ish
    # arcname the validator must still catch via its component check.
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(zipfile.ZipInfo("/etc/evil"), "x")
    dest = tmp_path / "out"

    with (
        zipfile.ZipFile(archive, "r") as zf,
        pytest.raises(ValueError, match="Unsafe path in archive"),
    ):
        safe_extract_zip(zf, dest)


# ---------------------------------------------------------------------------
# download_to_file — bounded timeout regression (issue #7184)
# ---------------------------------------------------------------------------


def test_download_to_file_rejects_nonpositive_timeout(tmp_path: Path) -> None:
    """A non-positive timeout is a precondition violation."""
    with pytest.raises(ValueError, match="timeout must be positive"):
        download_to_file("http://127.0.0.1/x", tmp_path / "out", timeout=0)


def test_download_to_file_times_out_on_unresponsive_socket(tmp_path: Path) -> None:
    """A non-responsive server causes a bounded failure, not an infinite hang.

    We bind a listening socket that never ``accept``s, so the client's
    connect/read blocks; the short timeout must abort it quickly.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    # Backlog 0 and no accept() => connections stall in the kernel queue.
    server.listen(0)
    host, port = server.getsockname()
    url = f"http://{host}:{port}/never"
    try:
        with pytest.raises((TimeoutError, OSError)):
            download_to_file(url, tmp_path / "out", timeout=1)
    finally:
        server.close()


def test_download_timeout_constant_is_positive() -> None:
    """The shared default timeout constant is a sane positive bound."""
    assert isinstance(DOWNLOAD_TIMEOUT_SECONDS, (int, float))
    assert DOWNLOAD_TIMEOUT_SECONDS > 0
