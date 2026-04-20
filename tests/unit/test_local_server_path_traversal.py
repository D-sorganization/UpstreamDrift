"""Tests for local server path traversal hardening (#2805).

These tests verify that the local server's logo lookup and SPA static file
helpers refuse traversal, absolute paths, NUL bytes, and symlink escape so
that callers cannot read files outside the intended asset/UI roots.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

local_server = pytest.importorskip("src.api.local_server")


class TestSafeJoin:
    """Unit tests for :func:`_safe_join` path traversal rejection."""

    def test_valid_relative_path_returns_resolved(self, tmp_path: Path) -> None:
        """A simple relative filename resolves under the root."""
        target = tmp_path / "logo.png"
        target.write_bytes(b"ok")
        resolved = local_server._safe_join(tmp_path, "logo.png")
        assert resolved is not None
        assert resolved == target.resolve()

    def test_valid_nested_relative_path(self, tmp_path: Path) -> None:
        """Nested relative paths under the root are allowed."""
        nested = tmp_path / "sub" / "logo.png"
        nested.parent.mkdir()
        nested.write_bytes(b"ok")
        resolved = local_server._safe_join(tmp_path, "sub/logo.png")
        assert resolved is not None
        assert resolved == nested.resolve()

    def test_rejects_parent_traversal(self, tmp_path: Path) -> None:
        """``..`` segments that escape the root are rejected."""
        assert local_server._safe_join(tmp_path, "../secret.txt") is None

    def test_rejects_deep_parent_traversal(self, tmp_path: Path) -> None:
        """Deeply nested traversal is still rejected after normalization."""
        assert local_server._safe_join(tmp_path, "sub/../../escape.txt") is None

    def test_rejects_absolute_posix_path(self, tmp_path: Path) -> None:
        """Absolute POSIX paths are rejected regardless of root."""
        assert local_server._safe_join(tmp_path, "/etc/passwd") is None

    def test_rejects_empty_path(self, tmp_path: Path) -> None:
        """An empty path fragment is refused."""
        assert local_server._safe_join(tmp_path, "") is None

    def test_rejects_nul_byte(self, tmp_path: Path) -> None:
        """NUL byte injection is refused before filesystem access."""
        assert local_server._safe_join(tmp_path, "logo.png\x00.txt") is None

    def test_rejects_symlink_escape(self, tmp_path: Path) -> None:
        """A symlink pointing outside the root must not be served."""
        outside = tmp_path.parent / "outside-secret.txt"
        outside.write_bytes(b"secret")
        root = tmp_path / "root"
        root.mkdir()
        link = root / "escape"
        try:
            os.symlink(outside, link)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks are not supported in this environment")
        assert local_server._safe_join(root, "escape") is None


class TestFindLogoFile:
    """Behavior tests for :func:`_find_logo_file` with unsafe inputs."""

    def test_traversal_returns_none(self) -> None:
        """Traversal attempts should not resolve to any file."""
        assert local_server._find_logo_file("../../etc/passwd") is None

    def test_absolute_path_returns_none(self) -> None:
        """Absolute paths are rejected before the filesystem is checked."""
        assert local_server._find_logo_file("/etc/passwd") is None

    def test_nul_byte_returns_none(self) -> None:
        """NUL byte injection is rejected."""
        assert local_server._find_logo_file("logo\x00.png") is None
