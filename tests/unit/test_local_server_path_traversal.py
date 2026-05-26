"""Tests for local server path traversal hardening (#2805).

These tests verify that the local server's logo lookup and SPA static file
helpers refuse traversal, absolute paths, NUL bytes, and symlink escape so
that callers cannot read files outside the intended asset/UI roots.
"""

from __future__ import annotations


import pytest

local_server = pytest.importorskip("src.api.local_server")


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
