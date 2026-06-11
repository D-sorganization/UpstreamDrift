"""Tests for the optional-dependency skip helper (issue #7158 D1)."""

from __future__ import annotations

import pytest

from tests.support.optional_deps import missing_is_optional, skip_unless_optional

pytestmark = pytest.mark.unit


def test_missing_is_optional_matches_top_level_package() -> None:
    exc = ImportError("No module named 'fastapi.testclient'", name="fastapi.testclient")
    assert missing_is_optional(exc, allowed={"fastapi"}) is True


def test_missing_is_optional_rejects_unlisted_module() -> None:
    exc = ImportError("No module named 'src.api.server'", name="src.api.server")
    assert missing_is_optional(exc, allowed={"fastapi", "httpx"}) is False


def test_skip_unless_optional_skips_for_optional_dependency() -> None:
    exc = ImportError("No module named 'httpx'", name="httpx")
    with pytest.raises(pytest.skip.Exception):
        skip_unless_optional(exc, allowed={"httpx"})


def test_skip_unless_optional_returns_for_real_bug() -> None:
    """A non-optional ImportError (a genuine bug) must NOT skip — the helper
    returns so the caller re-raises and collection fails loudly."""
    exc = ImportError(
        "cannot import name 'app' from 'src.api.server'", name="src.api.server"
    )
    # Returns None (does not raise skip) so the caller's `raise` propagates.
    assert skip_unless_optional(exc, allowed={"fastapi", "httpx"}) is None


def test_message_fallback_when_name_unset() -> None:
    exc = ImportError("No module named mujoco")  # no .name attribute set
    assert missing_is_optional(exc, allowed={"mujoco"}) is True
