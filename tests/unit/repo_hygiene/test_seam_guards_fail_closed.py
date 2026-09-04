"""Tests ensuring seam guards fail closed when vendor/ud-tools is missing.

Issue #9501: Seam guards silently skip without vendor/ud-tools, so the checks
protecting the 143-file convergence pass by not running.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.helpers.seam_guards import (
    missing_vendor_instructions,
    require_vendor_path,
    seam_tests_allow_skip,
)


def test_seam_tests_allow_skip_defaults_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEAM_TESTS_ALLOW_SKIP", raising=False)
    assert not seam_tests_allow_skip()


def test_seam_tests_allow_skip_recognizes_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEAM_TESTS_ALLOW_SKIP", "1")
    assert seam_tests_allow_skip()

    # Whitespace resilience
    monkeypatch.setenv("SEAM_TESTS_ALLOW_SKIP", "1 ")
    assert seam_tests_allow_skip()

    monkeypatch.setenv("SEAM_TESTS_ALLOW_SKIP", "0")
    assert not seam_tests_allow_skip()


def test_missing_vendor_instructions_includes_workaround() -> None:
    msg = missing_vendor_instructions()
    assert "vendor/ud-tools" in msg
    assert "git submodule update --init" in msg
    assert "git -C ../Tools archive" in msg
    assert "SEAM_TESTS_ALLOW_SKIP=1" in msg


def test_require_vendor_path_returns_existing_path(tmp_path: Path) -> None:
    f = tmp_path / "dummy.txt"
    f.write_text("hello", encoding="utf-8")
    assert require_vendor_path(f) == f

    d = tmp_path / "non_empty_dir"
    d.mkdir()
    (d / "item.txt").write_text("data", encoding="utf-8")
    assert require_vendor_path(d) == d


def test_require_vendor_path_fails_closed_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SEAM_TESTS_ALLOW_SKIP", raising=False)
    monkeypatch.delenv("CI", raising=False)
    missing = tmp_path / "does_not_exist"

    with pytest.raises(AssertionError) as exc_info:
        require_vendor_path(missing)

    assert "The vendored Tools tree is missing" in str(exc_info.value)
    assert "git -C ../Tools archive" in str(exc_info.value)


def test_require_vendor_path_fails_closed_for_empty_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SEAM_TESTS_ALLOW_SKIP", raising=False)
    monkeypatch.delenv("CI", raising=False)
    empty_dir = tmp_path / "empty_vendor"
    empty_dir.mkdir()

    with pytest.raises(AssertionError) as exc_info:
        require_vendor_path(empty_dir)

    assert "The vendored Tools tree is missing" in str(exc_info.value)


def test_require_vendor_path_skips_when_opted_out_outside_ci(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEAM_TESTS_ALLOW_SKIP", "1")
    monkeypatch.delenv("CI", raising=False)
    missing = tmp_path / "does_not_exist"

    with pytest.raises(pytest.skip.Exception) as exc_info:
        require_vendor_path(missing)

    assert "The vendored Tools tree is missing" in str(exc_info.value)


def test_require_vendor_path_fails_closed_in_ci_even_if_opted_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEAM_TESTS_ALLOW_SKIP", "1")
    monkeypatch.setenv("CI", "true")
    missing = tmp_path / "does_not_exist"

    with pytest.raises(AssertionError) as exc_info:
        require_vendor_path(missing)

    assert "The vendored Tools tree is missing" in str(exc_info.value)
