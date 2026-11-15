"""Regression for pytest xdist INTERNALERROR when ``os.name`` is patched (#15103)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]
pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONFTEST = ROOT / "conftest.py"

_HOST_OS_NAME = os.name
_FAKE_OS_NAME = "nt" if _HOST_OS_NAME != "nt" else "posix"


def _isolated_repo(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(CONFTEST.read_text(encoding="utf-8"))


@pytest.mark.unit
def test_unmonkeypatched_os_name_does_not_leak_to_next_test(
    pytester: pytest.Pytester,
) -> None:
    _isolated_repo(pytester)
    pytester.makepyfile(
        test_leak=f"""
import os

def test_mutates_name():
    os.name = {_FAKE_OS_NAME!r}

def test_host_name_restored():
    assert os.name == {_HOST_OS_NAME!r}
"""
    )
    result = pytester.runpytest_inprocess("-q")
    combined = "\n".join(result.outlines + result.errlines)
    assert "INTERNALERROR" not in combined, combined
    result.assert_outcomes(passed=2)


@pytest.mark.unit
def test_failing_test_under_os_name_patch_does_not_hit_internalerror(
    pytester: pytest.Pytester,
) -> None:
    _isolated_repo(pytester)
    pytester.makepyfile(
        test_fail=f"""
import os

import pytest


@pytest.fixture
def patch_os_name(monkeypatch):
    monkeypatch.setattr(os, "name", {_FAKE_OS_NAME!r})


def test_fail_under_patch(patch_os_name):
    assert False, "expected failure"
"""
    )
    result = pytester.runpytest_inprocess("-q")
    combined = "\n".join(result.outlines + result.errlines)
    assert "INTERNALERROR" not in combined, combined
    if _HOST_OS_NAME != "nt":
        assert "WindowsPath" not in combined, combined
    if _HOST_OS_NAME != "posix":
        assert "PosixPath" not in combined, combined
    result.assert_outcomes(failed=1)