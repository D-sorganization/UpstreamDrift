"""Tests for the suite-marker classification helpers (issue #7158, defect 2).

The root ``tests/conftest.py`` consumes these helpers in a
``pytest_collection_modifyitems`` hook to report (report-only by default) or
fail (when ``UD_ENFORCE_SUITE_MARKERS`` is set) on tests that carry none of the
recognized suite markers.
"""

from __future__ import annotations

import pytest

from tests.support.suite_markers import (
    SUITE_MARKERS,
    find_unmarked,
    item_has_suite_marker,
    suite_markers_enforced,
)

pytestmark = pytest.mark.unit


class _FakeMarker:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeItem:
    """Minimal stand-in for a pytest Item exposing markers + nodeid."""

    def __init__(self, nodeid: str, marker_names: list[str]) -> None:
        self.nodeid = nodeid
        self._markers = [_FakeMarker(n) for n in marker_names]

    def iter_markers(self) -> list[_FakeMarker]:
        return list(self._markers)


def test_suite_markers_set_includes_core_lanes() -> None:
    for expected in ("unit", "integration", "e2e", "slow"):
        assert expected in SUITE_MARKERS


def test_enforced_flag_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UD_ENFORCE_SUITE_MARKERS", raising=False)
    assert suite_markers_enforced() is False
    monkeypatch.setenv("UD_ENFORCE_SUITE_MARKERS", "1")
    assert suite_markers_enforced() is True
    monkeypatch.setenv("UD_ENFORCE_SUITE_MARKERS", "true")
    assert suite_markers_enforced() is True
    monkeypatch.setenv("UD_ENFORCE_SUITE_MARKERS", "0")
    assert suite_markers_enforced() is False


def test_item_has_suite_marker() -> None:
    assert item_has_suite_marker(_FakeItem("x::test", ["unit"])) is True
    # requires_gl is a capability marker, not a suite marker.
    assert item_has_suite_marker(_FakeItem("x::test", ["requires_gl"])) is False
    assert item_has_suite_marker(_FakeItem("x::test", [])) is False


def test_find_unmarked_identifies_only_unmarked() -> None:
    items = [
        _FakeItem("tests/test_a.py::test_marked", ["unit"]),
        _FakeItem("tests/test_b.py::test_unmarked", []),
        _FakeItem("tests/test_c.py::test_cap_only", ["requires_gl"]),
    ]
    unmarked = find_unmarked(items)
    unmarked_ids = {i.nodeid for i in unmarked}
    assert unmarked_ids == {
        "tests/test_b.py::test_unmarked",
        "tests/test_c.py::test_cap_only",
    }


def test_find_unmarked_empty_when_all_marked() -> None:
    items = [
        _FakeItem("tests/test_a.py::test_marked", ["unit"]),
        _FakeItem("tests/test_b.py::test_slow", ["slow"]),
    ]
    assert find_unmarked(items) == []
