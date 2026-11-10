"""Unit tests for suite marker reach and directory inference (#14947)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.ci.audit_suite_marker_reach import audit_suite_markers, main as audit_main
from tests.helpers.suite_marker_reach import (
    OPT_OUT_MARKER,
    apply_directory_suite_auto_markers,
    infer_directory_suite_marker,
    summarize_gate_selectability,
)
from tests.support.suite_markers import BASELINE_PATH

pytestmark = pytest.mark.unit


class _MockMarker:
    def __init__(self, name: str) -> None:
        self.name = name


class _MockItem:
    def __init__(self, nodeid: str, markers: list[str] | None = None) -> None:
        self.nodeid = nodeid
        self._markers = [_MockMarker(m) for m in (markers or [])]

    def iter_markers(self) -> list[_MockMarker]:
        return list(self._markers)

    def add_marker(self, marker: Any) -> None:
        name = getattr(marker, "name", str(marker))
        self._markers.append(_MockMarker(name))


def test_infer_directory_suite_marker() -> None:
    cases = (
        ("tests/unit/foo/test_bar.py::test_fn", "unit"),
        ("tests\\unit\\sub\\test_baz.py::test_fn", "unit"),
        ("C:/repo/tests/unit/test_a.py", "unit"),
        ("tests/integration/test_sync.py::test_fn", "integration"),
        ("tests\\integration\\test_b.py", "integration"),
        ("tests/smoke/test_smoke.py::test_fn", None),
        ("tests/helpers/test_helper.py", None),
    )
    for path, expected in cases:
        assert infer_directory_suite_marker(path) == expected


def test_apply_directory_suite_auto_markers_marking_and_opt_out() -> None:
    items = [
        _MockItem("tests/unit/test_u.py::t1"),
        _MockItem("tests/integration/test_i.py::t2"),
        _MockItem("tests/unit/test_opt.py::t3", markers=[OPT_OUT_MARKER]),
        _MockItem("tests/unit/test_dup.py::t4", markers=["unit"]),
        _MockItem("tests/smoke/test_s.py::t5"),
    ]
    assert apply_directory_suite_auto_markers(items, enabled=True) == 2
    assert any(m.name == "unit" for m in items[0].iter_markers())
    assert any(m.name == "integration" for m in items[1].iter_markers())
    assert not any(m.name in {"unit", "integration"} for m in items[2].iter_markers())
    assert sum(1 for m in items[3].iter_markers() if m.name == "unit") == 1
    assert not any(m.name in {"unit", "integration"} for m in items[4].iter_markers())


def test_apply_directory_suite_auto_markers_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _MockItem("tests/unit/test_u.py::t1")
    assert apply_directory_suite_auto_markers([item], enabled=False) == 0
    assert not any(m.name == "unit" for m in item.iter_markers())

    monkeypatch.delenv("UD_AUTO_MARK_BY_DIRECTORY", raising=False)
    assert apply_directory_suite_auto_markers([item], enabled=None) == 0

    monkeypatch.setenv("UD_AUTO_MARK_BY_DIRECTORY", "1")
    assert apply_directory_suite_auto_markers([item], enabled=None) == 1
    assert any(m.name == "unit" for m in item.iter_markers())


def test_summarize_gate_selectability() -> None:
    s = summarize_gate_selectability(100, 75)
    assert (s["selected"], s["total"], s["deselected"], s["selected_pct"]) == (
        75,
        100,
        25,
        75.0,
    )
    assert s["summary_line"] == "selected/total: 75/100 (75.0%)"
    s0 = summarize_gate_selectability(0, 0)
    assert (s0["selected"], s0["total"], s0["selected_pct"]) == (0, 0, 0.0)
    assert s0["summary_line"] == "selected/total: 0/0 (0.0%)"


def test_audit_suite_marker_reach_synthetic(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    (tests_dir / "unit").mkdir(parents=True)
    (tests_dir / "integration").mkdir(parents=True)
    (tests_dir / "unit" / "test_1.py").write_text(
        "def test_u1(): pass\ndef test_u2(): pass\n", encoding="utf-8"
    )
    (tests_dir / "integration" / "test_2.py").write_text(
        "import pytest\n@pytest.mark.integration\ndef test_i1(): pass\ndef test_i2(): pass\n",
        encoding="utf-8",
    )
    base_file = tmp_path / "baseline.json"
    base_file.write_text(
        json.dumps(
            {
                "unmarked_nodeids": [
                    "tests/unit/test_1.py::test_u1",
                    "tests/unit/test_1.py::test_u2",
                    "tests/integration/test_2.py::test_i2",
                    "tests/other/test_3.py::test_o1",
                ]
            }
        ),
        encoding="utf-8",
    )
    rep = audit_suite_markers(tests_dir, base_file)
    assert rep["baseline"] == {
        "total": 4,
        "unit": 2,
        "integration": 1,
        "covered_by_auto_marking": 3,
        "remaining_unmarked": 1,
    }
    c = rep["census"]
    assert c["total_tests"] == 4 and c["explicitly_marked"] == 1
    assert c["auto_marked_unit"] == 2 and c["auto_marked_integration"] == 1
    assert c["after_marked"] == 4 and c["target_met"] is True


def test_audit_suite_marker_reach_repo_baseline() -> None:
    if not BASELINE_PATH.exists():
        pytest.skip("repo suite_marker_baseline.json not found")
    rep = audit_suite_markers(
        Path(__file__).resolve().parents[3] / "tests", BASELINE_PATH
    )
    assert rep["baseline"]["unit"] > 10000 and rep["baseline"]["integration"] > 200
    assert (
        rep["census"]["after"]["selected_pct"] >= 85.0
        and rep["census"]["target_met"] is True
    )


def test_audit_suite_marker_reach_cli_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tests_dir = tmp_path / "tests" / "unit"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_s.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    assert audit_main(["--tests-root", str(tests_dir.parent), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert "baseline" in data and data["census"]["total_tests"] == 1
