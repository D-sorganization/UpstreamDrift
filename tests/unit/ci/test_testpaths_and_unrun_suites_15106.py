"""Unit tests for #15106: verified unit auto-marking and unrun suite exclusions."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.ci import check_unrun_suite_exclusions as mod
from tests.helpers import suite_marker_reach as smr

pytestmark = pytest.mark.unit


def _item(nodeid: str, *markers: str) -> Any:
    ms = [SimpleNamespace(name=m) for m in markers]
    return SimpleNamespace(
        nodeid=nodeid,
        iter_markers=lambda: list(ms),
        add_marker=lambda m: ms.append(SimpleNamespace(name=m.name)),
    )


def test_verified_unit_prefixes_infer_unit_and_bypass_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for prefix in smr.VERIFIED_UNIT_DIRECTORY_PREFIXES:
        assert smr.infer_directory_suite_marker(f"{prefix}test_x.py::test_fn") == "unit"
    monkeypatch.setenv("UD_AUTO_MARK_BY_DIRECTORY", "1")
    monkeypatch.delenv("UD_AUTO_MARK_INCLUDE_BASELINE", raising=False)
    v = _item("tests/output/a/test_x.py::test_1")
    o = _item("tests/output/a/test_x.py::test_2", smr.OPT_OUT_MARKER)
    u = _item("tests/unit/b.py::test_3")
    base = frozenset({v.nodeid, u.nodeid})
    assert smr.apply_directory_suite_auto_markers([v, o, u], baseline_nodeids=base) == 1
    assert [m.name for m in v.iter_markers()] == ["unit"]
    assert not any(m.name == "unit" for m in (*o.iter_markers(), *u.iter_markers()))


def test_repo_unrun_suite_exclusions_contract_passes() -> None:
    ok, report = mod.check_unrun_suite_exclusions()
    assert ok, report


def test_find_unrun_suite_violations_contract(tmp_path: Path) -> None:
    for rel in ("tests/untracked", "src/pkg/tests", "src/cap/tests"):
        (tmp_path / rel).mkdir(parents=True)
        (tmp_path / rel / "test_x.py").write_text(
            "def test_1(): pass\ndef test_2(): pass\n"
        )
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests", "src/pkg/tests", "src/cap/tests", "src/gone/tests"]\n'
    )
    wf = tmp_path / ".github" / "workflows" / "ci.yml"
    wf.parent.mkdir(parents=True)
    wf.write_text('run: pytest -m "unit" -m "integration"\n')
    cfg = tmp_path / "exclusions.json"
    cfg.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "intree_testpaths": {
                    "src/cap/tests": {"max_unmarked_tests": 1, "reason": "cap"},
                    "src/bad/tests": {"max_unmarked_tests": True, "reason": "bool"},
                },
                "tests_subdirs": {"tests/missing_dir": "stale"},
                "suite_markers": {},
            }
        )
    )
    joined = "\n".join(mod.find_unrun_suite_violations(tmp_path, cfg))
    for w in ("untracked", "pkg", "gone", "missing_dir", "2 > 1", "Invalid"):
        assert w in joined
