"""Tests for suite-marker baseline ratchet enforcement (#7272)."""

from __future__ import annotations

import json
from importlib import util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "ci" / "check_suite_marker_ratchet.py"


def _load_module():
    spec = util.spec_from_file_location("suite_marker_ratchet_under_test", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load suite marker ratchet script")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_scan_honors_module_class_and_function_markers(tmp_path, monkeypatch):
    mod = _load_module()
    repo_root = tmp_path
    tests_root = repo_root / "tests"
    tests_root.mkdir()
    test_file = tests_root / "test_sample.py"
    test_file.write_text(
        "\n".join(
            [
                "import pytest",
                "",
                "pytestmark = pytest.mark.unit",
                "",
                "def test_module_marked():",
                "    pass",
                "",
                "class TestMarkedByModule:",
                "    def test_method(self):",
                "        pass",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "REPO_ROOT", repo_root)
    monkeypatch.setattr(mod, "TESTS_ROOT", tests_root)

    assert mod.collect_unmarked_nodeids() == []


def test_static_scan_uses_configured_pytest_testpaths(tmp_path, monkeypatch):
    mod = _load_module()
    repo_root = tmp_path
    tests_root = repo_root / "tests"
    colocated_root = repo_root / "src" / "pkg" / "tests"
    tests_root.mkdir(parents=True)
    colocated_root.mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[tool.pytest.ini_options]",
                'testpaths = ["tests", "src/pkg/tests"]',
            ]
        ),
        encoding="utf-8",
    )
    (tests_root / "test_marked.py").write_text(
        "import pytest\n\npytestmark = pytest.mark.unit\n\n"
        "def test_marked():\n    pass\n",
        encoding="utf-8",
    )
    (colocated_root / "test_unmarked.py").write_text(
        "def test_colocated_needs_marker():\n    pass\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "REPO_ROOT", repo_root)

    assert mod.collect_unmarked_nodeids() == [
        "src/pkg/tests/test_unmarked.py::test_colocated_needs_marker"
    ]


def test_static_scan_accepts_relative_roots_from_repo_root(tmp_path, monkeypatch):
    mod = _load_module()
    repo_root = tmp_path
    src_root = repo_root / "src" / "pkg" / "tests"
    src_root.mkdir(parents=True)
    (src_root / "test_unmarked.py").write_text(
        "def test_relative_root_needs_marker():\n    pass\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "REPO_ROOT", repo_root)
    monkeypatch.chdir(repo_root)

    assert mod.collect_unmarked_nodeids(Path("src")) == [
        "src/pkg/tests/test_unmarked.py::test_relative_root_needs_marker"
    ]


def test_ratchet_fails_for_new_unmarked_test(tmp_path, monkeypatch, capsys):
    mod = _load_module()
    repo_root = tmp_path
    tests_root = repo_root / "tests"
    config_root = repo_root / "scripts" / "config"
    tests_root.mkdir(parents=True)
    config_root.mkdir(parents=True)
    baseline_path = config_root / "suite_marker_baseline.json"
    baseline_path.write_text(
        json.dumps({"unmarked_nodeids": []}),
        encoding="utf-8",
    )
    (tests_root / "test_new.py").write_text(
        "def test_needs_marker():\n    pass\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "REPO_ROOT", repo_root)
    monkeypatch.setattr(mod, "TESTS_ROOT", tests_root)
    monkeypatch.setattr(mod, "BASELINE_PATH", baseline_path)

    rc = mod.main([])
    captured = capsys.readouterr()

    assert rc == 1
    assert "Suite-marker ratchet FAILED" in captured.err
    assert "tests/test_new.py::test_needs_marker" in captured.err


def test_ratchet_passes_when_unmarked_test_is_in_baseline(tmp_path, monkeypatch):
    mod = _load_module()
    repo_root = tmp_path
    tests_root = repo_root / "tests"
    config_root = repo_root / "scripts" / "config"
    tests_root.mkdir(parents=True)
    config_root.mkdir(parents=True)
    baseline_path = config_root / "suite_marker_baseline.json"
    nodeid = "tests/test_legacy.py::test_legacy_unmarked"
    baseline_path.write_text(
        json.dumps({"unmarked_nodeids": [nodeid]}),
        encoding="utf-8",
    )
    (tests_root / "test_legacy.py").write_text(
        "def test_legacy_unmarked():\n    pass\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "REPO_ROOT", repo_root)
    monkeypatch.setattr(mod, "TESTS_ROOT", tests_root)
    monkeypatch.setattr(mod, "BASELINE_PATH", baseline_path)

    assert mod.main([]) == 0
