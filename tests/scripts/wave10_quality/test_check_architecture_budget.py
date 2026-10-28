"""Tests for the changed-file architecture budget guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ci import check_architecture_budget as mod


def test_analyze_python_file_reports_long_function_and_parameter_count(
    tmp_path: Path,
) -> None:
    source = "\n".join(
        [
            "def acceptable(a, b):",
            "    return a + b",
            "",
            "def too_large(a, b, c, d):",
            "    total = 0",
            "    total += a",
            "    total += b",
            "    total += c",
            "    total += d",
            "    return total",
        ]
    )
    path = tmp_path / "module.py"
    path.write_text(source, encoding="utf-8")

    findings = mod.analyze_python_file(
        path,
        max_function_lines=4,
        max_parameters=3,
    )

    assert [finding.rule for finding in findings] == [
        "function-lines",
        "parameters",
    ]
    assert {finding.symbol for finding in findings} == {"too_large"}
    assert all(finding.path == path for finding in findings)


def test_analyze_python_file_ignores_self_and_cls_parameters(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text(
        "\n".join(
            [
                "class Builder:",
                "    def build(self, a, b, c):",
                "        return a + b + c",
                "    @classmethod",
                "    def make(cls, a, b, c):",
                "        return cls()",
            ]
        ),
        encoding="utf-8",
    )

    findings = mod.analyze_python_file(
        path,
        max_function_lines=10,
        max_parameters=3,
    )

    assert findings == []


def test_collect_violations_skips_tests_and_configured_exceptions(
    tmp_path: Path,
) -> None:
    production = tmp_path / "src" / "large.py"
    production.parent.mkdir()
    production.write_text(
        "def legacy(a, b, c, d):\n    return a + b + c + d\n",
        encoding="utf-8",
    )
    test_file = tmp_path / "tests" / "test_large.py"
    test_file.parent.mkdir()
    test_file.write_text(
        "def helper(a, b, c, d):\n    return a + b + c + d\n",
        encoding="utf-8",
    )
    config = {
        "max_function_lines": 10,
        "max_parameters": 3,
        "exceptions": [
            {
                "path": "src/large.py",
                "symbol": "legacy",
                "rule": "parameters",
                "owner": "@core",
                "reason": "legacy API tracked in issue #7131",
            }
        ],
    }

    violations = mod.collect_violations(
        repo_root=tmp_path,
        paths=[production, test_file],
        config=config,
    )

    assert violations == []


def test_collect_violations_reports_invalid_exception_metadata(tmp_path: Path) -> None:
    config = {
        "max_function_lines": 10,
        "max_parameters": 3,
        "exceptions": [
            {
                "path": "src/large.py",
                "symbol": "legacy",
                "rule": "parameters",
                "owner": "",
                "reason": "temporary",
            }
        ],
    }

    violations = mod.collect_violations(
        repo_root=tmp_path,
        paths=[],
        config=config,
    )

    assert violations == [
        "Invalid exception entry: {'path': 'src/large.py', 'symbol': 'legacy', "
        "'rule': 'parameters', 'owner': '', 'reason': 'temporary'}"
    ]


@pytest.mark.unit
def test_checked_in_architecture_budget_exceptions_are_valid() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config = mod._load_config(
        repo_root, Path("scripts/config/architecture_budget.json")
    )

    _, invalid = mod._collect_active_exceptions(config)

    assert invalid == []


def test_main_fails_when_changed_production_file_exceeds_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "scripts" / "config" / "architecture_budget.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "max_function_lines": 10,
                "max_parameters": 3,
                "exceptions": [],
            }
        ),
        encoding="utf-8",
    )
    production = tmp_path / "src" / "bad.py"
    production.parent.mkdir()
    production.write_text(
        "def too_many(a, b, c, d):\n    return a + b + c + d\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(mod, "_changed_python_files", lambda repo, base: [production])

    assert mod.main(["--config-path", "scripts/config/architecture_budget.json"]) == 1


@pytest.mark.unit
def test_architecture_budget_full_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full-scan mode (--all) scans all tracked production files against baseline (UD-25)."""
    config_path = tmp_path / "scripts" / "config" / "architecture_budget.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "max_function_lines": 10,
                "max_parameters": 3,
                "exceptions": [],
            }
        ),
        encoding="utf-8",
    )
    baseline_path = (
        tmp_path / "scripts" / "config" / "architecture_budget_baseline.json"
    )
    baseline_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "description": "test baseline",
                "entries": [
                    {
                        "path": "src/legacy.py",
                        "symbol": "grandfathered",
                        "rule": "function-lines",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    legacy = tmp_path / "src" / "legacy.py"
    legacy.parent.mkdir()
    legacy.write_text(
        "def grandfathered(a, b):\n"
        + "".join(f"    x_{i} = {i}\n" for i in range(15))
        + "    return a + b\n",
        encoding="utf-8",
    )
    clean_changed = tmp_path / "src" / "clean.py"
    clean_changed.write_text("def ok():\n    return 1\n", encoding="utf-8")

    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        mod, "_changed_python_files", lambda repo, base: [clean_changed]
    )
    monkeypatch.setattr(
        mod, "_tracked_python_files", lambda repo: [legacy, clean_changed]
    )

    # Changed-only mode ignores unchanged legacy.py
    assert mod.main(["--config-path", "scripts/config/architecture_budget.json"]) == 0

    # Full-scan mode scans legacy.py; with baseline it passes
    assert (
        mod.main(
            [
                "--config-path",
                "scripts/config/architecture_budget.json",
                "--all",
                "--baseline-path",
                "scripts/config/architecture_budget_baseline.json",
            ]
        )
        == 0
    )


@pytest.mark.unit
def test_architecture_budget_full_scan_fails_on_new_violation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full-scan mode fails if any violation is not in the baseline (UD-25 DbC postcondition)."""
    config_path = tmp_path / "scripts" / "config" / "architecture_budget.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "max_function_lines": 10,
                "max_parameters": 3,
                "exceptions": [],
            }
        ),
        encoding="utf-8",
    )
    baseline_path = (
        tmp_path / "scripts" / "config" / "architecture_budget_baseline.json"
    )
    baseline_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "description": "test baseline",
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    production = tmp_path / "src" / "new_bad.py"
    production.parent.mkdir()
    production.write_text(
        "def new_violation(a, b, c, d):\n    return a + b + c + d\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(mod, "_tracked_python_files", lambda repo: [production])

    assert (
        mod.main(
            [
                "--config-path",
                "scripts/config/architecture_budget.json",
                "--all",
                "--baseline-path",
                "scripts/config/architecture_budget_baseline.json",
            ]
        )
        == 1
    )


@pytest.mark.unit
def test_checked_in_architecture_budget_baseline_matches_repository() -> None:
    """Repository-wide architecture budget baseline exists and covers all current findings."""
    repo_root = Path(__file__).resolve().parents[3]
    baseline_path = (
        repo_root / "scripts" / "config" / "architecture_budget_baseline.json"
    )
    assert baseline_path.is_file(), f"Checked-in baseline missing: {baseline_path}"

    config = mod._load_config(
        repo_root, Path("scripts/config/architecture_budget.json")
    )
    baseline_entries = mod._load_baseline(
        repo_root, Path("scripts/config/architecture_budget_baseline.json")
    )

    paths = mod._tracked_python_files(repo_root)
    violations = mod.collect_violations(
        repo_root=repo_root,
        paths=paths,
        config=config,
        baseline=baseline_entries,
    )
    assert violations == [], (
        f"Unexpected architecture violations outside baseline: {violations}"
    )
