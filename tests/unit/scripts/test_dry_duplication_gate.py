"""Tests for the blocking DRY duplicate-logic quality gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ci import check_dry_duplication_gate as mod

pytestmark = pytest.mark.unit


DUPLICATED_BODY = "\n".join(
    [
        "    total = value",
        "    total += 1",
        "    total *= 2",
        "    if total > 10:",
        "        total -= 3",
        "    return total",
    ]
)


def _write_module(path: Path, function_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"def {function_name}(value):\n{DUPLICATED_BODY}\n",
        encoding="utf-8",
    )


def _config(**overrides) -> dict:
    config = {
        "threshold_lines": 6,
        "min_tokens_per_window": 8,
        "include": ["src/**/*.py"],
        "exclude": [],
        "baseline_path": "scripts/config/dry_duplication_baseline.json",
    }
    config.update(overrides)
    return config


def test_collect_findings_blocks_new_six_line_duplicate_in_src(tmp_path: Path) -> None:
    first = tmp_path / "src" / "package" / "alpha.py"
    second = tmp_path / "src" / "package" / "beta.py"
    _write_module(first, "alpha")
    _write_module(second, "beta")

    findings = mod.collect_findings(
        repo_root=tmp_path,
        paths=[first, second],
        config=_config(),
        baseline={"entries": {}},
    )

    assert len(findings) == 1
    assert findings[0].occurrence_count == 2
    assert [
        occ.path.relative_to(tmp_path).as_posix() for occ in findings[0].sample
    ] == [
        "src/package/alpha.py",
        "src/package/beta.py",
    ]


def test_collect_findings_honors_generated_path_exclusions(tmp_path: Path) -> None:
    first = tmp_path / "src" / "generated" / "alpha.py"
    second = tmp_path / "src" / "generated" / "beta.py"
    _write_module(first, "alpha")
    _write_module(second, "beta")

    findings = mod.collect_findings(
        repo_root=tmp_path,
        paths=[first, second],
        config=_config(exclude=["src/generated/**"]),
        baseline={"entries": {}},
    )

    assert findings == []


def test_baseline_allows_current_duplicates_but_blocks_growth(tmp_path: Path) -> None:
    first = tmp_path / "src" / "package" / "alpha.py"
    second = tmp_path / "src" / "package" / "beta.py"
    third = tmp_path / "src" / "package" / "gamma.py"
    _write_module(first, "alpha")
    _write_module(second, "beta")

    initial_findings = mod.collect_findings(
        repo_root=tmp_path,
        paths=[first, second],
        config=_config(),
        baseline={"entries": {}},
    )
    fingerprint = initial_findings[0].fingerprint
    baseline = {
        "entries": {
            fingerprint: {
                "max_occurrences": 2,
                "owner": "@core",
                "issue": "#7315",
                "reason": "Grandfathered by DRY duplication gate ratchet.",
            }
        }
    }

    assert (
        mod.collect_findings(
            repo_root=tmp_path,
            paths=[first, second],
            config=_config(),
            baseline=baseline,
        )
        == []
    )

    _write_module(third, "gamma")
    grown_findings = mod.collect_findings(
        repo_root=tmp_path,
        paths=[first, second, third],
        config=_config(),
        baseline=baseline,
    )

    assert len(grown_findings) == 1
    assert grown_findings[0].baseline_max_occurrences == 2
    assert grown_findings[0].occurrence_count == 3


def test_main_reports_blocking_duplicate_locations(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config_path = tmp_path / "scripts" / "config" / "dry_duplication_gate.json"
    baseline_path = tmp_path / "scripts" / "config" / "dry_duplication_baseline.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(_config()), encoding="utf-8")
    baseline_path.write_text(json.dumps({"entries": {}}), encoding="utf-8")
    first = tmp_path / "src" / "package" / "alpha.py"
    second = tmp_path / "src" / "package" / "beta.py"
    _write_module(first, "alpha")
    _write_module(second, "beta")

    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(mod, "_tracked_python_files", lambda repo: [first, second])

    exit_code = mod.main(["--config-path", "scripts/config/dry_duplication_gate.json"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert (
        "FAIL: DRY duplication gate found unapproved duplicated logic" in captured.err
    )
    assert "src/package/alpha.py:" in captured.err
    assert "src/package/beta.py:" in captured.err
