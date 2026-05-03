from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from scripts import check_mypy_exclusion_budget as checker


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _pyproject(tmp_path: Path, exclusions: list[str]) -> Path:
    quoted_exclusions = ",\n".join(f'    "{entry}"' for entry in exclusions)
    return _write(
        tmp_path / "pyproject.toml",
        f'[tool.mypy]\npython_version = "3.10"\nexclude = [\n{quoted_exclusions}\n]\n',
    )


def _pyproject_with_override(
    tmp_path: Path, exclusions: list[str], override: str
) -> Path:
    pyproject = _pyproject(tmp_path, exclusions)
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8") + "\n" + override,
        encoding="utf-8",
    )
    return pyproject


def _budget(tmp_path: Path, entries: list[dict[str, str]], cap: int = 5) -> Path:
    lines = [
        "{",
        '  "schema_version": 1,',
        '  "schedule": [{"effective_on": "2026-01-01", "max_exclusions": '
        f"{cap}" + "}],",
        '  "exclusions": [',
    ]
    rendered_entries = []
    for entry in entries:
        rendered_entries.append(
            "    {"
            f'"path": "{entry.get("path", "")}", '
            f'"owner": "{entry.get("owner", "")}", '
            f'"reason": "{entry.get("reason", "")}", '
            f'"expires_on": "{entry.get("expires_on", "")}"'
            "}"
        )
    lines.append(",\n".join(rendered_entries))
    lines.extend(["  ]", "}", ""])
    return _write(tmp_path / "scripts" / "config" / "budget.json", "\n".join(lines))


def test_budget_passes_when_pyproject_matches_metadata_and_cap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A matching, owned, unexpired budget should pass the ratchet."""
    exclusions = ["src/legacy/", "src/typed_later.py"]
    pyproject = _pyproject(tmp_path, exclusions)
    budget = _budget(
        tmp_path,
        [
            {
                "path": "src/legacy/",
                "owner": "@platform",
                "reason": "legacy imports need incremental typing",
                "expires_on": "2026-08-01",
            },
            {
                "path": "src/typed_later.py",
                "owner": "@physics-core",
                "reason": "third-party stubs diverge across supported runners",
                "expires_on": "2026-08-01",
            },
        ],
        cap=2,
    )

    assert (
        checker.main(
            [
                "--pyproject",
                str(pyproject),
                "--budget",
                str(budget),
                "--today",
                "2026-05-03",
            ]
        )
        == 0
    )
    assert "mypy exclusion budget passed" in capsys.readouterr().out


def test_budget_fails_for_missing_owner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every exclusion needs an accountable owner."""
    pyproject = _pyproject(tmp_path, ["src/legacy/"])
    budget = _budget(
        tmp_path,
        [
            {
                "path": "src/legacy/",
                "owner": "",
                "reason": "legacy imports need incremental typing",
                "expires_on": "2026-08-01",
            }
        ],
    )

    assert (
        checker.main(
            [
                "--pyproject",
                str(pyproject),
                "--budget",
                str(budget),
                "--today",
                "2026-05-03",
            ]
        )
        == 1
    )
    assert "missing owner" in capsys.readouterr().err


def test_budget_fails_for_expired_entry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Expired exclusions must be removed or renewed deliberately."""
    pyproject = _pyproject(tmp_path, ["src/legacy/"])
    budget = _budget(
        tmp_path,
        [
            {
                "path": "src/legacy/",
                "owner": "@platform",
                "reason": "legacy imports need incremental typing",
                "expires_on": "2026-05-02",
            }
        ],
    )

    assert (
        checker.main(
            [
                "--pyproject",
                str(pyproject),
                "--budget",
                str(budget),
                "--today",
                "2026-05-03",
            ]
        )
        == 1
    )
    assert "expired on 2026-05-02" in capsys.readouterr().err


def test_budget_fails_when_pyproject_adds_unbudgeted_exclusion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The TOML exclude list cannot grow without explicit budget metadata."""
    pyproject = _pyproject(tmp_path, ["src/legacy/", "src/new_skip/"])
    budget = _budget(
        tmp_path,
        [
            {
                "path": "src/legacy/",
                "owner": "@platform",
                "reason": "legacy imports need incremental typing",
                "expires_on": "2026-08-01",
            }
        ],
    )

    assert (
        checker.main(
            [
                "--pyproject",
                str(pyproject),
                "--budget",
                str(budget),
                "--today",
                "2026-05-03",
            ]
        )
        == 1
    )
    assert "not present in budget" in capsys.readouterr().err


def test_budget_fails_when_count_exceeds_current_schedule_cap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The active schedule cap is a ratchet and cannot be exceeded."""
    exclusions = ["src/one.py", "src/two.py"]
    pyproject = _pyproject(tmp_path, exclusions)
    budget = _budget(
        tmp_path,
        [
            {
                "path": "src/one.py",
                "owner": "@platform",
                "reason": "legacy imports need incremental typing",
                "expires_on": "2026-08-01",
            },
            {
                "path": "src/two.py",
                "owner": "@platform",
                "reason": "legacy imports need incremental typing",
                "expires_on": "2026-08-01",
            },
        ],
        cap=1,
    )

    assert (
        checker.main(
            [
                "--pyproject",
                str(pyproject),
                "--budget",
                str(budget),
                "--today",
                "2026-05-03",
            ]
        )
        == 1
    )
    assert "exclusions exceed active cap" in capsys.readouterr().err


def test_budget_fails_for_ignore_errors_override(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Whole-module ignore_errors bypasses are not accountable enough."""
    pyproject = _pyproject_with_override(
        tmp_path,
        ["src/legacy/"],
        "\n".join(
            [
                "[[tool.mypy.overrides]]",
                'module = ["src.shared.python.plotting.*"]',
                "ignore_errors = true",
            ]
        ),
    )
    budget = _budget(
        tmp_path,
        [
            {
                "path": "src/legacy/",
                "owner": "@platform",
                "reason": "legacy imports need incremental typing",
                "expires_on": "2026-08-01",
            }
        ],
    )

    assert (
        checker.main(
            [
                "--pyproject",
                str(pyproject),
                "--budget",
                str(budget),
                "--today",
                "2026-05-03",
            ]
        )
        == 1
    )
    assert "ignore_errors=true" in capsys.readouterr().err


def test_ci_standard_runs_mypy_exclusion_budget() -> None:
    """The ratchet is only useful when the main CI gate runs it."""
    workflow = yaml.safe_load(Path(".github/workflows/ci-standard.yml").read_text())
    quality_gate_steps = workflow["jobs"]["quality-gate"]["steps"]

    matching_steps = [
        step
        for step in quality_gate_steps
        if step.get("name") == "MyPy Exclusion Budget"
    ]

    assert matching_steps == [
        {
            "name": "MyPy Exclusion Budget",
            "run": "python3 scripts/check_mypy_exclusion_budget.py",
        }
    ]
