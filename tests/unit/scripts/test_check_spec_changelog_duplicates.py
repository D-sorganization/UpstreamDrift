"""The changelog duplicate ratchet must bite, and must not bite the innocent.

SPEC.md's changelog is hot-prepended by every PR, so two branches routinely
choose the same next-free version before either merges. Keeping both rows --
the right call for the prose -- silently duplicates the number, and nothing
detected that until 54 versions were already used more than once.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_spec_changelog_duplicates.py"

_HEADER = (
    "## 12. Change Log\n\n| Date | Version | Changes |\n| ---- | ------- | ------- |\n"
)


def _run(tmp_path: Path, spec_body: str, baseline: dict[str, int] | None) -> int:
    """Run the guard against a synthetic repo and return its exit code."""
    (tmp_path / "SPEC.md").write_text(spec_body, encoding="utf-8")
    config = tmp_path / "scripts" / "config"
    config.mkdir(parents=True, exist_ok=True)
    script_dir = tmp_path / "scripts" / "ci"
    script_dir.mkdir(parents=True, exist_ok=True)
    (script_dir / SCRIPT.name).write_text(
        SCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    if baseline is not None:
        (config / "spec_changelog_duplicate_baseline.json").write_text(
            json.dumps({"duplicates": baseline}), encoding="utf-8"
        )
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(script_dir / SCRIPT.name)],
        capture_output=True,
        text=True,
        check=False,
    ).returncode


class TestTheRatchetBites:
    def test_a_repeated_version_fails(self, tmp_path: Path) -> None:
        body = _HEADER + (
            "| 2026-08-29 | 1.0.500 | A. |\n| 2026-08-29 | 1.0.500 | B. |\n"
        )
        assert _run(tmp_path, body, {}) == 1

    def test_distinct_versions_pass(self, tmp_path: Path) -> None:
        body = _HEADER + (
            "| 2026-08-29 | 1.0.501 | A. |\n| 2026-08-29 | 1.0.500 | B. |\n"
        )
        assert _run(tmp_path, body, {}) == 0

    def test_a_third_row_fails_even_when_two_are_baselined(
        self, tmp_path: Path
    ) -> None:
        # Historical debt is tolerated at its recorded multiplicity, not beyond.
        body = _HEADER + "".join(
            f"| 2026-08-29 | 1.0.500 | Row {n}. |\n" for n in range(3)
        )
        assert _run(tmp_path, body, {"1.0.500": 2}) == 1


class TestTheBaselineIsAnAllowanceNotABlanket:
    def test_a_baselined_duplicate_passes_at_its_recorded_count(
        self, tmp_path: Path
    ) -> None:
        body = _HEADER + (
            "| 2026-08-29 | 1.0.500 | A. |\n| 2026-08-29 | 1.0.500 | B. |\n"
        )
        assert _run(tmp_path, body, {"1.0.500": 2}) == 0

    def test_an_unrelated_new_duplicate_still_fails(self, tmp_path: Path) -> None:
        body = _HEADER + (
            "| 2026-08-29 | 1.0.500 | A. |\n| 2026-08-29 | 1.0.500 | B. |\n"
            "| 2026-08-29 | 1.0.600 | C. |\n| 2026-08-29 | 1.0.600 | D. |\n"
        )
        assert _run(tmp_path, body, {"1.0.500": 2}) == 1


class TestItFailsLoudlyRatherThanSilently:
    def test_a_missing_changelog_table_is_an_error_not_a_pass(
        self, tmp_path: Path
    ) -> None:
        # A format change must not turn the guard into a no-op that reports green.
        assert _run(tmp_path, "# SPEC\n\nNo changelog here.\n", {}) == 1


class TestTheRealSpecIsClean:
    def test_the_committed_spec_satisfies_its_own_baseline(self) -> None:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, str(SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
