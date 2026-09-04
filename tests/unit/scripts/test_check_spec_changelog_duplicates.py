"""The changelog guard must bite, and must not bite the innocent.

SPEC.md's changelog used to carry a serial spec version that every pull request
had to claim the next free value of, so two branches routinely chose the same
number before either merged; keeping both rows -- the right call for the prose
-- silently duplicated the number, and nothing detected it until 54 versions
were already used more than once.

Repository_Management#1520 removed the cause: a row is keyed by its pull
request, which is unique by construction. These tests pin what the guard
enforces now -- the PR-keyed row contract, key uniqueness after the migration
cutover, and the still-orthogonal duplicate-*body* ratchet, which catches one
change logged twice regardless of how rows are keyed.
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
CONTRACT = REPO_ROOT / "shared_scripts" / "spec_changelog.py"

_HEADER = "## 12. Change Log\n\n| Date | PR | Changes |\n| ---- | -- | ------- |\n"

# The body ratchet only fingerprints substantial prose, so synthetic rows that
# must collide have to clear its 80-character floor.
_LONG_BODY = (
    "A substantial change-log body, long enough that the duplicate-body "
    "ratchet treats it as a fingerprint rather than an innocent repeat."
)


def _run(
    tmp_path: Path,
    spec_body: str,
    baseline: dict[str, object] | None,
) -> subprocess.CompletedProcess[str]:
    """Run the guard against a synthetic repo and return the completed process."""
    (tmp_path / "SPEC.md").write_text(spec_body, encoding="utf-8", newline="\n")
    config = tmp_path / "scripts" / "config"
    config.mkdir(parents=True, exist_ok=True)
    script_dir = tmp_path / "scripts" / "ci"
    script_dir.mkdir(parents=True, exist_ok=True)
    (script_dir / SCRIPT.name).write_text(
        SCRIPT.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
    )
    shared = tmp_path / "shared_scripts"
    shared.mkdir(parents=True, exist_ok=True)
    (shared / CONTRACT.name).write_text(
        CONTRACT.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
    )
    if baseline is not None:
        (config / "spec_changelog_duplicate_baseline.json").write_text(
            json.dumps(baseline), encoding="utf-8", newline="\n"
        )
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(script_dir / SCRIPT.name)],
        capture_output=True,
        text=True,
        check=False,
    )


class TestThePrKeyedRowContract:
    def test_a_serial_version_in_the_key_column_fails(self, tmp_path: Path) -> None:
        """The old row format is now the error, with the fix named."""
        body = _HEADER + "| 2026-09-04 | 1.0.718 | A serial where a key belongs. |\n"
        result = _run(tmp_path, body, {})
        assert result.returncode == 1
        assert "serial spec version" in result.stderr
        assert "#<pr>" in result.stderr

    def test_pr_keyed_rows_pass(self, tmp_path: Path) -> None:
        body = _HEADER + ("| 2026-09-04 | #9500 | A. |\n| 2026-09-04 | #9501 | B. |\n")
        assert _run(tmp_path, body, {}).returncode == 0

    def test_a_malformed_key_fails(self, tmp_path: Path) -> None:
        body = _HEADER + "| 2026-09-04 | 9500 | Missing the hash. |\n"
        assert _run(tmp_path, body, {}).returncode == 1

    def test_the_legacy_no_key_marker_passes(self, tmp_path: Path) -> None:
        """Migrated historical rows that referenced nothing carry `n/a`."""
        body = _HEADER + "| 2026-06-01 | n/a | A migrated row (spec 1.0.300). |\n"
        assert _run(tmp_path, body, {}).returncode == 0


class TestKeyUniqueness:
    def test_a_reused_key_after_the_cutover_fails(self, tmp_path: Path) -> None:
        """One row per pull request; a second means a row was copied."""
        body = _HEADER + ("| 2026-09-04 | #9500 | A. |\n| 2026-09-04 | #9500 | B. |\n")
        result = _run(tmp_path, body, {})
        assert result.returncode == 1
        assert "duplicate change-log key #9500" in result.stderr

    def test_a_reused_key_before_the_cutover_passes(self, tmp_path: Path) -> None:
        """Historical rows legitimately share a governing issue."""
        body = _HEADER + (
            "| 2026-06-01 | #7000 | A (spec 1.0.300). |\n"
            "| 2026-06-02 | #7000 | B (spec 1.0.301). |\n"
        )
        assert _run(tmp_path, body, {}).returncode == 0


class TestTheDuplicateBodyRatchetStillBites:
    """Orthogonal to keys: one change logged twice is still one change twice."""

    def test_two_rows_with_identical_prose_fail(self, tmp_path: Path) -> None:
        body = _HEADER + (
            f"| 2026-06-01 | #7001 | {_LONG_BODY} |\n"
            f"| 2026-06-02 | #7002 | {_LONG_BODY} |\n"
        )
        result = _run(tmp_path, body, {})
        assert result.returncode == 1
        assert "logged twice" in result.stderr

    def test_a_baselined_pair_passes_at_its_recorded_keys(self, tmp_path: Path) -> None:
        import hashlib

        digest = hashlib.sha256(_LONG_BODY.encode("utf-8")).hexdigest()[:16]
        body = _HEADER + (
            f"| 2026-06-01 | #7001 | {_LONG_BODY} |\n"
            f"| 2026-06-02 | #7002 | {_LONG_BODY} |\n"
        )
        baseline = {"duplicate_text": {digest: ["#7001", "#7002"]}}
        assert _run(tmp_path, body, baseline).returncode == 0

    def test_a_third_copy_fails_even_when_two_are_baselined(
        self, tmp_path: Path
    ) -> None:
        import hashlib

        digest = hashlib.sha256(_LONG_BODY.encode("utf-8")).hexdigest()[:16]
        body = _HEADER + "".join(
            f"| 2026-06-0{n} | #700{n} | {_LONG_BODY} |\n" for n in (1, 2, 3)
        )
        baseline = {"duplicate_text": {digest: ["#7001", "#7002"]}}
        assert _run(tmp_path, body, baseline).returncode == 1

    def test_a_short_body_may_repeat_innocently(self, tmp_path: Path) -> None:
        body = _HEADER + (
            "| 2026-06-01 | #7001 | Version bump. |\n"
            "| 2026-06-02 | #7002 | Version bump. |\n"
        )
        assert _run(tmp_path, body, {}).returncode == 0


class TestItFailsLoudlyRatherThanSilently:
    def test_a_missing_changelog_table_is_an_error_not_a_pass(
        self, tmp_path: Path
    ) -> None:
        # A format change must not turn the guard into a no-op that reports green.
        assert _run(tmp_path, "# SPEC\n\nNo changelog here.\n", {}).returncode == 1

    def test_an_empty_changelog_table_is_an_error_not_a_pass(
        self, tmp_path: Path
    ) -> None:
        assert _run(tmp_path, _HEADER, {}).returncode == 1


class TestTheRealSpecIsClean:
    def test_the_committed_spec_satisfies_its_own_baseline(self) -> None:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, str(SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_the_baseline_no_longer_carries_serial_version_allowances(self) -> None:
        """The 54 serial-collision allowances describe an impossible defect now."""
        baseline = json.loads(
            (
                REPO_ROOT
                / "scripts"
                / "config"
                / "spec_changelog_duplicate_baseline.json"
            ).read_text(encoding="utf-8")
        )
        assert "duplicates" not in baseline
        assert "duplicate_text" in baseline
