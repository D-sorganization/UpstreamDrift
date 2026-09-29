"""TDD tests for assessment script bugs (issue #2497).

Four bugs:
1. create_issues_from_assessment.py reads 'critical_issues' key but current
   assessment_summary.json uses 'issues'. Findings are silently skipped.
2. Same script caps processing at 20 entries, silently dropping the majority.
3. finalize_comprehensive_assessment.py hardcodes 'review_2026-01-31.json' which
   doesn't exist; the current file is 'review.json'.
4. maintain_workflows.py __main__ block is a silent no-op (ends in `pass`).
"""

from __future__ import annotations

from pathlib import Path


class TestCreateIssuesSchema:
    """create_issues_from_assessment.py must read the 'issues' key, not 'critical_issues'."""

    def _source(self) -> str:
        return Path("scripts/create_issues_from_assessment.py").read_text()

    def test_reads_issues_key(self) -> None:
        """Script must read 'issues' key from summary JSON (current schema)."""
        source = self._source()
        # The fix: script should look up 'issues', not just 'critical_issues'
        assert 'summary.get("issues"' in source or "summary.get('issues'" in source, (
            "create_issues_from_assessment.py does not read 'issues' key from summary JSON. "
            "The current assessment_summary.json uses 'issues', not 'critical_issues'."
        )

    def test_no_hard_cap_of_20(self) -> None:
        """Script must not silently cap findings to 20 entries."""
        source = self._source()
        lines = source.splitlines()
        hard_cap_lines = [
            line
            for line in lines
            if "[:20]" in line and not line.strip().startswith("#")
        ]
        assert not hard_cap_lines, (
            "create_issues_from_assessment.py hard-caps processing to 20 entries, "
            "silently dropping the rest. Remove or replace with a configurable limit.\n"
            "Offending lines:\n" + "\n".join(hard_cap_lines)
        )


class TestFinalizeAssessmentPath:
    """finalize_comprehensive_assessment.py must use the current pragmatic review filename."""

    def _source(self) -> str:
        return Path("scripts/finalize_comprehensive_assessment.py").read_text()

    def test_no_stale_review_filename(self) -> None:
        """Script must not reference the nonexistent review_2026-01-31.json."""
        source = self._source()
        assert "review_2026-01-31.json" not in source, (
            "finalize_comprehensive_assessment.py references stale filename "
            "'review_2026-01-31.json' which no longer exists. "
            "The current file is 'review.json'."
        )

    def test_uses_current_review_filename(self) -> None:
        """Script must reference the current review.json filename."""
        source = self._source()
        assert '"review.json"' in source or "'review.json'" in source, (
            "finalize_comprehensive_assessment.py does not reference 'review.json'. "
            "Update PRAGMATIC_REPORT to point to docs/assessments/pragmatic_programmer/review.json."
        )


class TestMaintainWorkflowsNotSilent:
    """maintain_workflows.py __main__ block must not be a silent no-op."""

    def _source(self) -> str:
        return Path("scripts/maintain_workflows.py").read_text()

    def test_main_block_not_only_pass(self) -> None:
        """__main__ block must do more than `pass`."""
        source = self._source()
        lines = source.splitlines()
        in_main = False
        main_body_lines = []
        for line in lines:
            if line.strip() == 'if __name__ == "__main__":':
                in_main = True
                continue
            if in_main:
                if (
                    line.strip()
                    and not line.startswith(" ")
                    and not line.startswith("\t")
                ):
                    break
                main_body_lines.append(line.strip())

        # Exclude comments and bare `pass` — only count executable lines
        executable_body = [
            line
            for line in main_body_lines
            if line and not line.startswith("#") and line != "pass"
        ]
        assert executable_body, (
            "maintain_workflows.py __main__ block contains only `pass`. "
            "A script that does nothing but exits 0 misleads callers. "
            "Either implement the logic or raise NotImplementedError."
        )
