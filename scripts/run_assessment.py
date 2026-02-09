#!/usr/bin/env python3
"""
Run a specific assessment (A-O) on the repository using shared utilities.
"""

import argparse
from datetime import datetime
from pathlib import Path

from scripts.script_utils import (
    check_docs_status,
    count_test_files,
    find_python_files,
    run_main,
    run_tool_check,
    setup_script_logging,
)

logger = setup_script_logging(__name__)

_CHECK = "\u2713"
_CROSS = "\u2717"

ASSESSMENTS = {
    "A": {"name": "Architecture", "description": "Code structure and organization"},
    "B": {
        "name": "Hygiene & Quality",
        "description": "Linting, formatting, code quality",
    },
    "C": {"name": "Documentation", "description": "README, docstrings, comments"},
    "G": {"name": "Testing", "description": "Test coverage, test quality"},
}


def _assess_architecture(py_files: list) -> tuple[list[str], int]:
    """Run architecture assessment (A)."""
    has_src = Path("src").exists() or Path("python").exists()
    has_tests = Path("tests").exists()
    findings = [
        f"- Python files found: {len(py_files)}",
        f"- Source directory structure: {_CHECK if has_src else _CROSS}",
        f"- Tests directory: {_CHECK if has_tests else _CROSS}",
    ]
    penalty = (0 if has_src else 2) + (0 if has_tests else 1)
    return findings, 10 - penalty


def _assess_hygiene() -> tuple[list[str], int]:
    """Run hygiene & quality assessment (B)."""
    ruff = run_tool_check(["ruff", "check", ".", "--statistics"])
    black = run_tool_check(["black", "--check", "--quiet", "."])
    findings = [
        f"- Ruff check: {_CHECK + ' passed' if ruff['exit_code'] == 0 else _CROSS + ' issues found'}",
        f"- Black formatting: {_CHECK + ' formatted' if black['exit_code'] == 0 else _CROSS + ' needs formatting'}",
    ]
    penalty = (0 if ruff["exit_code"] == 0 else 2) + (
        0 if black["exit_code"] == 0 else 1
    )
    return findings, 10 - penalty


def _assess_documentation() -> tuple[list[str], int]:
    """Run documentation assessment (C)."""
    docs = check_docs_status()
    findings = [
        f"- README.md: {_CHECK if docs['readme'] else _CROSS}",
        f"- docs/ directory: {_CHECK if docs['docs_dir'] else _CROSS}",
        f"- CHANGELOG.md: {_CHECK if docs['changelog'] else _CROSS}",
    ]
    penalty = (0 if docs["readme"] else 3) + (0 if docs["docs_dir"] else 1)
    return findings, 10 - penalty


def _assess_testing() -> tuple[list[str], int]:
    """Run testing assessment (G)."""
    cnt = count_test_files()
    findings = [
        f"- Test files found: {cnt}",
        "- Test coverage: Run pytest --cov for details",
    ]
    penalty = 5 if cnt == 0 else (2 if cnt < 5 else 0)
    return findings, 10 - penalty


def _assess_unsupported(py_files: list) -> tuple[list[str], int | None]:
    """Return placeholder for unsupported assessment categories."""
    findings = [
        f"- Python files analyzed: {len(py_files)}",
        "- **REQUIRES REVIEW**: No automated checks available for this category",
        "- Score must be assigned by Jules bot or manual code review",
        "- Do NOT use a default score - real analysis is required",
    ]
    return findings, None


def _format_report(
    assessment_id: str, name: str, findings: list[str], score: int | None
) -> str:
    """Format an assessment report as markdown."""
    if score is not None:
        score = max(0, min(10, score))
        score_display = f"{score}/10"
    else:
        score_display = "PENDING REVIEW"

    return f"""# Assessment {assessment_id}: {name}
**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Score**: {score_display}

## Findings
{chr(10).join(findings)}

## Recommendations
- Review findings and address {_CROSS} items.
"""


def run_assessment(assessment_id: str, output_path: Path) -> int:
    """Run a specific assessment and generate report."""
    assessment = ASSESSMENTS.get(
        assessment_id, {"name": "General", "description": "Manual review required"}
    )
    logger.info(f"Running Assessment {assessment_id}: {assessment['name']}...")

    py_files = find_python_files()

    handlers = {
        "A": lambda: _assess_architecture(py_files),
        "B": lambda: _assess_hygiene(),
        "C": lambda: _assess_documentation(),
        "G": lambda: _assess_testing(),
    }
    handler = handlers.get(assessment_id, lambda: _assess_unsupported(py_files))
    findings, score = handler()

    report = _format_report(assessment_id, assessment["name"], findings, score)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)

    score_display = f"{score}/10" if score is not None else "PENDING REVIEW"
    logger.info(
        f"\u2713 Assessment {assessment_id} saved to {output_path} (Score: {score_display})"
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run repository assessment")
    parser.add_argument("--assessment", required=True, choices=list("ABCDEFGHIJKLMNO"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    return run_assessment(args.assessment, args.output)


if __name__ == "__main__":
    run_main(main, logger)
