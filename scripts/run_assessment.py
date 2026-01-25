#!/usr/bin/env python3
"""
Run a specific assessment (A-O) on the repository using shared utilities.
"""

import argparse
import sys
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

ASSESSMENTS = {
    "A": {"name": "Architecture", "description": "Code structure and organization"},
    "B": {"name": "Hygiene & Quality", "description": "Linting, formatting, code quality"},
    "C": {"name": "Documentation", "description": "README, docstrings, comments"},
    "G": {"name": "Testing", "description": "Test coverage, test quality"},
}


def run_assessment(assessment_id: str, output_path: Path) -> int:
    """Run a specific assessment and generate report."""
    assessment = ASSESSMENTS.get(assessment_id, {"name": "General", "description": "Manual review required"})
    logger.info(f"Running Assessment {assessment_id}: {assessment['name']}...")

    findings = []
    score = 10
    py_files = find_python_files()

    if assessment_id == "A":
        has_src = Path("src").exists() or Path("python").exists()
        has_tests = Path("tests").exists()
        findings = [
            f"- Python files found: {len(py_files)}",
            f"- Source directory structure: {'✓' if has_src else '✗'}",
            f"- Tests directory: {'✓' if has_tests else '✗'}",
        ]
        score -= (0 if has_src else 2) + (0 if has_tests else 1)

    elif assessment_id == "B":
        ruff = run_tool_check(["ruff", "check", ".", "--statistics"])
        black = run_tool_check(["black", "--check", "--quiet", "."])
        findings = [
            f"- Ruff check: {'✓ passed' if ruff['exit_code'] == 0 else '✗ issues found'}",
            f"- Black formatting: {'✓ formatted' if black['exit_code'] == 0 else '✗ needs formatting'}",
        ]
        score -= (0 if ruff["exit_code"] == 0 else 2) + (0 if black["exit_code"] == 0 else 1)

    elif assessment_id == "C":
        docs = check_docs_status()
        findings = [
            f"- README.md: {'✓' if docs['readme'] else '✗'}",
            f"- docs/ directory: {'✓' if docs['docs_dir'] else '✗'}",
            f"- CHANGELOG.md: {'✓' if docs['changelog'] else '✗'}",
        ]
        score -= (0 if docs["readme"] else 3) + (0 if docs["docs_dir"] else 1)

    elif assessment_id == "G":
        cnt = count_test_files()
        findings = [f"- Test files found: {cnt}", "- Test coverage: Run pytest --cov for details"]
        score -= (5 if cnt == 0 else (2 if cnt < 5 else 0))

    else:
        findings = [f"- Python files analyzed: {len(py_files)}", "- Manual review recommended"]

    score = max(0, min(10, score))
    
    report = f"""# Assessment {assessment_id}: {assessment['name']}
**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Score**: {score}/10

## Findings
{chr(10).join(findings)}

## Recommendations
- Review findings and address ✗ items.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    logger.info(f"✓ Assessment {assessment_id} saved to {output_path} (Score: {score}/10)")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run repository assessment")
    parser.add_argument("--assessment", required=True, choices=list("ABCDEFGHIJKLMNO"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    return run_assessment(args.assessment, args.output)


if __name__ == "__main__":
    run_main(main, logger)
