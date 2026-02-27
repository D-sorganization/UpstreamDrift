#!/usr/bin/env python3
"""
Orchestrate the full repository assessment workflow.

1. Run individual assessments (A-O).
2. Generate comprehensive summary.
3. Stage/Create GitHub issues.
"""

import logging
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ASSESSMENTS = {
    "A": "Code Structure",
    "B": "Documentation",
    "C": "Test Coverage",
    "D": "Error Handling",
    "E": "Performance",
    "F": "Security",
    "G": "Dependencies",
    "H": "CI/CD",
    "I": "Code Style",
    "J": "API Design",
    "K": "Data Handling",
    "L": "Logging",
    "M": "Configuration",
    "N": "Scalability",
    "O": "Maintainability",
}


def run_command(cmd: list[str]) -> bool:
    """Run a command and return success status."""
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        logger.error(f"Command failed: {' '.join(cmd)}")
        return False


def main():
    logger.info("Starting fresh assessment workflow...")

    docs_dir = Path("docs/assessments")
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run individual assessments
    logger.info("Running individual assessments (A-O)...")
    for assessment_id, name in ASSESSMENTS.items():
        safe_name = name.replace(" ", "_").replace("/", "_")
        output_file = docs_dir / f"Assessment_{assessment_id}_{safe_name}.md"

        cmd = [
            sys.executable,
            "scripts/run_assessment.py",
            "--assessment",
            assessment_id,
            "--output",
            str(output_file),
        ]

        logger.info(f"Running Assessment {assessment_id}...")
        run_command(cmd)

    # 2. Generate summary
    logger.info("Generating comprehensive summary...")
    summary_md = docs_dir / "Comprehensive_Assessment.md"
    summary_json = docs_dir / "assessment_summary.json"

    cmd_summary = [
        sys.executable,
        "scripts/generate_assessment_summary.py",
        "--input",
        str(docs_dir / "Assessment_*.md"),
        "--output",
        str(summary_md),
        "--json-output",
        str(summary_json),
    ]

    if not run_command(cmd_summary):
        logger.error("Failed to generate summary.")
        sys.exit(1)

    # 3. Create/Stage Issues
    logger.info("Staging GitHub issues...")
    issues_file = docs_dir / "ISSUES_TO_CREATE.md"

    cmd_issues = [
        sys.executable,
        "scripts/create_issues_from_assessment.py",
        "--input",
        str(summary_json),
        "--output-file",
        str(issues_file),
        "--dry-run",  # Default to dry-run to avoid spamming if GH is configured
    ]

    if run_command(cmd_issues):
        logger.info(f"Issues staged in {issues_file}")
    else:
        logger.error("Failed to process issues.")

    logger.info("Assessment workflow complete.")
    logger.info(f"Summary: {summary_md}")
    logger.info(f"Issues: {issues_file}")


if __name__ == "__main__":
    main()
