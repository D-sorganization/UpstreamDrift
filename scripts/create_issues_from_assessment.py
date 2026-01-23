#!/usr/bin/env python3
"""
Create GitHub issues from assessment findings.

This script reads the assessment summary JSON and creates GitHub issues
for untracked critical findings.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_existing_issues() -> list[dict[str, Any]]:
    """Fetch existing GitHub issues."""
    try:
        result = subprocess.run(
            ["gh", "issue", "list", "--limit", "200", "--json", "number,title,state,labels"],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except Exception as e:
        logger.warning(f"Could not fetch existing issues: {e}")
        return []


def issue_exists(title: str, existing_issues: list[dict[str, Any]]) -> bool:
    """Check if an issue with similar title already exists."""
    title_lower = title.lower()
    for issue in existing_issues:
        if issue.get("state") == "OPEN":
            existing_title = issue.get("title", "").lower()
            if title_lower in existing_title or existing_title in title_lower:
                return True
    return False


def create_github_issue(
    title: str,
    body: str,
    labels: list[str],
    dry_run: bool = False,
) -> bool:
    """
    Create a GitHub issue.

    Args:
        title: Issue title
        body: Issue body
        labels: List of label names
        dry_run: If True, log instead of creating

    Returns:
        True if successful, False otherwise
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would create issue: {title}")
        return True

    try:
        cmd = ["gh", "issue", "create", "--title", title, "--body", body]

        # Add labels (only if they exist in the repo)
        if labels:
            cmd.extend(["--label", ",".join(labels)])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            issue_url = result.stdout.strip()
            logger.info(f"✓ Created issue: {issue_url}")
            return True
        else:
            # Try without labels if label creation failed
            if "label" in result.stderr.lower():
                cmd_no_labels = ["gh", "issue", "create", "--title", title, "--body", body]
                result = subprocess.run(cmd_no_labels, capture_output=True, text=True, check=True)
                logger.info(f"✓ Created issue (without labels): {result.stdout.strip()}")
                return True
            logger.error(f"✗ Failed to create issue '{title}': {result.stderr}")
            return False

    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to create issue '{title}': {e.stderr}")
        return False


def process_assessment_findings(
    summary_file: Path,
    severities: list[str],
    check_existing: bool = True,
    dry_run: bool = False,
) -> int:
    """
    Process assessment findings and create issues.

    Args:
        summary_file: Path to assessment_summary.json
        severities: List of severity levels to process
        check_existing: Whether to check for existing issues
        dry_run: If True, don't actually create issues

    Returns:
        Number of issues created
    """
    if not summary_file.exists():
        logger.warning(f"Summary file not found: {summary_file}")
        return 0

    try:
        with open(summary_file) as f:
            summary = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {summary_file}: {e}")
        return 0

    # Get existing issues if checking
    existing_issues = get_existing_issues() if check_existing else []

    issues_created = 0
    critical_issues = summary.get("critical_issues", [])

    if not critical_issues:
        logger.info("No critical issues found in assessment summary")
        return 0

    for issue in critical_issues:
        severity = issue.get("severity", "UNKNOWN")
        if severity.upper() not in [s.upper() for s in severities]:
            continue

        title = issue.get("title", "Assessment Finding")
        description = issue.get("description", "No description provided")
        assessment = issue.get("assessment", "Unknown")
        recommendation = issue.get("recommendation", "Review and address this finding")

        # Create full issue title
        full_title = f"[Assessment {assessment}] {title}"

        # Check if exists
        if check_existing and issue_exists(full_title, existing_issues):
            logger.info(f"⏭ Skipping (exists): {full_title}")
            continue

        # Build issue body
        body = f"""## Assessment Finding

**Assessment**: {assessment}
**Severity**: {severity}

### Description

{description}

### Recommendation

{recommendation}

---
*This issue was created automatically by the Jules Assessment workflow.*
"""

        # Determine labels
        labels = ["assessment"]
        if severity.upper() == "CRITICAL":
            labels.append("critical")
        elif severity.upper() == "BLOCKER":
            labels.append("blocker")

        # Create issue
        if create_github_issue(full_title, body, labels, dry_run):
            issues_created += 1

    return issues_created


def main():
    parser = argparse.ArgumentParser(description="Create GitHub issues from assessment findings")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to assessment_summary.json",
    )
    parser.add_argument(
        "--severity",
        default="CRITICAL,BLOCKER",
        help="Comma-separated severity levels to process (default: CRITICAL,BLOCKER)",
    )
    parser.add_argument(
        "--check-existing",
        action="store_true",
        default=True,
        help="Check for existing issues before creating (default: True)",
    )
    parser.add_argument(
        "--no-check-existing",
        action="store_false",
        dest="check_existing",
        help="Don't check for existing issues",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would happen without creating issues",
    )

    args = parser.parse_args()

    severities = [s.strip() for s in args.severity.split(",")]

    logger.info(f"Processing assessment findings from: {args.input}")
    logger.info(f"Severity filter: {', '.join(severities)}")

    issues_created = process_assessment_findings(
        args.input,
        severities,
        check_existing=args.check_existing,
        dry_run=args.dry_run,
    )

    logger.info(f"✓ Created {issues_created} issues")
    sys.exit(0)


if __name__ == "__main__":
    main()
