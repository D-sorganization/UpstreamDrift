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
            [
                "gh",
                "issue",
                "list",
                "--limit",
                "200",
                "--json",
                "number,title,state,labels",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return list(json.loads(result.stdout))
    except Exception as e:
        logger.warning(f"Could not fetch existing issues: {e}")
        return []


def issue_exists(title: str, existing_issues: list[dict[str, Any]]) -> bool:
    """Check if an issue with similar title already exists."""
    # Simple check for now - could be more sophisticated
    title_lower = title.lower()
    for issue in existing_issues:
        if issue["state"] == "OPEN":
            existing_title = issue["title"].lower()
            # Check for significant overlap
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
        logger.debug(f"Labels: {', '.join(labels)}")
        logger.debug(f"Body:\n{body}")
        return True

    try:
        # Build gh command
        cmd = ["gh", "issue", "create", "--title", title, "--body", body]

        # Add labels
        if labels:
            cmd.extend(["--label", ",".join(labels)])

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        issue_url = result.stdout.strip()
        logger.info(f"✓ Created issue: {issue_url}")
        return True

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
        summary_file: Path to assessment summary JSON
        severities: List of severities to create issues for
        check_existing: If True, skip issues that already exist
        dry_run: If True, log instead of creating

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    # Load assessment summary
    try:
        with open(summary_file) as f:
            summary = json.load(f)
    except Exception as e:
        logger.error(f"Could not load summary file: {e}")
        return 1

    critical_issues = summary.get("critical_issues", [])

    if not critical_issues:
        logger.info("No critical issues found in assessment")
        return 0

    logger.info(f"Found {len(critical_issues)} critical issues in assessment")

    # Get existing issues if checking
    existing_issues = []
    if check_existing:
        logger.info("Fetching existing GitHub issues...")
        existing_issues = get_existing_issues()
        logger.info(f"Found {len(existing_issues)} existing issues")

    # Filter by severity
    filtered_issues = [i for i in critical_issues if i.get("severity") in severities]

    logger.info(
        f"Filtered to {len(filtered_issues)} issues with severities: {', '.join(severities)}"
    )

    # Get repository name from current directory
    repo_name = Path.cwd().name
    repo_short_names = {
        "Gasification_Model": "GasModel",
        "Tools": "Tools",
        "AffineDrift": "AffineDrift",
        "Games": "Games",
        "Golf_Modeling_Suite": "GolfSuite",
        "MLProjects": "MLProj",
        "Playground": "Playground",
        "MEB_Conversion": "MEBConv",
        "Repository_Management": "RepoMgmt",
    }
    repo_short = repo_short_names.get(repo_name, repo_name[:8])

    # Category classification based on source
    def classify_category(source_name: str, description: str) -> str:
        """Classify issue into a category."""
        text = (source_name + " " + description).lower()

        if (
            "architecture" in text
            or "implementation" in text
            or "Assessment_A" in source_name
        ):
            return "Architecture"
        elif "quality" in text or "hygiene" in text or "Assessment_B" in source_name:
            return "Code Quality"
        elif "documentation" in text or "Assessment_C" in source_name:
            return "Documentation"
        elif "user" in text or "ux" in text or "Assessment_D" in source_name:
            return "User Experience"
        elif "performance" in text or "Assessment_E" in source_name:
            return "Performance"
        elif (
            "installation" in text
            or "deployment" in text
            or "Assessment_F" in source_name
        ):
            return "Installation"
        elif "test" in text or "Assessment_G" in source_name:
            return "Testing"
        elif "error" in text or "Assessment_H" in source_name:
            return "Error Handling"
        elif "security" in text or "Assessment_I" in source_name:
            return "Security"
        elif "extensibility" in text or "Assessment_J" in source_name:
            return "Extensibility"
        elif "reproducibility" in text or "Assessment_K" in source_name:
            return "Reproducibility"
        elif "maintainability" in text or "Assessment_L" in source_name:
            return "Maintainability"
        elif "educational" in text or "Assessment_M" in source_name:
            return "Documentation"
        elif "visualization" in text or "Assessment_N" in source_name:
            return "Visualization"
        elif "ci" in text or "cd" in text or "Assessment_O" in source_name:
            return "CI/CD"
        else:
            return "General"

    # Create issues
    created_count = 0
    skipped_count = 0

    for issue in filtered_issues[:20]:  # Limit to 20 to avoid spam
        severity = issue.get("severity", "UNKNOWN")
        description = issue.get("description", "No description")
        source = issue.get("source", "Unknown")

        # Classify category
        category = classify_category(source, description)

        # Clean description for title (remove markdown, truncate)
        clean_desc = description.replace("**", "").replace("*", "").replace("`", "")
        clean_desc = clean_desc.split("\n")[0]  # First line only
        if len(clean_desc) > 60:
            clean_desc = clean_desc[:57] + "..."

        # Generate standardized title
        title = f"[{repo_short}] {severity} {category}: {clean_desc}"

        body = f"""## Issue Description

**Severity**: {severity}
**Category**: {category}
**Source**: {source}
**Identified**: {summary.get('timestamp', 'Unknown')}

### Problem

{description}


### Impact

This issue was identified during automated repository assessment and requires attention.

### References

- Assessment Report: {source}
- Full Assessment: docs/assessments/COMPREHENSIVE_ASSESSMENT_SUMMARY_{summary.get('timestamp', '')[:10]}.md

### Next Steps

1. Investigate the issue
2. Determine root cause
3. Implement fix
4. Verify resolution
5. Update tests if needed

---

🤖 Auto-generated by [Jules Assessment Auto-Fix](https://github.com/D-sorganization/Gasification_Model/actions/workflows/Jules-Assessment-AutoFix.yml)
"""

        # Determine labels
        labels = ["auto-generated", "quality-control"]
        if severity in ("BLOCKER", "CRITICAL"):
            labels.append("bug")
        else:
            labels.append("enhancement")

        # Check if already exists
        if check_existing and issue_exists(title, existing_issues):
            logger.info(f"⊘ Skipping (already exists): {title}")
            skipped_count += 1
            continue

        # Create the issue
        if create_github_issue(title, body, labels, dry_run):
            created_count += 1

    logger.info(f"\n✓ Summary: Created {created_count} issues, skipped {skipped_count}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Create GitHub issues from assessment")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Assessment summary JSON file",
    )
    parser.add_argument(
        "--severity",
        default="BLOCKER,CRITICAL",
        help="Comma-separated list of severities to create issues for",
    )
    parser.add_argument(
        "--check-existing",
        action="store_true",
        help="Check for existing issues before creating",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print issues instead of creating them",
    )

    args = parser.parse_args()

    severities = [s.strip().upper() for s in args.severity.split(",")]

    exit_code = process_assessment_findings(
        args.input,
        severities,
        args.check_existing,
        args.dry_run,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
