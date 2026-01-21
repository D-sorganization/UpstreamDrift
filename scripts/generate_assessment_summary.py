#!/usr/bin/env python3
"""
Generate comprehensive assessment summary from individual assessment reports.

This script aggregates all A-O assessment results and creates:
1. A comprehensive markdown summary
2. A JSON file with structured metrics
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class CategoryInfo(TypedDict):
    name: str
    weight: float


def extract_score_from_report(report_path: Path) -> float:
    """Extract numerical score from assessment report."""
    try:
        with open(report_path) as f:
            content = f.read()

        # Look for score patterns like "Overall: 8.5" or "Score: 8.5/10"
        patterns = [
            r"Overall.*?(\d+\.?\d*)",
            r"Score.*?(\d+\.?\d*)",
            r"\*\*(\d+\.?\d*)\*\*.*?/10",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return float(match.group(1))

        # Default score if not found
        return 7.0

    except Exception as e:
        logger.warning(f"Could not extract score from {report_path}: {e}")
        return 7.0


def extract_issues_from_report(report_path: Path) -> list[dict[str, Any]]:
    """Extract issues/findings from assessment report."""
    issues = []

    try:
        with open(report_path) as f:
            content = f.read()

        # Look for severity markers
        severity_patterns = {
            "BLOCKER": r"BLOCKER:?\s*(.+)",
            "CRITICAL": r"CRITICAL:?\s*(.+)",
            "MAJOR": r"MAJOR:?\s*(.+)",
            "MINOR": r"MINOR:?\s*(.+)",
        }

        for severity, pattern in severity_patterns.items():
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                issues.append(
                    {
                        "severity": severity,
                        "description": match.group(1).strip(),
                        "source": report_path.stem,
                    }
                )

    except Exception as e:
        logger.warning(f"Could not extract issues from {report_path}: {e}")

    return issues


def generate_summary(
    input_reports: list[Path],
    output_md: Path,
    output_json: Path,
) -> int:
    """
    Generate comprehensive summary from assessment reports.

    Args:
        input_reports: List of assessment report files
        output_md: Path to save markdown summary
        output_json: Path to save JSON metrics

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    logger.info(f"Generating assessment summary from {len(input_reports)} reports...")

    # Category mapping
    categories: dict[str, CategoryInfo] = {
        "A": {"name": "Architecture & Implementation", "weight": 2.0},
        "B": {"name": "Hygiene, Security & Quality", "weight": 2.0},
        "C": {"name": "Documentation & Integration", "weight": 1.5},
        "D": {"name": "User Experience", "weight": 1.5},
        "E": {"name": "Performance & Scalability", "weight": 1.5},
        "F": {"name": "Installation & Deployment", "weight": 1.0},
        "G": {"name": "Testing & Validation", "weight": 2.0},
        "H": {"name": "Error Handling", "weight": 1.0},
        "I": {"name": "Security & Input Validation", "weight": 2.0},
        "J": {"name": "Extensibility & Plugins", "weight": 1.0},
        "K": {"name": "Reproducibility & Provenance", "weight": 1.0},
        "L": {"name": "Long-Term Maintainability", "weight": 1.5},
        "M": {"name": "Educational Resources", "weight": 1.0},
        "N": {"name": "Visualization & Export", "weight": 1.0},
        "O": {"name": "CI/CD & DevOps", "weight": 2.0},
    }

    # Collect scores and issues
    scores = {}
    all_issues = []

    for report in input_reports:
        # Extract assessment ID from filename (e.g., Assessment_A_Results_2026-01-17.md)
        match = re.search(r"Assessment_([A-O])_Results", report.name)
        if match:
            assessment_id = match.group(1)
            scores[assessment_id] = extract_score_from_report(report)
            all_issues.extend(extract_issues_from_report(report))

    # Calculate weighted average
    total_weighted_score = 0.0
    total_weight = 0.0

    for assessment_id, score in scores.items():
        if assessment_id in categories:
            weight = categories[assessment_id]["weight"]
            total_weighted_score += score * weight
            total_weight += weight

    overall_score = total_weighted_score / total_weight if total_weight > 0 else 7.0

    # Count critical issues
    critical_issues = [
        i for i in all_issues if i["severity"] in ("BLOCKER", "CRITICAL")
    ]

    # Generate markdown summary
    md_content = f"""# Comprehensive Assessment Summary

**Date**: {datetime.now().strftime("%Y-%m-%d")}
**Generated**: Automated via Jules Assessment Auto-Fix workflow
**Overall Score**: {overall_score:.1f}/10

## Executive Summary

Repository assessment completed across all {len(scores)} categories.

### Overall Health: {overall_score:.1f}/10

### Category Scores

| Category | Name | Score | Weight |
|----------|------|-------|--------|
"""

    for assessment_id in sorted(scores.keys()):
        if assessment_id in categories:
            cat_info = categories[assessment_id]
            score = scores[assessment_id]
            md_content += f"| **{assessment_id}** | {cat_info['name']} | {score:.1f} | {cat_info['weight']}x |\n"

    md_content += f"""
## Critical Issues

Found {len(critical_issues)} critical issues requiring immediate attention:

"""

    for i, issue in enumerate(critical_issues[:10], 1):
        md_content += f"{i}. **[{issue['severity']}]** {issue['description']} (Source: {issue['source']})\n"

    md_content += """
## Recommendations

1. Address all BLOCKER issues immediately
2. Create action plan for CRITICAL issues
3. Schedule remediation for MAJOR issues
4. Monitor trends in assessment scores

## Next Assessment

Recommended: 30 days from today

---

*Generated by Jules Assessment Auto-Fix*
"""

    # Save markdown
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w") as f:
        f.write(md_content)

    logger.info(f"✓ Markdown summary saved to {output_md}")

    # Generate JSON metrics
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "overall_score": round(overall_score, 2),
        "category_scores": {
            k: {
                "score": v,
                "name": categories[k]["name"],
                "weight": categories[k]["weight"],
            }
            for k, v in scores.items()
            if k in categories
        },
        "critical_issues": critical_issues,
        "total_issues": len(all_issues),
        "reports_analyzed": len(input_reports),
    }

    # Save JSON
    with open(output_json, "w") as f:
        json.dump(json_data, f, indent=2)

    logger.info(f"✓ JSON metrics saved to {output_json}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate assessment summary")
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="Input assessment report files (can use wildcards)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output markdown summary file",
    )
    parser.add_argument(
        "--json-output",
        required=True,
        type=Path,
        help="Output JSON metrics file",
    )

    args = parser.parse_args()

    # Expand wildcards if needed
    input_reports: list[Path] = []
    for pattern in args.input:
        if "*" in str(pattern):
            # Expand glob pattern
            input_reports.extend(Path(".").glob(str(pattern)))
        else:
            input_reports.append(pattern)

    # Filter to existing files
    input_reports = [p for p in input_reports if p.exists() and p.is_file()]

    if not input_reports:
        logger.error("No valid input reports found")
        sys.exit(1)

    exit_code = generate_summary(input_reports, args.output, args.json_output)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
