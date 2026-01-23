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
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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
        return 0.0

    except Exception as e:
        logger.warning(f"Could not extract score from {report_path}: {e}")
        return 0.0


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

    # Categories definition
    categories = {
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

    # Weight Groups
    # Code 25%, Testing 15%, Docs 10%, Security 15%, Perf 15%, Ops 10%, Design 10%
    groups = {
        "Code Quality": {"weight": 0.25, "cats": ["A", "D", "I", "K", "O"]},
        "Testing": {"weight": 0.15, "cats": ["C"]},
        "Documentation": {"weight": 0.10, "cats": ["B"]},
        "Security": {"weight": 0.15, "cats": ["F"]},
        "Performance": {"weight": 0.15, "cats": ["E", "N"]},
        "Ops & Config": {"weight": 0.10, "cats": ["G", "H", "L", "M"]},
        "Design": {"weight": 0.10, "cats": ["J"]},
    }

    # Collect scores and issues
    scores = {}
    all_issues = []

    for report in input_reports:
        # Extract assessment ID from filename (e.g., Assessment_A_Results_2026-01-17.md or Assessment_A_Code_Structure.md)
        match = re.search(r"Assessment_([A-O])_", report.name)
        if match:
            assessment_id = match.group(1)
            # Use the latest score if multiple files exist for same category (overwriting)
            scores[assessment_id] = extract_score_from_report(report)
            all_issues.extend(extract_issues_from_report(report))

    # Calculate weighted average
    overall_score = 0.0
    total_group_weight = 0.0

    group_scores = {}

    for group_name, info in groups.items():
        cats = info["cats"]
        weight = info["weight"]

        # Calculate average for this group
        group_total = 0.0
        group_count = 0
        for cat in cats:
            if cat in scores:
                group_total += scores[cat]
                group_count += 1

        if group_count > 0:
            avg_score = group_total / group_count
            group_scores[group_name] = avg_score
            overall_score += avg_score * weight
            total_group_weight += weight
        else:
            group_scores[group_name] = 0.0

    # Normalize if some groups are missing (shouldn't happen with full run)
    if total_group_weight > 0:
        overall_score = overall_score / total_group_weight
    else:
        overall_score = 0.0

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

| Category | Name | Score | Group |
|----------|------|-------|-------|
"""

    for cat_id in sorted(categories.keys()):
        if cat_id in scores:
            name = categories[cat_id]
            score = scores[cat_id]

            # Find group
            group_name = "Unknown"
            for g_name, g_info in groups.items():
                if cat_id in g_info["cats"]:
                    group_name = g_name
                    break

            md_content += f"| **{cat_id}** | {name} | {score:.1f} | {group_name} |\n"

    md_content += """
### Group Breakdown

| Group | Weight | Score |
|-------|--------|-------|
"""
    for g_name, info in groups.items():
        score = group_scores.get(g_name, 0.0)
        md_content += f"| {g_name} | {info['weight']*100:.0f}% | {score:.1f} |\n"

    md_content += f"""
## Critical Issues

Found {len(critical_issues)} critical issues requiring immediate attention:

"""
    if critical_issues:
        for i, issue in enumerate(critical_issues[:10], 1):
            md_content += f"{i}. **[{issue['severity']}]** {issue['description']} (Source: {issue['source']})\n"
    else:
        md_content += "None.\n"

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
                "name": categories[k],
            }
            for k, v in scores.items()
            if k in categories
        },
        "group_scores": group_scores,
        "critical_issues": critical_issues,
        "total_issues": len(all_issues),
        "reports_analyzed": len(input_reports),
    }

    # Save JSON
    with open(output_json, "w") as f:
        json.dump(json_data, f, indent=2)

    logger.info(f"✓ JSON metrics saved to {output_json}")

    return 0


def main():
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
    input_reports = []
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
        return 1

    exit_code = generate_summary(input_reports, args.output, args.json_output)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
