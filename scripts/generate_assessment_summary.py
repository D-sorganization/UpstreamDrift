#!/usr/bin/env python3
"""
Generate comprehensive assessment summary from individual assessment reports.

This script aggregates all A-O assessment results and creates:
1. A comprehensive markdown summary
2. A JSON file with structured metrics
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.shared.python.logging_config import get_logger, setup_logging

# Configure logging using centralized module
setup_logging(use_simple_format=True)
logger = get_logger(__name__)


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

        # Look for findings
        # Simple extraction of bullets under "Findings"
        findings_section = re.search(r"## Findings\n(.*?)\n##", content, re.DOTALL)
        if findings_section:
            findings_text = findings_section.group(1)
            for line in findings_text.split("\n"):
                if line.strip().startswith("- "):
                    issues.append(
                        {
                            "severity": "MAJOR",  # Defaulting to MAJOR as script doesn't discern yet
                            "description": line.strip()[2:],
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

    # Category mapping (A-O)
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
    groups = {
        "Code Quality": {"weight": 0.25, "cats": ["A", "D", "I", "O"]},
        "Testing": {"weight": 0.15, "cats": ["C"]},
        "Documentation": {"weight": 0.10, "cats": ["B"]},
        "Security": {"weight": 0.15, "cats": ["F", "K"]},
        "Performance": {"weight": 0.15, "cats": ["E", "N"]},
        "Ops & Config": {"weight": 0.10, "cats": ["G", "H", "L", "M"]},
        "Design": {"weight": 0.10, "cats": ["J"]},
    }

    # Collect scores and issues
    scores = {}
    all_issues = []

    for report in input_reports:
        # Extract assessment ID from filename (e.g., Assessment_A_Code_Structure.md)
        match = re.search(r"Assessment_([A-O])_", report.name)
        if match:
            assessment_id = match.group(1)
            scores[assessment_id] = extract_score_from_report(report)
            all_issues.extend(extract_issues_from_report(report))

    # Calculate weighted average
    total_weighted_score = 0.0
    total_weight_used = 0.0

    group_scores = {}

    for group_name, group_info in groups.items():
        cat_scores = []
        for cat in group_info["cats"]:
            if cat in scores:
                cat_scores.append(scores[cat])

        if cat_scores:
            avg_group_score = sum(cat_scores) / len(cat_scores)
            group_scores[group_name] = avg_group_score
            total_weighted_score += avg_group_score * group_info["weight"]
            total_weight_used += group_info["weight"]
        else:
            group_scores[group_name] = 0.0

    overall_score = (
        total_weighted_score / total_weight_used if total_weight_used > 0 else 0.0
    )

    # Generate markdown summary
    md_content = f"""# Comprehensive Assessment Summary

**Date**: {datetime.now().strftime("%Y-%m-%d")}
**Generated**: Automated via Jules Assessment Auto-Fix workflow
**Overall Score**: {overall_score:.1f}/10

## Executive Summary

Repository assessment completed across all {len(scores)} categories.

### Overall Health: {overall_score:.1f}/10

### Weighted Average Breakdown

| Group | Weight | Score | Categories |
|-------|--------|-------|------------|
"""
    for group_name, group_info in groups.items():
        score = group_scores.get(group_name, 0.0)
        cats = ", ".join(group_info["cats"])
        md_content += f"| **{group_name}** | {group_info['weight'] * 100:.0f}% | {score:.1f} | {cats} |\n"

    md_content += """
### Individual Category Scores

| Category | Name | Score |
|----------|------|-------|
"""

    for cat_code in sorted(categories.keys()):
        name = categories[cat_code]
        score = scores.get(cat_code, 0.0)
        md_content += f"| **{cat_code}** | {name} | {score:.1f} |\n"

    md_content += f"""
## Findings Summary

Found {len(all_issues)} issues across all categories.

"""
    # Simply list top 10 findings if available
    for i, issue in enumerate(all_issues[:10], 1):
        md_content += f"{i}. {issue['description']} (Source: {issue['source']})\n"

    md_content += """
## Recommendations

1. Address categories with scores below 5.0 immediately.
2. Focus on improving Code Quality and Security as they carry high weight.
3. Maintain documentation to ensure project longevity.

---

*Generated by Jules Assessment Auto-Fix*
"""

    # Save markdown
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w") as f:
        f.write(md_content)

    logger.info(f"✓ Markdown summary saved to {output_md}")

    # Generate JSON metrics
    # Reconstruct category_scores format expected by assess_repository.py
    # assess_repository.py uses: category_scores[cat_code] = { "score": X, "name": Y }

    category_scores_json = {}
    for cat_code, name in categories.items():
        category_scores_json[cat_code] = {
            "score": scores.get(cat_code, 0.0),
            "name": name,
        }

    json_data = {
        "timestamp": datetime.now().isoformat(),
        "overall_score": round(overall_score, 2),
        "category_scores": category_scores_json,
        "group_scores": group_scores,
        "total_issues": len(all_issues),
        "reports_analyzed": len(input_reports),
    }

    # Save JSON
    with open(output_json, "w") as f:
        json.dump(json_data, f, indent=2)

    logger.info(f"✓ JSON metrics saved to {output_json}")

    return 0


def main():
    """Parse CLI arguments and generate assessment summary."""
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
    return exit_code


if __name__ == "__main__":
    sys.exit(main() or 0)
