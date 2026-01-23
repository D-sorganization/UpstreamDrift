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
from typing import Any, cast

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

        # Look for findings (lines starting with "- ")
        findings_match = re.search(r"## Findings\n\n(.*?)\n\n##", content, re.DOTALL)
        if findings_match:
            findings_block = findings_match.group(1)
            for line in findings_block.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    issues.append(
                        {
                            "description": line[2:].strip(),
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

    # Category definitions
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

    # Group definitions and weights
    # Groups: Code Quality, Testing, Documentation, Security, Performance, Ops & Config, Design
    groups: dict[str, dict[str, Any]] = {
        "Code Quality": {"weight": 0.25, "categories": ["A", "D", "I", "O"]},
        "Testing": {"weight": 0.15, "categories": ["C"]},
        "Documentation": {"weight": 0.10, "categories": ["B"]},
        "Security": {"weight": 0.15, "categories": ["F"]},
        "Performance": {"weight": 0.15, "categories": ["E", "N"]},
        "Ops & Config": {"weight": 0.10, "categories": ["G", "H", "L", "M"]},
        "Design": {"weight": 0.10, "categories": ["J", "K"]},
    }

    # Collect scores and issues
    scores: dict[str, float] = {}
    all_issues = []

    for report in input_reports:
        # Extract assessment ID from filename (e.g., Assessment_A_Code_Structure.md)
        match = re.search(r"Assessment_([A-O])_", report.name)
        if match:
            assessment_id = match.group(1)
            scores[assessment_id] = extract_score_from_report(report)
            all_issues.extend(extract_issues_from_report(report))

    # Calculate weighted average
    group_scores = {}
    total_weighted_score = 0.0
    total_weight = 0.0

    for group_name, info in groups.items():
        cats = cast(list[str], info["categories"])
        weight = cast(float, info["weight"])

        # Calculate average score for the group
        group_sum = 0.0
        group_count = 0
        for cat in cats:
            if cat in scores:
                group_sum += scores[cat]
                group_count += 1

        if group_count > 0:
            group_avg = group_sum / group_count
            group_scores[group_name] = group_avg
            total_weighted_score += group_avg * weight
            total_weight += weight
        else:
            group_scores[group_name] = 0.0
            # If a group is missing entirely, we effectively treat it as 0 for now,
            # or we could skip its weight. Assuming all categories should exist.
            total_weight += weight

    # Normalize if weights don't sum to 1.0
    overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0

    # Generate markdown summary
    md_content = f"""# Comprehensive Assessment Summary

**Date**: {datetime.now().strftime("%Y-%m-%d")}
**Generated**: Automated via Jules Assessment Generator
**Overall Score**: {overall_score:.2f}/10

## Executive Summary

Repository assessment completed across all {len(scores)} categories.

### Overall Health: {overall_score:.2f}/10

### Group Scores (Weighted)

| Group | Weight | Categories | Score |
|-------|--------|------------|-------|
"""

    for group_name, info in groups.items():
        score = group_scores.get(group_name, 0.0)
        weight = cast(float, info["weight"])
        cats_str = ", ".join(cast(list[str], info["categories"]))
        md_content += f"| **{group_name}** | {weight*100:.0f}% | {cats_str} | {score:.2f} |\n"

    md_content += """
### Detailed Category Scores

| Category | Name | Score |
|----------|------|-------|
"""

    for cat_code in sorted(categories.keys()):
        cat_name = categories[cat_code]
        score = scores.get(cat_code, 0.0)
        md_content += f"| **{cat_code}** | {cat_name} | {score} |\n"

    md_content += """
## Top 5 Recommendations

"""

    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    low_scores = sorted_scores[:5]

    for cat_code, score in low_scores:
        cat_name = categories.get(cat_code, "Unknown")
        md_content += f"1. **{cat_name} ({cat_code})**: Score {score}/10. See `Assessment_{cat_code}_*.md` for details.\n"

    md_content += """
## Next Steps

1. Review issues in `docs/assessments/issues/`.
2. Address categories with scores below 5/10.
3. Re-run assessment after improvements.

---

*Generated by Jules Assessment Generator*
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
        "group_scores": group_scores,
        "category_scores": {
            k: {
                "score": v,
                "name": categories.get(k, "Unknown"),
            }
            for k, v in scores.items()
        },
        "reports_analyzed": len(input_reports),
        "total_issues": len(all_issues),
        "issues": all_issues,
    }

    # Save JSON
    with open(output_json, "w") as f:
        json.dump(json_data, f, indent=2)

    logger.info(f"✓ JSON metrics saved to {output_json}")

    return 0


def main() -> int:
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
        return 1

    return generate_summary(input_reports, args.output, args.json_output)


if __name__ == "__main__":
    sys.exit(main())
