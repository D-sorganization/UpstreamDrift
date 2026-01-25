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

# Add project root to path for imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.script_utils import run_main, setup_script_logging  # noqa: E402

# Configure logging using centralized utility
logger = setup_script_logging(__name__)


def extract_score_from_report(report_path: Path) -> float:
    """Extract numerical score from assessment report."""
    try:
        content = report_path.read_text(encoding="utf-8", errors="ignore")

        # Look for score patterns like "Overall: 8.5" or "Score: 8.5/10"
        patterns = [
            r"Overall.*?(\d+\.?\d*)",
            r"Score.*?(\d+\.?\d*)",
            r"\*\*(\d+\.?\d*)\*\*.*?/10",
            r"\*\*Grade\*\*: (\d+\.?\d*)/10",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return float(match.group(1))

        return 7.0
    except Exception as e:
        logger.warning(f"Could not extract score from {report_path}: {e}")
        return 7.0


def extract_issues_from_report(report_path: Path) -> list[dict[str, Any]]:
    """Extract issues/findings from assessment report."""
    issues = []
    try:
        content = report_path.read_text(encoding="utf-8", errors="ignore")
        findings_section = re.search(r"## Findings\n(.*?)\n##", content, re.DOTALL)
        if findings_section:
            findings_text = findings_section.group(1)
            for line in findings_text.split("\n"):
                if line.strip().startswith("- "):
                    issues.append(
                        {
                            "severity": "MAJOR",
                            "description": line.strip()[2:],
                            "source": report_path.stem,
                        }
                    )
    except Exception as e:
        logger.warning(f"Could not extract issues from {report_path}: {e}")
    return issues


def _calculate_weighted_score(
    scores: dict[str, float],
    groups: dict[str, dict[str, Any]],
) -> tuple[float, dict[str, float]]:
    """Calculate weighted scores based on group weights."""
    total_weighted_score = 0.0
    total_weight_used = 0.0
    group_scores = {}

    for group_name, group_info in groups.items():
        cat_scores = [scores[cat] for cat in group_info["cats"] if cat in scores]
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
    return overall_score, group_scores


def _build_markdown_report(
    overall_score: float,
    group_scores: dict[str, float],
    groups: dict[str, dict[str, Any]],
    scores: dict[str, float],
    categories: dict[str, str],
    all_issues: list[dict[str, Any]],
) -> str:
    """Construct markdown report content."""
    md_content = f"""# Comprehensive Assessment Summary

**Date**: {datetime.now().strftime("%Y-%m-%d")}
**Generated**: Automated via DRY-Orthogonality Overhaul
**Overall Health Score**: {overall_score:.1f}/10

## Executive Summary

Repository assessment completed across {len(scores)} categories.

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
    for i, issue in enumerate(all_issues[:10], 1):
        md_content += f"{i}. {issue['description']} (Source: {issue['source']})\n"

    md_content += """
## Recommendations

1. Address categories with scores below 5.0 immediately (DRY/Orthogonality prioritised).
2. Focus on improving Code Quality and Security as they carry high weight.
3. Maintain documentation to ensure project longevity.

---
*Generated by Golf Modeling Suite Assessment Tooling*
"""
    return md_content


def generate_summary(
    input_reports: list[Path],
    output_md: Path,
    output_json: Path,
) -> int:
    """Generate comprehensive summary from assessment reports."""
    logger.info(f"Generating assessment summary from {len(input_reports)} reports...")

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

    groups = {
        "Code Quality": {"weight": 0.25, "cats": ["A", "D", "I", "O"]},
        "Testing": {"weight": 0.15, "cats": ["C"]},
        "Documentation": {"weight": 0.10, "cats": ["B"]},
        "Security": {"weight": 0.15, "cats": ["F", "K"]},
        "Performance": {"weight": 0.15, "cats": ["E", "N"]},
        "Ops & Config": {"weight": 0.10, "cats": ["G", "H", "L", "M"]},
        "Design": {"weight": 0.10, "cats": ["J"]},
    }

    scores = {}
    all_issues = []

    for report in input_reports:
        match = re.search(r"Assessment_([A-O])_", report.name)
        if match:
            assessment_id = match.group(1)
            scores[assessment_id] = extract_score_from_report(report)
            all_issues.extend(extract_issues_from_report(report))

    overall_score, group_scores = _calculate_weighted_score(scores, groups)

    md_content = _build_markdown_report(
        overall_score, group_scores, groups, scores, categories, all_issues
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(md_content)
    logger.info(f"[OK] Markdown summary saved to {output_md}")

    category_scores_json = {
        code: {"score": scores.get(code, 0.0), "name": name}
        for code, name in categories.items()
    }
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "overall_score": round(overall_score, 2),
        "category_scores": category_scores_json,
        "group_scores": group_scores,
        "total_issues": len(all_issues),
        "reports_analyzed": len(input_reports),
    }

    output_json.write_text(json.dumps(json_data, indent=2))
    logger.info(f"[OK] JSON metrics saved to {output_json}")

    return 0


def main():
    """Parse CLI arguments and generate assessment summary."""
    parser = argparse.ArgumentParser(description="Generate assessment summary")
    parser.add_argument(
        "--input", nargs="+", type=Path, required=True, help="Input reports"
    )
    parser.add_argument("--output", required=True, type=Path, help="Output MD")
    parser.add_argument("--json-output", required=True, type=Path, help="Output JSON")

    args = parser.parse_args()

    input_reports = []
    for pattern in args.input:
        if "*" in str(pattern):
            input_reports.extend(Path(".").glob(str(pattern)))
        else:
            input_reports.append(pattern)

    input_reports = [p for p in input_reports if p.exists() and p.is_file()]

    if not input_reports:
        logger.error("No valid input reports found")
        return 1

    return generate_summary(input_reports, args.output, args.json_output)


if __name__ == "__main__":
    run_main(main, logger)
