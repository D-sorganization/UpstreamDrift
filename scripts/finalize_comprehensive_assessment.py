#!/usr/bin/env python3
"""
Finalize the Comprehensive Assessment Report by merging General, Completist, and Pragmatic data.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.script_utils import run_main, setup_script_logging

logger = setup_script_logging(__name__)

DOCS_DIR = _REPO_ROOT / "docs" / "assessments"
COMPLETIST_REPORT = DOCS_DIR / "completist" / "COMPLETIST_LATEST.md"
PRAGMATIC_REPORT = DOCS_DIR / "pragmatic_programmer" / "review_2026-01-31.json"
SUMMARY_JSON = DOCS_DIR / "assessment_summary.json"
OUTPUT_MD = DOCS_DIR / "Comprehensive_Assessment.md"


def load_general_data():
    if not SUMMARY_JSON.exists():
        logger.error(f"Summary JSON not found: {SUMMARY_JSON}")
        return None
    with open(SUMMARY_JSON, "r") as f:
        return json.load(f)


def load_pragmatic_data():
    if not PRAGMATIC_REPORT.exists():
        logger.warning(f"Pragmatic report not found: {PRAGMATIC_REPORT}")
        return {"issues": []}
    with open(PRAGMATIC_REPORT, "r") as f:
        return json.load(f)


def load_completist_data():
    if not COMPLETIST_REPORT.exists():
        logger.warning(f"Completist report not found: {COMPLETIST_REPORT}")
        return 0
    content = COMPLETIST_REPORT.read_text(encoding="utf-8")
    # Extract Critical Gaps count
    match = re.search(r"\*\*Critical Gaps\*\*: (\d+)", content)
    if match:
        return int(match.group(1))
    return 0


def calculate_scores(general_data, critical_gaps, pragmatic_issues):
    general_score = general_data.get("overall_score", 0.0)

    # Completist Score: 10 - (Critical Gaps * 0.1), min 0
    completist_score = max(0.0, 10.0 - (critical_gaps * 0.1))

    # Pragmatic Score: 10 - (Issues * 0.1), min 0
    pragmatic_score = max(0.0, 10.0 - (len(pragmatic_issues) * 0.1))

    unified_grade = (general_score + completist_score + pragmatic_score) / 3.0

    return {
        "general": general_score,
        "completist": completist_score,
        "pragmatic": pragmatic_score,
        "unified": unified_grade
    }


def generate_recommendations(general_data, critical_gaps, pragmatic_issues):
    recommendations = []

    # General recommendations (from low scoring categories)
    for code, info in general_data.get("category_scores", {}).items():
        if info["score"] < 6.0:
            recommendations.append({
                "source": "General Assessment",
                "text": f"Improve {info['name']} (Score: {info['score']}) - See Assessment {code}",
                "priority": 10 - info["score"]
            })

    # Completist recommendations
    if critical_gaps > 0:
        recommendations.append({
            "source": "Completist Audit",
            "text": f"Address {critical_gaps} Critical Implementation Gaps",
            "priority": 9.0
        })

    # Pragmatic recommendations
    if pragmatic_issues:
        # Group by principle
        principles = {}
        for issue in pragmatic_issues:
            p = issue.get("principle", "General")
            principles[p] = principles.get(p, 0) + 1

        for p, count in principles.items():
            recommendations.append({
                "source": "Pragmatic Review",
                "text": f"Fix {count} violations of {p} principle",
                "priority": 8.0
            })

    # Sort by priority
    recommendations.sort(key=lambda x: x["priority"], reverse=True)
    return recommendations[:10]


def main():
    logger.info("Finalizing Comprehensive Assessment Report...")

    general_data = load_general_data()
    if not general_data:
        return 1

    pragmatic_data = load_pragmatic_data()
    pragmatic_issues = pragmatic_data.get("issues", [])

    critical_gaps = load_completist_data()

    scores = calculate_scores(general_data, critical_gaps, pragmatic_issues)
    recommendations = generate_recommendations(general_data, critical_gaps, pragmatic_issues)

    # Generate Markdown
    md = [
        f"# Comprehensive Assessment Report",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        f"**Unified Grade**: {scores['unified']:.1f}/10",
        "",
        "## Unified Scorecard",
        "| Component | Score | Weight | Status |",
        "|---|---|---|---|",
        f"| **General Code Quality** | {scores['general']:.1f}/10 | 33% | {'🟢' if scores['general'] > 7 else '🔴'} |",
        f"| **Completist Audit** | {scores['completist']:.1f}/10 | 33% | {'🟢' if scores['completist'] > 7 else '🔴'} |",
        f"| **Pragmatic Review** | {scores['pragmatic']:.1f}/10 | 33% | {'🟢' if scores['pragmatic'] > 7 else '🔴'} |",
        "",
        "## Top 10 Unified Recommendations",
    ]

    for i, rec in enumerate(recommendations, 1):
        md.append(f"{i}. **[{rec['source']}]** {rec['text']}")

    md.append("")
    md.append("## Detailed Assessments")

    # Append General Assessment Summary
    md.append("### General Assessment (Categories A-O)")
    md.append("See individual `docs/assessments/Assessment_X_Category.md` files for details.")
    md.append("")

    # Reconstruct the table from general data
    md.append("| Category | Name | Score |")
    md.append("|---|---|---|")
    for code in sorted(general_data["category_scores"].keys()):
        info = general_data["category_scores"][code]
        md.append(f"| {code} | {info['name']} | {info['score']:.1f} |")

    md.append("")

    # Append Completist Details
    md.append("### Completist Audit")
    md.append(f"- **Critical Gaps**: {critical_gaps}")
    md.append("- See `docs/assessments/completist/COMPLETIST_LATEST.md` for full report.")
    md.append("")

    # Append Pragmatic Details
    md.append("### Pragmatic Programmer Review")
    md.append(f"- **Total Issues**: {len(pragmatic_issues)}")
    if pragmatic_issues:
        md.append("| Principle | Severity | Title |")
        md.append("|---|---|---|")
        for issue in pragmatic_issues[:10]:
            md.append(f"| {issue.get('principle')} | {issue.get('severity')} | {issue.get('title')} |")
        if len(pragmatic_issues) > 10:
            md.append(f"\n*...and {len(pragmatic_issues) - 10} more issues.*")

    md.append("")
    md.append("---")
    md.append("*Generated by Finalize Assessment Script*")

    OUTPUT_MD.write_text("\n".join(md), encoding="utf-8")
    logger.info(f"Report written to {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    run_main(main, logger)
