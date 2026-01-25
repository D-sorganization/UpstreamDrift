"""Utilities for generating assessment reports and issues."""

from datetime import datetime
from pathlib import Path


def generate_markdown_report(
    category_id: str,
    category_name: str,
    grade: float,
    details: str,
    recommendations: list[str],
    output_dir: Path,
) -> Path:
    """Generate a standardized Markdown report for a category."""
    filename = f"Assessment_{category_id}_{category_name.replace(' ', '_')}.md"
    filepath = output_dir / filename

    content = f"""# Assessment: {category_name}

**Date**: {datetime.now().strftime("%Y-%m-%d")}
**Grade**: {grade:.1f}/10

## Details
{details}

## Recommendations
"""
    for i, rec in enumerate(recommendations, 1):
        content += f"{i}. {rec}\n"

    output_dir.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content, encoding="utf-8")
    return filepath


def generate_issue_document(
    category_id: str,
    category_name: str,
    grade: float,
    details: str,
    output_dir: Path,
) -> Path:
    """Generate a GitHub issue document for a low-scoring category."""
    filename = f"ISSUE_Assessment_{category_id}_{category_name.replace(' ', '_')}.md"
    filepath = output_dir / filename

    content = f"""---
title: "Assessment Finding: Low Score in {category_name}"
labels: jules:assessment, needs-attention
---

# Issue: Low Score in {category_name}

**Grade**: {grade:.1f}/10
**Details**: {details}

## Recommended Actions
- Review the detailed assessment in `docs/assessments/Assessment_{category_id}_{category_name.replace(" ", "_")}.md`
- Create a remediation plan.
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content, encoding="utf-8")
    return filepath
