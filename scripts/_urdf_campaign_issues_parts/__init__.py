"""Issue definitions for ``scripts/_urdf_campaign_issues.py``, split by workflow domain."""

from __future__ import annotations

from _urdf_campaign_issues_parts import (
    architecture,
    audits_docs,
    failures,
    quality,
    testing,
)

ISSUES: list[tuple[str, list[str], str]] = [
    *architecture.ISSUES,
    *failures.ISSUES,
    *quality.ISSUES,
    *testing.ISSUES,
    *audits_docs.ISSUES,
]

__all__ = ["ISSUES"]
