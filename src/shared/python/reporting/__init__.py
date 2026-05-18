"""Reporting package — standardised simulation report generation.

Public surface::

    from src.shared.python.reporting import (
        ReportTemplate,
        AgenticSummaryGenerator,
        GLOBAL_REPORT_REGISTRY,
    )

Implements part of Issue #5423: Global Report Templates and Agentic Summaries.
"""

from __future__ import annotations

from src.shared.python.reporting._template_engine import ReportTemplate
from src.shared.python.reporting._agentic_summary import AgenticSummaryGenerator
from src.shared.python.reporting._registry import GLOBAL_REPORT_REGISTRY

__all__ = [
    "ReportTemplate",
    "AgenticSummaryGenerator",
    "GLOBAL_REPORT_REGISTRY",
]
