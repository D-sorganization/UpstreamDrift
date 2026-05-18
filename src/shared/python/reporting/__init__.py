"""Reporting — standardized simulation and calculation report templates.

Public surface::

    from src.shared.python.reporting import (
        ReportSection,
        ReportTemplate,
        REPORT_TEMPLATES,
    )

Implements Epic #5393: Standardized Simulation and Calculation Report Templates.
"""

from __future__ import annotations

from src.shared.python.reporting._templates import REPORT_TEMPLATES
from src.shared.python.reporting._template_engine import ReportSection, ReportTemplate

__all__ = [
    "ReportSection",
    "ReportTemplate",
    "REPORT_TEMPLATES",
]
