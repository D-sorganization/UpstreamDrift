"""Shared styling primitives for the three canonical viz views.

Centralised so the overlay, error timecourse, and quality card all agree on
colours, DPI, and font sizing per VISUALIZATION_SPEC.md § Styling.
"""

from __future__ import annotations

# Hex colours mandated by VISUALIZATION_SPEC.md.
COLOR_MEASURED = "#1f77b4"
COLOR_SIMULATED = "#d62728"
COLOR_ERROR = "#7f7f7f"

DPI_PNG = 200
TITLE_FONTSIZE = 13
AXES_FONTSIZE = 11
