"""Launcher layout metrics (pixel ints for Qt layout APIs; issue #8972).

UD-owned home for the spacing/margin constants the launcher UI uses.
They cannot live on ``style_constants.Styles`` because that module is a
Tools-owned child copy which UpstreamDrift must not edit directly
(tests/unit/repo_hygiene/test_tools_child_copy_contract.py).
"""

from __future__ import annotations


class LayoutMetrics:
    """Pixel spacing/margin constants for launcher layouts."""

    SPACING_SM = 6
    """Small spacing between tightly related widgets (px)."""

    SPACING_MD = 12
    """Default spacing between grouped widgets (px)."""

    SPACING_LG = 16
    """Large spacing between page-level sections (px)."""

    MARGIN_PAGE = 24
    """Uniform page content margin (px)."""

    SIDEBAR_MIN_WIDTH = 240
    """Minimum width of the launcher navigation sidebar (px)."""

    RADIUS_SM = 6
    """Corner radius of chips, badges and toolbar frames (px)."""

    ICON_BUTTON_WIDTH = 32
    """Width of a square icon-only button such as a file browser "…" (px)."""

    TRANSPORT_BUTTON_HEIGHT = 40
    """Height of a primary transport control (Record / Stop) (px)."""

    PROGRESS_BAR_HEIGHT = 8
    """Height of a slim, text-free progress bar (px)."""

    READOUT_MIN_WIDTH = 180
    """Minimum width of a clock-style readout so it does not jitter (px)."""
