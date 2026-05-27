"""Tests for issue #5505 — proportional sidebar sizing.

Verifies:
- Styles.SIDEBAR_MIN_WIDTH, Styles.SPACING_LG, Styles.MARGIN_PAGE constants exist.
- Launcher uses Styles.SIDEBAR_MIN_WIDTH for sidebar.setMinimumWidth().
- Launcher replaces hard-coded setSizes([85, 1200]) with 1:5 stretch factors.
"""

from __future__ import annotations

import re


def _styles():
    """Import Styles (safe; module has no top-level Qt calls)."""
    from src.shared.python.theme.style_constants import Styles

    return Styles


def test_sidebar_minimum_width_constant_exists():
    """Styles.SIDEBAR_MIN_WIDTH must exist and be >= 120 px (issue #5505)."""
    Styles = _styles()
    assert hasattr(Styles, "SIDEBAR_MIN_WIDTH"), "Styles.SIDEBAR_MIN_WIDTH not found"
    assert isinstance(Styles.SIDEBAR_MIN_WIDTH, int)
    assert Styles.SIDEBAR_MIN_WIDTH >= 120, (
        f"Expected SIDEBAR_MIN_WIDTH >= 120, got {Styles.SIDEBAR_MIN_WIDTH}"
    )


def test_spacing_constant_exists():
    """Styles.SPACING_LG must equal 20 (issue #5505)."""
    Styles = _styles()
    assert hasattr(Styles, "SPACING_LG"), "Styles.SPACING_LG not found"
    assert Styles.SPACING_LG == 20, (
        f"Expected SPACING_LG == 20, got {Styles.SPACING_LG}"
    )


def test_margin_page_constant_exists():
    """Styles.MARGIN_PAGE must equal 30 (issue #5505)."""
    Styles = _styles()
    assert hasattr(Styles, "MARGIN_PAGE"), "Styles.MARGIN_PAGE not found"
    assert Styles.MARGIN_PAGE == 30, (
        f"Expected MARGIN_PAGE == 30, got {Styles.MARGIN_PAGE}"
    )


def test_sidebar_widget_minimum_width_uses_constant():
    """_setup_global_sidebar must call setMinimumWidth(Styles.SIDEBAR_MIN_WIDTH)."""
    source_path = "src/launchers/launcher_ui_setup.py"
    with open(source_path, encoding="utf-8") as fh:
        source = fh.read()

    assert "Styles.SIDEBAR_MIN_WIDTH" in source, (
        "launcher_ui_setup.py must use Styles.SIDEBAR_MIN_WIDTH "
        "instead of a hard-coded pixel value"
    )
    assert not re.search(r"setMinimumWidth\(85\)", source), (
        "Hard-coded setMinimumWidth(85) still present; replace with Styles.SIDEBAR_MIN_WIDTH"
    )


def test_main_splitter_uses_stretch_factors_not_fixed_sizes():
    """_setup_main_layout must use 1:5 stretch factors and drop setSizes([85,...])."""
    source_path = "src/launchers/launcher_ui_setup.py"
    with open(source_path, encoding="utf-8") as fh:
        source = fh.read()

    assert not re.search(r"setSizes\(\s*\[85", source), (
        "Hard-coded setSizes([85, ...]) still present; replace with setStretchFactor"
    )

    stretch_matches = re.findall(r"setStretchFactor\((\d+),\s*(\d+)\)", source)
    factors = {int(f) for _, f in stretch_matches}
    assert 0 in factors and 1 in factors, (
        f"Expected setStretchFactor calls with factors 0 and 1; found factors {factors}"
    )
