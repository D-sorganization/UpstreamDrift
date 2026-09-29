"""Regression tests for Sidekick tools-sidebar design tokens."""

from __future__ import annotations

import re

import pytest

from src.shared.python.sidekick.ui.tools_sidebar.design_tokens import (
    SIDEKICK_DESIGN_TOKENS,
    SIDEKICK_TAB_BAR_OBJECT_NAME,
    sidekick_qss,
)


def _qss_rule_body(stylesheet: str, selector: str) -> str:
    match = re.search(
        rf"{re.escape(selector)}\s*\{{(?P<body>.*?)\}}",
        stylesheet,
        re.DOTALL,
    )
    assert match is not None, f"Missing QSS selector: {selector}"
    return match.group("body")


@pytest.mark.unit
def test_sidekick_qss_highlights_unselected_sidebar_tabs_on_hover() -> None:
    """Unselected sidebar tabs must visibly respond to hover."""
    stylesheet = sidekick_qss()
    selector = f"QTabBar#{SIDEKICK_TAB_BAR_OBJECT_NAME}::tab:!selected:hover"

    body = _qss_rule_body(stylesheet, selector)

    assert f"background: {SIDEKICK_DESIGN_TOKENS['color.accent.soft']};" in body
    assert f"border-color: {SIDEKICK_DESIGN_TOKENS['color.border.strong']};" in body
    assert f"color: {SIDEKICK_DESIGN_TOKENS['color.text']};" in body


@pytest.mark.unit
def test_sidekick_qss_keeps_selected_tab_rule_separate_from_hover() -> None:
    """The hover rule should not override the selected tab state."""
    stylesheet = sidekick_qss()
    selected_selector = f"QTabBar#{SIDEKICK_TAB_BAR_OBJECT_NAME}::tab:selected"

    selected_body = _qss_rule_body(stylesheet, selected_selector)

    assert f"background: {SIDEKICK_DESIGN_TOKENS['color.surface']};" in selected_body
    assert "color.accent.soft" not in selected_body
