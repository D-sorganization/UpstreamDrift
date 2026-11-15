"""CLI help for --shooting-fit must not describe a multiple-shooting integrator (#15015)."""

from __future__ import annotations

import pytest

from src.shared.python.motion_matching.pipeline.cli import build_parser

pytestmark = pytest.mark.unit


def test_shooting_fit_help_describes_pelvis_ilc_not_shooting_integrator() -> None:
    """Regression: flag adjusts pelvis command iteratively; it is not BVP shooting."""
    parser = build_parser()
    action = next(act for act in parser._actions if act.dest == "shooting_fit")
    option_strings = action.option_strings
    assert "--shooting-fit" in option_strings
    assert "--pelvis-ilc-passes" in option_strings
    help_lower = (action.help or "").lower()
    assert "pelvis" in help_lower
    assert "iterative" in help_lower or "ilc" in help_lower
    assert "not a" in help_lower
    assert "bvp" in help_lower

    help_text = parser.format_help()
    assert "--pelvis-ilc-passes" in help_text