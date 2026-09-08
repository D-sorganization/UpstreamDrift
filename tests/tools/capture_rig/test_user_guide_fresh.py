"""The generated user guide must match the workflow step model (#9665)."""

from __future__ import annotations

import pytest

from scripts.generate_mocap_user_guide import OUTPUT, render
from src.tools.capture_rig.workflow import STEPS

pytestmark = pytest.mark.unit


def test_user_guide_is_fresh() -> None:
    assert OUTPUT.is_file(), "run python3 -m scripts.generate_mocap_user_guide"
    assert OUTPUT.read_text(encoding="utf-8") == render(STEPS)


def test_user_guide_states_every_requirement() -> None:
    text = render(STEPS)
    for step in STEPS:
        assert step.title in text
        for requirement in step.requirements:
            assert requirement in text
