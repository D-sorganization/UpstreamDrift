"""Wind-consumer accuracy for the flight shortcomings register (#15485).

PR #15482 review FIX-NEEDED: the register must not imply that only the shot-model
landing simulator sets wind; the ball flight GUI builds wind via
``EnvironmentalConditions.from_altitude(..., wind_velocity=...)``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
REGISTER = ROOT / "docs" / "physics" / "FLIGHT_AND_IMPACT_SHORTCOMINGS_REGISTER.md"

_WIND_HEADING = re.compile(r"^### Wind\s*$", re.MULTILINE)
_NEXT_HEADING = re.compile(r"^### ", re.MULTILINE)

_SHOT_MODEL_PATH = "src/shared/python/sg_optimizer/shot_model/physics_simulator.py"
_BALL_FLIGHT_GUI_PATH = "src/tools/ball_flight_gui/gui.py"

_SOLE_CONSUMER = re.compile(
    r"\b(?:only|sole)\b.+(?:consumer|set wind|sets wind|PipelineLandingSimulator)",
    re.IGNORECASE | re.DOTALL,
)


def extract_wind_section(register_text: str) -> str:
    """Return the markdown body under ``### Wind`` up to the next ``###`` heading."""
    if not register_text or not register_text.strip():
        raise ValueError("register_text must be non-empty")
    match = _WIND_HEADING.search(register_text)
    if match is None:
        raise ValueError("register missing ### Wind heading")
    start = match.end()
    rest = register_text[start:]
    next_match = _NEXT_HEADING.search(rest)
    end = start + next_match.start() if next_match else len(register_text)
    section = register_text[start:end].strip()
    if not section:
        raise ValueError("wind section is empty")
    return section


def test_wind_section_lists_both_ui_and_shot_model_consumers() -> None:
    """DbC: both cited paths exist and the wind section names each consumer."""
    section = extract_wind_section(REGISTER.read_text(encoding="utf-8"))
    assert _BALL_FLIGHT_GUI_PATH in section, "wind section must cite ball flight GUI"
    assert _SHOT_MODEL_PATH in section, "wind section must cite shot-model simulator"
    assert "_build_environment" in section
    assert "_compute_trajectory" in section
    assert not _SOLE_CONSUMER.search(section), (
        "wind section must not claim a sole wind consumer"
    )
    for rel in (_BALL_FLIGHT_GUI_PATH, _SHOT_MODEL_PATH):
        assert (ROOT / rel).is_file(), rel


def test_wind_section_mentions_explicit_pipeline_environment() -> None:
    """SwingBallFlightPipeline accepts an explicit environment; calm is the default."""
    section = extract_wind_section(REGISTER.read_text(encoding="utf-8"))
    assert "SwingBallFlightPipeline" in section
    assert "explicit environment" in section.lower()
