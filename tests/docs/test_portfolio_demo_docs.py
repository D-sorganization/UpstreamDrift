from __future__ import annotations

import csv
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
DEMO_DOC = ROOT / "docs" / "portfolio" / "golf_modeling_demo.md"
DEMO_OUTPUT = ROOT / "docs" / "portfolio" / "golf_modeling_demo_output.csv"
README = ROOT / "README.md"


def test_readme_links_to_portfolio_demo() -> None:
    readme = README.read_text(encoding="utf-8")

    assert "docs/portfolio/golf_modeling_demo.md" in readme


def test_portfolio_demo_has_reproducible_contract() -> None:
    doc = DEMO_DOC.read_text(encoding="utf-8")

    required_phrases = [
        'python -m pip install -e ".[dev,rust]"',
        "python examples/basic_flight_simulation.py",
        "golf_modeling_demo_output.csv",
        "Measured input",
        "Environmental assumption",
        "model-conditioned outputs",
        "not a validated coaching prescription",
    ]
    for phrase in required_phrases:
        assert phrase in doc


def test_portfolio_demo_output_artifact_is_parseable() -> None:
    with DEMO_OUTPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert {"quantity", "category", "value", "unit", "source"} <= set(rows[0])

    quantities = {row["quantity"] for row in rows}
    categories = {row["category"] for row in rows}

    assert {"ball_speed", "carry_distance", "peak_height", "flight_time"} <= quantities
    assert {"measured_input", "assumption", "simulated_output"} <= categories

    carry_m = [
        row
        for row in rows
        if row["quantity"] == "carry_distance" and row["unit"] == "m"
    ]
    assert carry_m
    assert float(carry_m[0]["value"]) > 0.0


def test_portfolio_demo_import_path_still_resolves() -> None:
    module = importlib.import_module("src.shared.python.physics.ball_flight_physics")

    assert hasattr(module, "BallFlightSimulator")
    assert hasattr(module, "LaunchConditions")
