"""Loads the one shared NSX plot-model golden (NSX-42) for the NSX-43 tests."""

from __future__ import annotations

import json
from typing import Any

from src.shared.python.loop_nullspace.plot_model import NSXPlotModel
from tests.unit.loop_nullspace.plot_model_golden import GOLDEN_PATH


def golden_dict() -> dict[str, Any]:
    """The ``plot_model`` object of ``plot_model_v1.json``, untouched."""
    doc = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    plot_model: dict[str, Any] = doc["plot_model"]
    return plot_model


def golden_model() -> NSXPlotModel:
    return NSXPlotModel.from_dict(golden_dict())
