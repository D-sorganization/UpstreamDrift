"""One source of truth for the workbench input ranges (issue #9545, PR #11516).

The v1 API route and the PyQt panels must accept exactly the same bounds.
Both read :data:`src.tools.bunker_shot_gui.design.INPUT_RANGES`; these tests
fail if either consumer drifts back to a private copy.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from src.tools.bunker_shot_gui.design import INPUT_RANGES

pytestmark = [pytest.mark.contract, pytest.mark.integration]

#: API request field -> shared range key (the API names the tolerance shortly).
_API_FIELDS = {
    "BunkerDesignV1": (
        "loft_deg",
        "marketed_bounce_deg",
        "sole_width_mm",
        "entry_height_mm",
        "leading_edge_radius_mm",
        "camber_area_mm2",
        "heel_relief_fraction",
        "toe_relief_fraction",
    ),
    "BunkerSandV1": ("firmness_kg_per_cm2",),
    "BunkerSwingV1": (
        "clubhead_speed_mps",
        "attack_angle_deg",
        "face_open_deg",
        "shaft_lean_deg",
        "entry_distance_behind_ball_m",
        "ball_depth_m",
    ),
    "BunkerObjectiveV1": ("target_carry_m", "tolerance_fraction"),
}
_API_KEY = {"tolerance_fraction": "carry_tolerance_fraction"}

#: Panel spin box -> (shared range key, display scale from the model unit).
_DESIGN_SPINS = {
    "_loft": ("loft_deg", 1.0),
    "_bounce": ("marketed_bounce_deg", 1.0),
    "_sole_width": ("sole_width_mm", 1.0),
    "_entry_height": ("entry_height_mm", 1.0),
    "_leading_radius": ("leading_edge_radius_mm", 1.0),
    "_camber_area": ("camber_area_mm2", 1.0),
    "_heel_relief": ("heel_relief_fraction", 1.0),
    "_toe_relief": ("toe_relief_fraction", 1.0),
}
_CONDITION_SPINS = {
    "_firmness": ("firmness_kg_per_cm2", 1.0),
    "_speed": ("clubhead_speed_mps", 1.0),
    "_attack": ("attack_angle_deg", 1.0),
    "_face_open": ("face_open_deg", 1.0),
    "_shaft_lean": ("shaft_lean_deg", 1.0),
    "_entry": ("entry_distance_behind_ball_m", 1e3),
    "_ball_depth": ("ball_depth_m", 1e3),
    "_target_carry": ("target_carry_m", 1.0),
    "_tolerance": ("carry_tolerance_fraction", 1.0),
}


def _field_bounds(model: Any, name: str) -> tuple[float, float]:
    """The ``ge``/``le`` bounds pydantic enforces on one field."""
    bounds: dict[str, float] = {}
    for item in model.model_fields[name].metadata:
        for attr in ("ge", "le"):
            if hasattr(item, attr):
                bounds[attr] = getattr(item, attr)
    return bounds["ge"], bounds["le"]


def test_every_shared_range_has_a_consumer_in_both_paths() -> None:
    api_keys = {
        _API_KEY.get(name, name) for names in _API_FIELDS.values() for name in names
    }
    panel_keys = {
        key for key, _ in (*_DESIGN_SPINS.values(), *_CONDITION_SPINS.values())
    }
    assert set(INPUT_RANGES) == api_keys == panel_keys


def test_api_bounds_are_the_shared_ranges() -> None:
    pytest.importorskip("fastapi")
    from src.api.routes import bunker_workbench

    assert bunker_workbench.INPUT_RANGES is INPUT_RANGES
    for model_name, fields in _API_FIELDS.items():
        model = getattr(bunker_workbench, model_name)
        for name in fields:
            expected = INPUT_RANGES[_API_KEY.get(name, name)]
            assert _field_bounds(model, name) == expected, (model_name, name)


def test_panel_bounds_are_the_shared_ranges() -> None:
    pytest.importorskip("PyQt6", reason="the workbench panels need a Qt binding")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from src.tools.bunker_shot_gui.widgets import ConditionPanel, DesignPanel

    _app = QApplication.instance() or QApplication([])
    for panel, spins in (
        (DesignPanel("A", "left", "sm9_58_m"), _DESIGN_SPINS),
        (ConditionPanel(), _CONDITION_SPINS),
    ):
        for attr, (key, scale) in spins.items():
            box = getattr(panel, attr)
            low, high = INPUT_RANGES[key]
            assert (box.minimum(), box.maximum()) == pytest.approx(
                (low * scale, high * scale)
            ), attr
