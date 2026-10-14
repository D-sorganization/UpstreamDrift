"""CMB-11 (#11662): compose to spec and export to every engine cross-engine parity."""

from __future__ import annotations

import pytest

from src.tools.character_builder.core import CharacterBuilderModel

pytestmark = pytest.mark.unit


def _assert_parity_report(report: dict[str, object]) -> None:
    """Helper verifying standard parity report bounds."""
    assert report["passed"] is True
    assert isinstance(report["max_mass_error_kg"], float)
    assert report["max_mass_error_kg"] <= 1e-9
    assert isinstance(report["max_frame_origin_error_m"], float)
    assert report["max_frame_origin_error_m"] <= 1e-3
    assert report["pinocchio_joint_count"] in (21, 24)


@pytest.mark.parametrize("preset_id", CharacterBuilderModel.preset_ids())
def test_all_presets_cross_engine_export_parity(preset_id: str) -> None:
    """Every shipped preset conserves mass and maintains sub-millimeter frame alignment."""
    model = CharacterBuilderModel(preset_id)
    report = model.verify_cross_engine_parity()
    _assert_parity_report(report)


def test_custom_configuration_cross_engine_export_parity() -> None:
    """A custom anthropometric configuration satisfies cross-engine parity."""
    model = CharacterBuilderModel()
    model.set_parameter("stature_m", 1.82)
    model.set_parameter("mass_kg", 84.0)
    model.set_parameter("club", "driver")
    report = model.verify_cross_engine_parity()
    _assert_parity_report(report)


@pytest.mark.parametrize(
    "mass_tol,frame_tol",
    [
        (0.0, 1e-3),
        (-1e-9, 1e-3),
        (1e-9, 0.0),
        (1e-9, -1.0),
    ],
)
def test_invalid_tolerances_rejected(mass_tol: float, frame_tol: float) -> None:
    """Non-positive mass or frame tolerances raise ValueError."""
    model = CharacterBuilderModel()
    with pytest.raises(ValueError, match="must be > 0"):
        model.verify_cross_engine_parity(
            mass_tol_kg=mass_tol,
            frame_tol_m=frame_tol,
        )
