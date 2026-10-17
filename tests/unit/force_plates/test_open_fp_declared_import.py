"""Declared plate facts and stance placement for a trial import (OF-07, #13087).

The desktop declarations dialog and the plate-overlay API build the same
``PlateDeclarations`` and ``TargetedPlacement`` from plain values, so the
conversion lives once in ``open_fp_format.declared_import``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.force_plates import PlateUncertainty
from src.shared.python.force_plates.open_fp_format import PlateDeclarations
from src.shared.python.force_plates.open_fp_format.declared_import import (
    DeclaredPlate,
    DeclaredStance,
)
from src.shared.python.impact_parameters.target_frame import UD_DEFAULT_TARGET_DIR

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _plate(**changes: object) -> DeclaredPlate:
    values: dict[str, object] = {
        "force_floor_n": 2.0,
        "force_fraction": 0.01,
        "cop_m": 0.010,
        "sensing_depth_sigma_m": 0.002,
        "uncertainty_source": "declared hypothesis for tests",
        "sensing_depth_m": 0.015,
        "capacity_n": 3923.0,
    }
    values.update(changes)
    return DeclaredPlate(**values)  # type: ignore[arg-type]


def test_plate_values_become_declarations() -> None:
    declared = _plate().to_declarations()
    assert isinstance(declared, PlateDeclarations)
    assert declared.uncertainty == PlateUncertainty(
        force_floor_n=2.0,
        force_fraction=0.01,
        cop_m=0.010,
        sensing_depth_sigma_m=0.002,
        source="declared hypothesis for tests",
    )
    assert declared.sensing_depth_m == 0.015
    assert declared.capacity_n == 3923.0


def test_device_label_is_kept_or_left_to_the_reader_default() -> None:
    assert _plate(device="Lab pair A").to_declarations().device == "Lab pair A"
    default = PlateDeclarations(
        uncertainty=_plate().to_declarations().uncertainty,
        sensing_depth_m=0.015,
        capacity_n=3923.0,
    ).device
    assert _plate().to_declarations().device == default


@pytest.mark.parametrize(
    ("field", "value"),
    [("capacity_n", -1.0), ("force_floor_n", 0.0), ("uncertainty_source", " ")],
)
def test_invalid_plate_values_are_refused(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _plate(**{field: value}).to_declarations()


def test_stance_builds_a_targeted_placement() -> None:
    stance = DeclaredStance(handedness="RH", origin_m=(1.0, 2.0, 0.0))
    assert stance.target_dir == UD_DEFAULT_TARGET_DIR
    targeted = stance.to_placement(frame_label="declared")
    assert targeted.handedness == "RH"
    assert targeted.frame_id == f"declared:target_{UD_DEFAULT_TARGET_DIR}"
    np.testing.assert_allclose(targeted.placement.origin_m, (1.0, 2.0, 0.0))


def test_left_handed_stance_uses_a_left_handed_frame() -> None:
    stance = DeclaredStance(handedness="LH", origin_m=(0.0, 0.0, 0.0))
    assert stance.to_placement(frame_label="declared").handedness == "LH"


@pytest.mark.parametrize("handedness", ["right", "", "rh"])
def test_unknown_handedness_is_refused(handedness: str) -> None:
    stance = DeclaredStance(handedness=handedness, origin_m=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="handedness"):
        stance.to_placement(frame_label="declared")


def test_vertical_target_direction_is_refused() -> None:
    stance = DeclaredStance(
        handedness="RH", origin_m=(0.0, 0.0, 0.0), target_dir=(0.0, 0.0, 1.0)
    )
    with pytest.raises(ValueError):
        stance.to_placement(frame_label="declared")


def test_frame_label_is_required() -> None:
    stance = DeclaredStance(handedness="RH", origin_m=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="frame_label"):
        stance.to_placement(frame_label=" ")
