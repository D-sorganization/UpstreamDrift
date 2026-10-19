"""Regression: CMB-11 Pinocchio parity uses the full neutral coordinate set.

``_verify_pinocchio`` once called ``frame_poses({})``, which the strict native
``configuration()`` contract rejects, so every CMB-11 parity case failed
whenever real Pinocchio was installed. CI's unit lane mocks Pinocchio and
never reached that branch.
"""

from __future__ import annotations

import types

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.requires_pinocchio]

pinocchio = pytest.importorskip("pinocchio")
if not isinstance(pinocchio, types.ModuleType) or not hasattr(pinocchio, "__file__"):
    pytest.skip("real Pinocchio is not installed", allow_module_level=True)

from src.engines.physics_engines.pinocchio.python.native_model import (  # noqa: E402
    FullBodyPinocchioModel,
)
from src.tools.character_builder.core import CharacterBuilderModel  # noqa: E402


@pytest.fixture(scope="module")
def pin_model() -> FullBodyPinocchioModel:
    return FullBodyPinocchioModel(CharacterBuilderModel().compile().document)


def test_neutral_coordinates_cover_native_inventory(
    pin_model: FullBodyPinocchioModel,
) -> None:
    neutral = pin_model.neutral_coordinates()
    assert set(neutral) == set(pin_model.coordinate_order)
    assert all(value == 0.0 for value in neutral.values())
    assert pin_model.frame_poses(neutral)


def test_configuration_still_rejects_partial_inventory(
    pin_model: FullBodyPinocchioModel,
) -> None:
    with pytest.raises(ValueError, match="exactly the native coordinate inventory"):
        pin_model.frame_poses({})


def test_cross_engine_parity_passes_with_real_pinocchio() -> None:
    report = CharacterBuilderModel().verify_cross_engine_parity()
    assert report["passed"] is True
