"""Actuator resolution in the Drake autodiff fitter (issue #13421).

``fit_swing_autodiff`` used to fall back to ``num_velocities() - 6`` actuators
when the plant declared none, then fix a 19-wide torque onto a zero-width
actuation port. The fitter must use the plant's real actuators and refuse a
plant that has none.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("pydrake.multibody.plant")

from src.engines.physics_engines.drake.python.motion_matching.fit_swing_autodiff import (  # noqa: E402
    _build_autodiff_plant,
)
from src.engines.physics_engines.drake.python.motion_matching.humanoid_urdf import (  # noqa: E402
    CANONICAL_URDF,
    EXPECTED_NUM_REVOLUTE_DOF,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_drake]


def _urdf_without_transmissions(tmp_path: Path) -> Path:
    text = CANONICAL_URDF.read_text(encoding="utf-8")
    stripped = re.sub(
        r"<transmission\b.*?</transmission>\s*", "", text, flags=re.DOTALL
    )
    assert "<transmission" not in stripped
    out = tmp_path / "golfer_unactuated.urdf"
    out.write_text(stripped, encoding="utf-8")
    return out


def test_autodiff_plant_uses_the_urdf_actuators() -> None:
    """The canonical golfer resolves one actuator per revolute joint."""
    plant_ad, plant_float, _n_q, n_v, n_actuators = _build_autodiff_plant(None, 1e-3)
    assert n_actuators == EXPECTED_NUM_REVOLUTE_DOF == n_v - 6
    assert int(plant_float.num_actuators()) == n_actuators
    assert int(plant_ad.num_actuators()) == n_actuators


def test_autodiff_plant_rejects_a_model_without_actuators(tmp_path: Path) -> None:
    """No actuators means theta cannot be applied, so fail closed."""
    with pytest.raises(ValueError, match="declares no joint actuators"):
        _build_autodiff_plant(_urdf_without_transmissions(tmp_path), 1e-3)
