"""Regression tests for MJCF generators in src/engines/physics_engines/mujoco.

Covers issue #4106: ``PhysicalConstant`` instances were interpolated directly
into the MJCF f-strings, producing ``gravity="0 0 -PhysicalConstant(...)"``
which ``mujoco.MjModel.from_xml_string`` rejects with
``XML Error: bad format in attribute 'gravity'``.

These tests load the three generated XML strings through MuJoCo and assert
that the gravity vector parses to the expected -9.80665 m/s^2 magnitude.

The package directory is named ``mujoco_mjcf`` rather than ``mujoco`` to
avoid an import-time name collision with the upstream ``mujoco`` package.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

# Imports happen after ``importorskip`` so the suite still runs in environments
# without MuJoCo (the engine modules import ``mujoco`` transitively).
from src.engines.physics_engines.mujoco._golf_swing_advanced_xml import (  # noqa: E402
    ADVANCED_BIOMECHANICAL_GOLF_SWING_XML,
)
from src.engines.physics_engines.mujoco._golf_swing_full_body_xml import (  # noqa: E402
    FULL_BODY_GOLF_SWING_XML,
)
from src.engines.physics_engines.mujoco._golf_swing_upper_body_xml import (  # noqa: E402
    UPPER_BODY_GOLF_SWING_XML,
)

_GRAVITY_TOLERANCE = 1e-6
_EXPECTED_GRAVITY_Z = -9.80665  # NIST CODATA standard gravity, sign-flipped.

_VARIANTS = [
    ("advanced", ADVANCED_BIOMECHANICAL_GOLF_SWING_XML),
    ("full_body", FULL_BODY_GOLF_SWING_XML),
    ("upper_body", UPPER_BODY_GOLF_SWING_XML),
]


@pytest.mark.parametrize(("name", "xml"), _VARIANTS)
def test_all_three_variants_compile(name: str, xml: str) -> None:
    """Each MJCF variant compiles via ``MjModel.from_xml_string`` without error.

    Pre-fix this raised ``ValueError: XML Error: bad format in attribute
    'gravity'`` because ``PhysicalConstant.__repr__`` rendered as
    ``PhysicalConstant(9.807, unit='m/s^2')`` inside the ``gravity=""``
    attribute.
    """
    # The PhysicalConstant repr leaked into the XML when the bug was present;
    # surface that explicitly so a future regression is unmistakable.
    assert "PhysicalConstant(" not in xml, (
        f"{name} XML still contains PhysicalConstant repr - interpolation "
        "regression. Cast PhysicalConstant -> float before f-string substitution."
    )

    model = mujoco.MjModel.from_xml_string(xml)
    assert model is not None
    # MuJoCo stores gravity as (gx, gy, gz). Z component must be -9.80665.
    assert model.opt.gravity[0] == pytest.approx(0.0, abs=_GRAVITY_TOLERANCE)
    assert model.opt.gravity[1] == pytest.approx(0.0, abs=_GRAVITY_TOLERANCE)
    assert model.opt.gravity[2] == pytest.approx(
        _EXPECTED_GRAVITY_Z, abs=_GRAVITY_TOLERANCE
    )


def test_gravity_is_approximately_minus_9_807() -> None:
    """Loose check that every variant's gravity is close to -9.807 m/s^2."""
    for name, xml in _VARIANTS:
        model = mujoco.MjModel.from_xml_string(xml)
        assert model.opt.gravity[2] == pytest.approx(
            -9.807, abs=1e-3
        ), f"{name}: unexpected gravity {model.opt.gravity[2]}"
