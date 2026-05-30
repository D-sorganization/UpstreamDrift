"""Regression test for the single gravity source of truth (issue #6638 F5).

Previously ``constants.GRAVITY = 9.81`` drifted ~0.05 m/s^2 from the canonical
``GRAVITY_M_S2`` (~9.80665), so cross-engine results diverged depending on which
symbol a call site happened to import. All gravity magnitudes must now resolve
to the single canonical value.
"""

from __future__ import annotations

import pytest


def test_constants_gravity_matches_canonical() -> None:
    from src.shared.python.core.constants import (
        GRAVITY,
        GRAVITY_FLOAT,
        GRAVITY_M_S2,
    )

    assert pytest.approx(float(GRAVITY_M_S2)) == GRAVITY
    assert pytest.approx(GRAVITY_FLOAT) == GRAVITY
    # Must be NIST standard gravity, not the 9.81 approximation.
    assert pytest.approx(9.80665, abs=1e-5) == GRAVITY
    assert pytest.approx(9.81, abs=1e-6) != GRAVITY


def test_common_physics_gravity_is_consistent() -> None:
    from src.engines.common import physics
    from src.shared.python.core.constants import GRAVITY

    assert pytest.approx(GRAVITY) == physics.STANDARD_GRAVITY
    assert pytest.approx(GRAVITY) == physics.GRAVITY_APPROX
    assert physics.GRAVITY_VECTOR[2] == pytest.approx(-GRAVITY)
