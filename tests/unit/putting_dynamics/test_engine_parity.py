"""Engine-to-shared parity for the putting migration (epic #14141, step 5b).

The legacy engine (``src/engines/physics_engines/putting_green``) is the
migration source until step 10 deletes it. These tests pin what the
shared core reproduces exactly, and list what it deliberately does not
port. They compare two implementations, so they guard the migration;
they are not external validation (that is the fixture file's job).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.engines.physics_engines.putting_green.python import (
    turf_properties as legacy_turf,
)
from src.engines.physics_engines.putting_green.python._surface_data import (
    SlopeRegion,
)
from src.engines.physics_engines.putting_green.python.green_surface import (
    GreenSurface,
)
from src.shared.python.core.physics_constants import GRAVITY_M_S2
from src.shared.python.putting_dynamics import (
    GrassType,
    HeightField,
    TurfCondition,
    TurfSpec,
    grain_factor,
)

pytestmark = pytest.mark.unit

_G = float(GRAVITY_M_S2)


def test_engine_turf_tables_are_the_shared_ones() -> None:
    """Step 4b made the engine import the shared enums (one table)."""
    assert legacy_turf.GrassType is GrassType
    assert legacy_turf.TurfCondition is TurfCondition


@pytest.mark.parametrize("grain_deg", [0.0, 37.0, 135.0, -90.0])
@pytest.mark.parametrize("travel_deg", [0.0, 60.0, 180.0, 250.0])
def test_grain_multiplier_matches_the_engine(
    grain_deg: float, travel_deg: float
) -> None:
    grain = math.radians(grain_deg)
    travel = math.radians(travel_deg)
    legacy = legacy_turf.TurfProperties(
        grass_type=legacy_turf.GrassType.BERMUDA,
        grain_direction=np.array([math.cos(grain), math.sin(grain)]),
    )
    shared = TurfSpec(
        stimp_ft=10.0, grass_type=GrassType.BERMUDA, grain_direction_rad=grain
    )
    direction = np.array([math.cos(travel), math.sin(travel)])
    legacy_factor = 1.0 + legacy.compute_grain_effect(direction)
    shared_factor = grain_factor(shared.to_friction_params(), travel)
    assert shared_factor == pytest.approx(legacy_factor, abs=1e-12)


@pytest.mark.parametrize("aspect_deg", [0.0, 90.0, 210.0])
def test_heightmap_slope_acceleration_matches_the_engine(aspect_deg: float) -> None:
    """Both cores accelerate a ball by ``-g * grad(h)`` on a heightmap."""
    grade = 0.02
    aspect = math.radians(aspect_deg)
    xs = np.linspace(0.0, 20.0, 81)
    xx, yy = np.meshgrid(xs, xs)
    heights = -grade * (xx * math.cos(aspect) + yy * math.sin(aspect))
    legacy = GreenSurface(20.0, 20.0)
    legacy.set_heightmap(heights, smooth=False)
    legacy_accel = legacy.get_gravitational_acceleration(np.array([10.0, 10.0]))

    shared = HeightField.planar(grade * 100.0, aspect_deg, extent_m=30.0)
    gx, gy = shared.gradient(0.0, 0.0)
    assert (-_G * gx, -_G * gy) == pytest.approx(tuple(legacy_accel), abs=1e-6)


def test_slope_region_sign_is_a_documented_non_port() -> None:
    """The engine's ``SlopeRegion`` pushes the ball uphill.

    Its docstring says ``slope_direction`` points downhill, but
    ``get_gravitational_acceleration`` returns ``-g * slope``, so the
    ball accelerates against it. The shared core takes slope only from
    a height field's gradient, so the defect is not ported.
    """
    legacy = GreenSurface(20.0, 20.0)
    legacy.add_slope_region(
        SlopeRegion(
            center=np.array([10.0, 10.0]),
            radius=50.0,
            slope_direction=np.array([1.0, 0.0]),
            slope_magnitude=0.02,
            falloff=1.0,
        )
    )
    ax, _ = legacy.get_gravitational_acceleration(np.array([10.0, 10.0]))
    assert ax < 0.0  # documented downhill is +x; the engine accelerates to -x
    downhill_x = HeightField.planar(2.0, 0.0, extent_m=30.0)
    assert -_G * downhill_x.gradient(0.0, 0.0)[0] > 0.0


#: Engine behaviours the shared core deliberately does not reproduce.
PERMANENT_NON_PORTS = {
    "legacy roll law mu = 0.196 / stimp": "ADR-0045 ud-legacy-roll/1; the "
    "shared core uses usga-stimp-roll/1 only",
    "height-of-cut friction multiplier": "fitted to the legacy law; no source",
    "compaction speed factor": "uncited; TurfSpec keeps compaction as metadata",
    "cross-grain lateral curve (apply_grain_to_velocity)": "uncited velocity "
    "rotation; the shared core models grain as anisotropic friction only",
    "gear-effect sidespin (putter_stroke.py factor 200)": "unsourced magic "
    "number; collision.strike models the off-centre speed loss only",
    "SlopeRegion acceleration sign": "pushes uphill; see the test above",
}


def test_permanent_non_ports_are_absent_from_the_shared_core() -> None:
    import src.shared.python.putting_dynamics as shared

    for name in ("apply_grain_to_velocity", "SlopeRegion", "compaction_speed"):
        assert not hasattr(shared, name)
    assert all(reason.strip() for reason in PERMANENT_NON_PORTS.values())
