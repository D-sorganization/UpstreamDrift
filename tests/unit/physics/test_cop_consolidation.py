"""One centre-of-pressure implementation (GCV-6, #11712; epic #11706).

Every legacy entry point returns the canonical result where the formulas agree,
warns that it is deprecated, and reports an unloaded foot as ``None`` rather
than zero.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock
import warnings

import numpy as np
import pytest

from src.shared.python.biomechanics.ground_reaction import center_of_pressure
from src.shared.python.motion_matching.force_torque import (
    SpatialWrench,
    compute_center_of_pressure as wrench_cop,
)
from src.shared.python.physics._contact_types import ContactPoint, ContactState
from src.shared.python.physics._grip_forces import (
    compute_center_of_pressure as grip_cop_legacy,
    compute_grip_pressure_centre,
)
from src.shared.python.physics.ground_reaction_forces import (
    compute_cop_from_grf,
    extract_grf_from_contacts,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

REPO = Path(__file__).resolve().parents[3]

FORCE_MOMENT_FIXTURES = [
    ([0.0, 0.0, 1000.0], [100.0, -200.0, 0.0]),
    ([30.0, -20.0, 640.0], [12.0, 45.0, 3.0]),
    ([-5.0, 8.0, 80.0], [-3.0, 2.0, 0.5]),
]


@pytest.mark.parametrize(("force", "moment"), FORCE_MOMENT_FIXTURES)
def test_legacy_ground_cop_matches_canonical_and_warns(force, moment) -> None:
    expected = center_of_pressure(force, moment)
    with pytest.warns(DeprecationWarning, match="center_of_pressure"):
        legacy = compute_cop_from_grf(np.array(force), np.array(moment))
    np.testing.assert_allclose(legacy, expected, atol=1e-12)


def test_legacy_ground_cop_honours_ground_height() -> None:
    force, moment = [20.0, -10.0, 500.0], [40.0, -60.0, 1.0]
    expected = center_of_pressure(force, moment, ground_height_m=0.05)
    with pytest.warns(DeprecationWarning):
        legacy = compute_cop_from_grf(np.array(force), np.array(moment), 0.05)
    np.testing.assert_allclose(legacy, expected, atol=1e-12)


def test_legacy_ground_cop_unloaded_is_none_not_zero() -> None:
    with pytest.warns(DeprecationWarning):
        assert compute_cop_from_grf(np.array([0, 0, 5.0]), np.array([100, 100, 0])) is None


@pytest.mark.parametrize(("force", "moment"), FORCE_MOMENT_FIXTURES)
def test_wrench_cop_matches_canonical_on_the_ground_plane_and_warns(
    force, moment
) -> None:
    wrench = SpatialWrench("ground", (0.0, 0.0, 0.0), tuple(force), tuple(moment))
    expected = center_of_pressure(force, moment, min_fz=5.0)
    with pytest.warns(DeprecationWarning, match="center_of_pressure"):
        legacy = wrench_cop(wrench, f_threshold_n=5.0)
    assert legacy is not None and expected is not None
    np.testing.assert_allclose(legacy, expected[:2], atol=1e-12)


def test_wrench_cop_with_offset_point_equals_canonical_of_the_moved_moment() -> None:
    p = np.array([0.2, -0.1, 0.0])
    force, torque = np.array([10.0, 5.0, 500.0]), np.array([25.0, -50.0, 2.0])
    wrench = SpatialWrench("ground", tuple(p), tuple(force), tuple(torque))
    expected = center_of_pressure(force, torque + np.cross(p, force))
    with pytest.warns(DeprecationWarning):
        legacy = wrench_cop(wrench)
    np.testing.assert_allclose(legacy, expected[:2], atol=1e-12)


def test_wrench_cop_below_threshold_is_none() -> None:
    wrench = SpatialWrench("g", (0, 0, 0), (0, 0, 2.5), (1, 1, 0))
    with pytest.warns(DeprecationWarning):
        assert wrench_cop(wrench) is None


def _contact(pos, fz):
    return ContactPoint(
        position=np.array(pos, float),
        normal=np.array([0.0, 0.0, 1.0]),
        normal_force=fz,
        tangent_force=np.zeros(3),
        slip_velocity=np.zeros(3),
        state=ContactState.STICKING,
    )


def test_grip_pressure_centre_is_the_normal_force_weighted_centroid() -> None:
    contacts = [_contact((0, 0, 0), 100.0), _contact((3, 0, 0), 50.0)]
    np.testing.assert_allclose(compute_grip_pressure_centre(contacts), [1.0, 0.0, 0.0])


def test_grip_pressure_centre_unloaded_is_none_not_zero() -> None:
    assert compute_grip_pressure_centre([]) is None
    assert compute_grip_pressure_centre([_contact((1, 1, 1), 0.0)]) is None


def test_legacy_grip_cop_name_warns_and_forwards() -> None:
    contacts = [_contact((1.0, 2.0, 0.0), 100.0)]
    with pytest.warns(DeprecationWarning, match="compute_grip_pressure_centre"):
        legacy = grip_cop_legacy(contacts)
    np.testing.assert_allclose(legacy, [1.0, 2.0, 0.0])


def _engine(contact, gravity=None):
    engine = MagicMock()
    engine.get_time.return_value = 0.0
    engine.compute_contact_forces.return_value = np.asarray(contact, float)
    engine.compute_gravity_forces.return_value = np.asarray(
        [-800.0] if gravity is None else gravity, float
    )
    engine.compute_jacobian.return_value = {"linear": np.zeros((3, 2))}
    return engine


def test_gravity_fallback_is_marked_estimated_with_unavailable_cop() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        grf = extract_grf_from_contacts(_engine(np.zeros(3)), ["foot"])
    assert grf.estimated is True
    assert grf.cop is None


def test_engine_contact_path_is_not_estimated_and_has_no_fabricated_cop() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        grf = extract_grf_from_contacts(_engine([0.0, 0.0, 800.0]), ["foot"])
    assert grf.estimated is False
    # the engine reports a force only: no point, so no CoP and no moment
    assert grf.cop is None
    assert np.isnan(grf.moment).all()


def test_cop_implementations_live_only_in_the_canonical_module() -> None:
    out = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            "-E",
            "def (compute_center_of_pressure|compute_cop)",
            "--",
            "src",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    shims = {
        "src/shared/python/motion_matching/force_torque.py",
        "src/shared/python/physics/_grip_forces.py",
        "src/shared/python/physics/ground_reaction_forces.py",
    }
    assert out, "expected the deprecation shims"
    for line in out:
        assert line.split(":", 1)[0] in shims, line
