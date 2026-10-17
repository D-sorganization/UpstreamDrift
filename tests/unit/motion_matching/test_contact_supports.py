"""Raised contact supports (a bench top) in the shared contact law (#11745).

A spec's ``contact.supports`` names spheres that press on a plane parallel to
the ground at the support's own height instead of on the ground.  Every
same-input engine resolves the plane per sphere with
:func:`~src.shared.python.motion_matching.contact_law.support_planes`, so the
law stays shared and a bench-supported spec replays identically.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.support.real_engines import real_module

from src.shared.python.motion_matching.contact_law import (
    GroundPlane,
    support_planes,
)

pytestmark = pytest.mark.unit

_GROUND = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)
_SPHERES = [
    {"name": "heel_r", "body": "calcn_r", "position_m": [0, 0, 0], "radius_m": 0.03},
    {"name": "back_a", "body": "pelvis", "position_m": [0, 0, 0], "radius_m": 0.04},
    {"name": "back_b", "body": "torso", "position_m": [0, 0, 0], "radius_m": 0.04},
]


def _contact(supports: list[dict]) -> dict:
    return {"spheres": _SPHERES, "supports": supports}


def test_spec_without_supports_presses_every_sphere_on_the_ground() -> None:
    assert support_planes({"spheres": _SPHERES}, _GROUND) == {}
    assert support_planes(_contact([]), _GROUND) == {}


def test_support_is_parallel_to_the_ground_at_its_own_height() -> None:
    tilted = GroundPlane(normal=(0.0, 0.6, 0.8), height_m=0.1)
    planes = support_planes(
        _contact([{"name": "bench", "height_m": 0.45, "spheres": ["back_a"]}]),
        tilted,
    )
    assert set(planes) == {"back_a"}
    assert planes["back_a"].normal == tilted.normal
    assert planes["back_a"].height_m == 0.45


@pytest.mark.parametrize(
    ("supports", "match"),
    [
        ([{"name": "b", "height_m": 0.4, "spheres": ["nope"]}], "unknown"),
        ([{"name": "b", "height_m": float("nan"), "spheres": ["back_a"]}], "finite"),
        ([{"name": "", "height_m": 0.4, "spheres": ["back_a"]}], "name"),
        ([{"name": "b", "height_m": 0.4, "spheres": []}], "at least one"),
        (
            [
                {"name": "b", "height_m": 0.4, "spheres": ["back_a"]},
                {"name": "c", "height_m": 0.5, "spheres": ["back_a"]},
            ],
            "more than one",
        ),
    ],
)
def test_malformed_supports_are_rejected(supports: list[dict], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        support_planes(_contact(supports), _GROUND)


def test_supports_must_be_a_list() -> None:
    with pytest.raises(TypeError, match="list"):
        support_planes({"spheres": _SPHERES, "supports": {"b": 0.4}}, _GROUND)


# -- every same-input engine applies the support ----------------------------

_ENGINES = [
    pytest.param("mujoco", "mujoco", marks=pytest.mark.requires_mujoco),
    pytest.param("drake", "pydrake", marks=pytest.mark.requires_drake),
    pytest.param("pinocchio", "pinocchio", marks=pytest.mark.requires_pinocchio),
    pytest.param("opensim", "opensim", marks=pytest.mark.requires_opensim),
]
_DEPTH_M = 1e-3


def _raised_heel_spec() -> tuple[bytes, bytes, np.ndarray]:
    """The lift body, and a copy whose right heel presses on a raised plate.

    The plate sits ``_DEPTH_M`` below the lowest point of the heel sphere at
    the seed pose, so only a support-aware engine sees that penetration.
    """
    real_module("mujoco")
    from src.shared.python.lifting.bundles import LIFT_SPEC, load_seed_pose
    from src.shared.python.motion_matching.same_input import VectorPlant

    base = LIFT_SPEC.read_bytes()
    q = load_seed_pose()
    sample = VectorPlant("mujoco", base).contact_samples(q, np.zeros(q.size))
    lowest = float(sample["heel_r"].contact_point_m[2])
    document = json.loads(base)
    document["contact"]["supports"] = [
        {"name": "plate", "height_m": lowest + _DEPTH_M, "spheres": ["heel_r"]}
    ]
    return base, json.dumps(document).encode(), q


@pytest.mark.integration
@pytest.mark.parametrize(("engine", "module"), _ENGINES)
def test_engine_presses_a_supported_sphere_on_its_support(
    engine: str, module: str
) -> None:
    real_module(module)
    from src.shared.python.motion_matching.same_input import VectorPlant

    base, raised, q = _raised_heel_spec()
    v = np.zeros(q.size)
    on_ground = VectorPlant(engine, base).contact_samples(q, v)
    on_plate = VectorPlant(engine, raised).contact_samples(q, v)
    assert on_ground["heel_r"].penetration_m == 0.0
    assert on_plate["heel_r"].penetration_m == pytest.approx(_DEPTH_M, abs=1e-12)
    assert float(on_plate["heel_r"].normal_force_n[2]) > 0.0
    for name, sample in on_ground.items():
        if name != "heel_r":
            np.testing.assert_array_equal(
                on_plate[name].normal_force_n, sample.normal_force_n
            )
