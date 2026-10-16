"""Native segment-orientation solve between Euler parametrisations (SPN-9c, #12758).

The golf humanoid shoulder is a ZXY sequence and the Bruno-style shoulder an
XYZ-like sequence, so the coordinates cannot be copied.  Two synthetic models
(invented geometry, invented tilt) stand in for them; nothing here reads a
licensed file.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.opensim.python.spine.fullbody_map import (
    OrientationGroup,
)
from src.engines.physics_engines.opensim.python.spine.fullbody_native import (
    body_rotations,
    coordinate_names,
    neutral_alignment,
    solve_orientation_group,
)
from tests.helpers.opensim_support import require_opensim

pytestmark = [pytest.mark.requires_opensim, pytest.mark.integration]

ZXY = (("z_flex", (0, 0, 1)), ("x_add", (1, 0, 0)), ("y_rot", (0, 1, 0)))
XYZ = (("elv", (1, 0, 0)), ("rot", (0, 1, 0)), ("ang", (0, 0, 1)))


def _model(axes, tilt_deg: float = 0.0):
    osim = require_opensim()
    model = osim.Model()
    thorax = osim.Body("thorax", 1.0, osim.Vec3(0), osim.Inertia(0.1, 0.1, 0.1))
    humerus = osim.Body("humerus", 1.0, osim.Vec3(0), osim.Inertia(0.1, 0.1, 0.1))
    model.addBody(thorax)
    model.addBody(humerus)
    tilt = osim.Vec3(0.0, 0.0, np.radians(tilt_deg))
    weld = osim.WeldJoint(
        "ground_thorax",
        model.getGround(),
        osim.Vec3(0),
        osim.Vec3(0),
        thorax,
        osim.Vec3(0),
        tilt,
    )
    model.addJoint(weld)
    transform = osim.SpatialTransform()
    for index, (name, axis) in enumerate(axes):
        item = transform.updTransformAxis(index)
        item.setCoordinateNames(osim.ArrayStr(name, 1))
        item.setAxis(osim.Vec3(*axis))
        item.setFunction(osim.LinearFunction())
    for index in range(3, 6):
        transform.updTransformAxis(index).setFunction(osim.Constant(0.0))
    joint = osim.CustomJoint("shoulder", thorax, humerus, transform)
    model.addJoint(joint)
    model.finalizeConnections()
    return model


def _group() -> OrientationGroup:
    return OrientationGroup(
        "shoulder_r",
        "humerus",
        tuple(n for n, _ in ZXY),
        "humerus",
        tuple(n for n, _ in XYZ),
    )


def _targets(n: int = 9) -> dict[str, np.ndarray]:
    t = np.linspace(0.0, 1.0, n)
    return {"z_flex": 0.9 * t, "x_add": -0.4 * np.sin(2 * t), "y_rot": 0.5 * t - 0.2}


def test_adapters_read_names_and_rotations() -> None:
    model = _model(XYZ)
    assert set(coordinate_names(model)) == {"elv", "rot", "ang"}
    rot = body_rotations(
        model,
        {"elv": np.array([0.0, 0.3]), "rot": np.zeros(2), "ang": np.zeros(2)},
        "humerus",
    )
    assert rot.shape == (2, 3, 3)
    np.testing.assert_allclose(rot[0], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        rot[1], Rotation.from_rotvec([0.3, 0, 0]).as_matrix(), atol=1e-12
    )
    with pytest.raises(ValueError, match="no body"):
        body_rotations(model, {}, "femur")
    with pytest.raises(ValueError, match="no coordinate"):
        body_rotations(model, {"bogus": np.zeros(2)}, "humerus")


def test_neutral_alignment_is_identity_for_identical_zero_poses() -> None:
    k = neutral_alignment(_model(ZXY), "humerus", _model(XYZ), "humerus")
    np.testing.assert_allclose(k, np.eye(3), atol=1e-12)
    tilted = neutral_alignment(_model(ZXY), "thorax", _model(ZXY, 30.0), "thorax")
    np.testing.assert_allclose(
        Rotation.from_matrix(tilted).magnitude(), np.radians(30.0), atol=1e-9
    )


def test_solve_reproduces_the_golf_style_orientation_with_another_sequence() -> None:
    golf = _model(ZXY)
    target_model = _model(XYZ, tilt_deg=25.0)
    golf_values = _targets()
    n = golf_values["z_flex"].size
    thorax_world = body_rotations(
        target_model,
        {"elv": np.zeros(n), "rot": np.zeros(n), "ang": np.zeros(n)},
        "thorax",
    )
    humerus_in_thorax = body_rotations(golf, golf_values, "humerus")
    target_world = thorax_world @ humerus_in_thorax
    solution = solve_orientation_group(target_model, _group(), {}, target_world)
    assert solution.residual_rad.max() < 1e-7
    achieved = body_rotations(target_model, dict(solution.coordinates), "humerus")
    np.testing.assert_allclose(achieved, target_world, atol=1e-6)
    # a genuine conversion: the copied ZXY angles would not reproduce it
    copied = {
        "elv": golf_values["z_flex"],
        "rot": golf_values["x_add"],
        "ang": golf_values["y_rot"],
    }
    copied_rot = body_rotations(target_model, copied, "humerus")
    assert np.abs(copied_rot - target_world).max() > 1e-2


def test_unreachable_targets_are_reported_not_hidden() -> None:
    target_model = _model(XYZ)
    n = 3
    far = Rotation.from_rotvec(np.tile([0.0, 0.0, 2.0], (n, 1))).as_matrix()
    limits = dict.fromkeys(("elv", "rot", "ang"), (-0.5, 0.5))
    solution = solve_orientation_group(target_model, _group(), {}, far, bounds=limits)
    assert solution.residual_rad.min() > 1.0
    for name in limits:
        assert np.abs(solution.coordinates[name]).max() <= 0.5 + 1e-9


def test_solve_preconditions() -> None:
    model = _model(XYZ)
    with pytest.raises(ValueError, match="shape"):
        solve_orientation_group(model, _group(), {}, np.eye(3))
    with pytest.raises(ValueError, match="proper rotation"):
        solve_orientation_group(model, _group(), {}, np.tile(2 * np.eye(3), (2, 1, 1)))
    with pytest.raises(ValueError, match="no coordinate"):
        solve_orientation_group(
            model,
            OrientationGroup("g", "humerus", ZXY_NAMES, "humerus", ("a", "b", "c")),
            {},
            np.tile(np.eye(3), (2, 1, 1)),
        )
    with pytest.raises(ValueError, match="tolerance_rad"):
        solve_orientation_group(
            model, _group(), {}, np.tile(np.eye(3), (2, 1, 1)), tolerance_rad=0.0
        )


ZXY_NAMES = tuple(n for n, _ in ZXY)
