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


_BODY = (
    '<Body name="{n}"><mass>1</mass><mass_center>0 0 0</mass_center>'
    "<inertia>0.1 0.1 0.1 0 0 0</inertia></Body>"
)
_FRAME = (
    '<PhysicalOffsetFrame name="{n}"><socket_parent>{p}</socket_parent>'
    "<translation>0 0 0</translation><orientation>{o}</orientation>"
    "</PhysicalOffsetFrame>"
)
_COORD = (
    '<Coordinate name="{n}"><default_value>0</default_value>'
    "<default_speed_value>0</default_speed_value><range>-6 6</range>"
    "<clamped>false</clamped><locked>false</locked></Coordinate>"
)
_AXIS = (
    '<TransformAxis name="rotation{i}"><coordinates>{n}</coordinates>'
    "<axis>{a}</axis><LinearFunction><coefficients> 1 0</coefficients>"
    "</LinearFunction></TransformAxis>"
)
_FIXED = (
    '<TransformAxis name="translation{i}"><coordinates /><axis>{a}</axis>'
    "<Constant><value>0</value></Constant></TransformAxis>"
)


def _xml(axes, tilt_deg: float) -> str:
    """OpenSim XML: thorax welded to ground (z tilt), humerus on a CustomJoint."""
    coords = "".join(_COORD.format(n=n) for n, _ in axes)
    rot = "".join(
        _AXIS.format(i=i + 1, n=n, a=" ".join(map(str, a)))
        for i, (n, a) in enumerate(axes)
    )
    fixed = "".join(
        _FIXED.format(i=i + 1, a=a) for i, a in enumerate(("1 0 0", "0 1 0", "0 0 1"))
    )
    weld_frames = _FRAME.format(
        n="g_off", p="/ground", o=f"0 0 {float(np.radians(tilt_deg))!r}"
    ) + _FRAME.format(n="t_off", p="/bodyset/thorax", o="0 0 0")
    joint_frames = _FRAME.format(
        n="p_off", p="/bodyset/thorax", o="0 0 0"
    ) + _FRAME.format(n="c_off", p="/bodyset/humerus", o="0 0 0")
    return (
        '<OpenSimDocument Version="40000"><Model name="t">'
        "<gravity>0 -9.81 0</gravity><BodySet><objects>"
        + _BODY.format(n="thorax")
        + _BODY.format(n="humerus")
        + "</objects></BodySet><JointSet><objects>"
        '<WeldJoint name="ground_thorax"><socket_parent_frame>g_off'
        "</socket_parent_frame><socket_child_frame>t_off</socket_child_frame>"
        f"<frames>{weld_frames}</frames></WeldJoint>"
        '<CustomJoint name="shoulder"><socket_parent_frame>p_off'
        "</socket_parent_frame><socket_child_frame>c_off</socket_child_frame>"
        f"<coordinates>{coords}</coordinates><frames>{joint_frames}</frames>"
        f"<SpatialTransform>{rot}{fixed}</SpatialTransform></CustomJoint>"
        "</objects></JointSet></Model></OpenSimDocument>"
    )


@pytest.fixture
def make(tmp_path):
    """Factory ``make(axes, tilt_deg)`` returning a loaded synthetic model."""
    osim = require_opensim()
    counter = iter(range(1000))

    def build(axes, tilt_deg: float = 0.0):
        path = tmp_path / f"shoulder_{next(counter)}.osim"
        path.write_text(_xml(axes, tilt_deg), encoding="utf-8")
        return osim.Model(str(path))

    return build


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


def test_adapters_read_names_and_rotations(make) -> None:
    model = make(XYZ)
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


def test_neutral_alignment_is_identity_for_identical_zero_poses(make) -> None:
    k = neutral_alignment(make(ZXY), "humerus", make(XYZ), "humerus")
    np.testing.assert_allclose(k, np.eye(3), atol=1e-12)
    tilted = neutral_alignment(make(ZXY), "thorax", make(ZXY, 30.0), "thorax")
    np.testing.assert_allclose(
        Rotation.from_matrix(tilted).magnitude(), np.radians(30.0), atol=1e-9
    )


def test_solve_reproduces_the_golf_style_orientation_with_another_sequence(
    make,
) -> None:
    golf = make(ZXY)
    target_model = make(XYZ, tilt_deg=25.0)
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


def test_unreachable_targets_are_reported_not_hidden(make) -> None:
    target_model = make(XYZ)
    n = 3
    far = Rotation.from_rotvec(np.tile([0.0, 0.0, 2.0], (n, 1))).as_matrix()
    limits = dict.fromkeys(("elv", "rot", "ang"), (-0.5, 0.5))
    solution = solve_orientation_group(target_model, _group(), {}, far, bounds=limits)
    assert solution.residual_rad.min() > 1.0
    for name in limits:
        assert np.abs(solution.coordinates[name]).max() <= 0.5 + 1e-9


def test_solve_preconditions(make) -> None:
    model = make(XYZ)
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
