"""Right-hand grip wrench for the Drake URDF humanoid (GCV-8, #11714).

The URDF welds the club to the right hand only, so the right-hand wrench on the
club is the club Newton-Euler load and the left hand is unavailable with a
reason (never zero).  Sign: wrench exerted by the hand ON THE CLUB, world frame.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pydrake")
pytest.importorskip("pydrake.multibody.plant")

from pydrake.multibody.plant import MultibodyPlant  # noqa: E402

from src.engines.physics_engines.drake.python.motion_matching.humanoid_urdf import (  # noqa: E402,E501
    build_humanoid_urdf,
    load_humanoid_into_plant,
    right_hand_grip_analysis,
)
from src.shared.python.biomechanics.grip_wrench import (  # noqa: E402
    to_overlay_wrenches,
)
from src.shared.python.force_overlay.contracts import WrenchKind  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_drake]

G = 9.81  # Drake default gravity magnitude


@pytest.fixture(scope="module")
def plant_and_club_mass(tmp_path_factory):
    urdf = build_humanoid_urdf(out_path=tmp_path_factory.mktemp("urdf") / "g.urdf")
    plant = MultibodyPlant(time_step=0.0)
    load_humanoid_into_plant(plant, urdf)
    plant.Finalize()
    club = plant.GetBodyByName("club_shaft")
    return plant, float(club.default_mass())


def test_static_right_hand_supports_club_weight(plant_and_club_mass) -> None:
    plant, mass = plant_and_club_mass
    ctx = plant.CreateDefaultContext()
    analysis = right_hand_grip_analysis(plant, ctx, np.zeros(plant.num_velocities()))
    assert analysis.right is not None
    np.testing.assert_allclose(
        analysis.right.force_on_club_n, [0.0, 0.0, mass * G], atol=1e-9
    )


def test_upward_acceleration_raises_force_on_club(plant_and_club_mass) -> None:
    plant, mass = plant_and_club_mass
    ctx = plant.CreateDefaultContext()
    vdot = np.zeros(plant.num_velocities())
    # Floating base: velocity order is [omega; v]; identity orientation.
    vdot[5] = 2.0
    analysis = right_hand_grip_analysis(plant, ctx, vdot)
    assert analysis.right is not None
    np.testing.assert_allclose(
        analysis.right.force_on_club_n, [0.0, 0.0, mass * (G + 2.0)], atol=1e-9
    )


def test_left_hand_is_unavailable_with_reason_not_zero(plant_and_club_mass) -> None:
    plant, _ = plant_and_club_mass
    ctx = plant.CreateDefaultContext()
    analysis = right_hand_grip_analysis(plant, ctx, np.zeros(plant.num_velocities()))
    assert analysis.left is None
    assert "left" in analysis.unavailable_reason.lower()
    assert "right hand only" in analysis.unavailable_reason.lower()
    assert analysis.net_force_n is None  # no net without both hands


def test_overlay_emits_only_the_right_hand_grip_frame(plant_and_club_mass) -> None:
    plant, _ = plant_and_club_mass
    ctx = plant.CreateDefaultContext()
    analysis = right_hand_grip_analysis(plant, ctx, np.zeros(plant.num_velocities()))
    wrenches = to_overlay_wrenches(analysis, source="drake:urdf")
    assert [w.label for w in wrenches] == ["grip:hand_right"]
    assert wrenches[0].kind == WrenchKind.GRIP


def test_rejects_bad_acceleration_shape_and_unknown_club(plant_and_club_mass) -> None:
    plant, _ = plant_and_club_mass
    ctx = plant.CreateDefaultContext()
    with pytest.raises(ValueError, match="vdot"):
        right_hand_grip_analysis(plant, ctx, np.zeros(3))
    with pytest.raises(ValueError, match="club"):
        right_hand_grip_analysis(
            plant, ctx, np.zeros(plant.num_velocities()), club_body="nope"
        )
