"""Contact equilibrium of the barbell hold's start pose (#11744, #13093).

The LIFT-4 hold used to start from ``preload_feet``: the IK standing pose
translated down until its lowest contact sphere (the toes) sank to the
equal-share depth ``W / (6 k)``.  At that state the feet carried 1051.8 N of
the 1768.3 N body plus bar, and their centre of pressure lay 6.7 cm ahead of
the combined centre of mass.  The body fell 2.1 degrees back onto its heels,
which rang a lightly damped fore-aft sway of the whole body (about 0.5 Hz)
whose CoM deceleration is the steady -102 N horizontal ground force and the
0.982 sag of the last 0.5 s.

The start pose is now the contact equilibrium of the held joint angles: only
the unactuated root moves, until the feet carry body plus bar with their
centre of pressure under the combined centre of mass.  Thresholds are fixed
before measuring:

* vertical force equals the weight within 1e-9 of it;
* contact moment about the centre of mass (tangential) within 1e-9 W m;
* the actuated coordinates are unchanged (exactly);
* the controller's model is at rest there: every acceleration at most
  1e-3 (m/s^2, rad/s^2).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.lifting.bundles import BACK_SQUAT_BUNDLE

pytestmark = pytest.mark.unit

pytest.importorskip("mujoco")

FORCE_FRACTION = 1e-9
MOMENT_FRACTION_M = 1e-9
REST_ACCELERATION = 1e-3


@pytest.fixture(scope="module")
def setting():  # type: ignore[no-untyped-def]
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )
    from src.shared.python.lifting.barbell_bushing import Barbell
    from src.shared.python.lifting.barbell_runs import (
        bushing_body_spec,
        rigid_bar_spec,
    )
    from src.shared.python.motion_matching import full_body_forward_dynamics as fs
    from src.shared.python.motion_matching.same_input import InputBundle

    source = InputBundle.load(BACK_SQUAT_BUNDLE)
    barbell = Barbell(float(source.provenance["optimizer"]["bar_mass_kg"]))
    body = bushing_body_spec(source.spec_bytes)
    simulator = fs.FullBodySimulator(
        NativeMujocoFullBodyModel(rigid_bar_spec(body, barbell))
    )
    q_hold = np.asarray(source.provenance["ik"]["q_standing"], dtype=float)
    return simulator, q_hold, body, barbell


def _contact(simulator, q):  # type: ignore[no-untyped-def]
    from src.shared.python.lifting.barbell_runs import contact_wrench_about_com

    return contact_wrench_about_com(simulator, q)


def test_preloaded_start_is_out_of_balance(setting) -> None:  # type: ignore[no-untyped-def]
    """The measured cause: the old start's CoP is 6.7 cm ahead of the CoM."""
    from src.shared.python.motion_matching import full_body_forward_dynamics as fs

    simulator, q_hold, _, _ = setting
    force, moment = _contact(simulator, fs.preload_feet(simulator, q_hold))
    weight = simulator.mass_kg * float(np.linalg.norm(simulator.gravity))
    assert force[2] / weight == pytest.approx(0.595, abs=0.005)
    # CoP ahead of the CoM: -M_y / F_z.
    assert -moment[1] / force[2] == pytest.approx(0.067, abs=0.002)


def test_contact_equilibrium_carries_the_weight_under_the_com(setting) -> None:  # type: ignore[no-untyped-def]
    from src.shared.python.lifting.barbell_runs import contact_equilibrium_pose

    simulator, q_hold, _, _ = setting
    q_eq = contact_equilibrium_pose(simulator, q_hold)
    force, moment = _contact(simulator, q_eq)
    weight = simulator.mass_kg * float(np.linalg.norm(simulator.gravity))
    assert abs(force[2] / weight - 1.0) <= FORCE_FRACTION
    assert np.abs(moment[:2]).max() <= MOMENT_FRACTION_M * weight
    actuated = simulator.actuated
    assert np.array_equal(q_eq[actuated], q_hold[actuated])
    assert q_eq[0] == q_hold[0] and q_eq[1] == q_hold[1]


def test_controller_model_is_at_rest_at_the_equilibrium(setting) -> None:  # type: ignore[no-untyped-def]
    from src.shared.python.lifting.barbell_runs import contact_equilibrium_pose
    from src.shared.python.motion_matching.pipeline.dynamics import (
        build_tracking_controller,
    )

    simulator, q_hold, _, _ = setting
    q_eq = contact_equilibrium_pose(simulator, q_hold)
    controller = build_tracking_controller(
        simulator, np.array([0.0, 1.0]), np.vstack([q_eq, q_eq])
    )
    rest = np.zeros_like(q_eq)
    tau = controller(0.5, q_eq, rest)
    affine, offset = simulator.affine_dynamics(q_eq, rest)
    acceleration = affine @ tau[simulator.actuated] + offset
    assert np.abs(acceleration).max() <= REST_ACCELERATION


def test_contact_equilibrium_rejects_bad_input(setting) -> None:  # type: ignore[no-untyped-def]
    from src.shared.python.lifting.barbell_runs import contact_equilibrium_pose

    simulator, q_hold, _, _ = setting
    with pytest.raises(ValueError, match="finite"):
        contact_equilibrium_pose(simulator, q_hold[:-1])
    bad = q_hold.copy()
    bad[7] = np.nan
    with pytest.raises(ValueError, match="finite"):
        contact_equilibrium_pose(simulator, bad)
    airborne = q_hold.copy()
    airborne[2] += 0.5
    with pytest.raises(ValueError, match="contact"):
        contact_equilibrium_pose(simulator, airborne, max_iterations=0)


def test_hold_starts_at_the_contact_equilibrium(setting) -> None:  # type: ignore[no-untyped-def]
    """The hold's start state is the equilibrium, with the bar hanging still."""
    from src.shared.python.lifting.barbell_runs import hold_start

    simulator, q_hold, body, barbell = setting
    start = hold_start(body, q_hold, barbell)
    force, moment = _contact(simulator, start.q_body)
    weight = simulator.mass_kg * float(np.linalg.norm(simulator.gravity))
    assert abs(force[2] / weight - 1.0) <= FORCE_FRACTION
    assert np.abs(moment[:2]).max() <= MOMENT_FRACTION_M * weight
    assert np.array_equal(start.v, np.zeros_like(start.v))
    assert np.array_equal(start.track[0], start.q_body)
