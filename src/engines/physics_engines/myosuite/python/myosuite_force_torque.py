"""MyoSuite force/torque overlay source (GCV-2, #11708; epic #11706).

MyoSuite holds a MuJoCo ``MjModel`` / ``MjData`` pair, so the extraction is the
MuJoCo one (:mod:`mujoco.python.ground_contacts`) with the frame tagged
``myosuite``.  Foot membership comes from the shared
:func:`~src.shared.python.biomechanics.foot_membership.foot_of_body` rule, which
covers the MyoHub ``calcn_l`` / ``calcn_r`` bodies and the four foot contact
bodies of ``golfer_scene`` (``ud_contact_heel_*`` / ``ud_contact_forefoot_*``).

The golfer scene's foot spheres do not collide natively (``contype=0``); their
ground reaction comes from the shared contact law through the bundle overlay
provider, not from this source.  This source reports the contacts MuJoCo itself
solved.  No MyoSuite import is needed here, so it works without the SDK.
"""

from __future__ import annotations

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_torque_source import (
    MujocoForceTorqueSource,
)

__all__ = ["MyoSuiteForceTorqueSource"]


class MyoSuiteForceTorqueSource(MujocoForceTorqueSource):
    """World-frame force/torque overlays from a MyoSuite-held MuJoCo model.

    Preconditions: ``model`` is a ``mujoco.MjModel`` (``env.mj_model``).
    Postconditions: frames carry ``engine == "myosuite"``; the ground-reaction
    breakdown is present only while a foot carries load.
    """

    ENGINE = "myosuite"
