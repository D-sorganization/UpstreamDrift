"""Render :class:`GolfModelParams` to MuJoCo MJCF XML (one model, many renderers).

This is the second renderer of the single source of truth (the first being
:meth:`GolfModelParams.to_double_pendulum_parameters`). Generating the MJCF
mechanically from the same parameter object — rather than hand-maintaining a
parallel XML file — is what prevents the analytical model and the MuJoCo model
from silently drifting apart (epic task M2.3).

Geometry & sign conventions (must match the analytical model so that
cross-validation in :mod:`simulation_backends.validation` holds):

* Planar mechanism in the world ``x``-``y`` plane; both joints are hinges about
  the ``z`` axis (out of the swing plane).
* At ``q = 0`` both links hang **straight down** (along ``-y``). With gravity
  along ``-y`` this makes the gravitational torque proportional to ``sin(theta)``
  and zero at ``q = 0`` — identical to the analytical ``gravity_vector``.
* Gravity magnitude is :attr:`GolfModelParams.projected_gravity` (already
  projected onto the swing plane), so the 2-D model reproduces the in-plane
  dynamics of the inclined 3-D swing.
* Each body's COM sits at its segment's COM distance from the proximal joint;
  the inertia tensor is isotropic with the segment's about-COM inertia. Only the
  ``z`` component enters ``M(q)`` for a ``z``-axis hinge, and MuJoCo's
  parallel-axis composition recovers ``I_com + m * lc**2`` automatically.

The integrator is RK4 to match the analytical RK4 stepper.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from src.shared.python.core.contracts import require

from .model_params import GolfModelParams

#: Stable MuJoCo names for the two bodies/joints, reused by the backends.
UPPER_BODY = "upper"
LOWER_BODY = "lower"
SHOULDER_JOINT = "shoulder"
WRIST_JOINT = "wrist"

#: Default integration step written into the MJCF ``<option>`` element [s]. The
#: backend overrides ``model.opt.timestep`` at run time; this is only a default.
DEFAULT_TIMESTEP_S = 0.01


def params_to_mjcf(
    params: GolfModelParams, *, timestep_s: float = DEFAULT_TIMESTEP_S
) -> str:
    """Render ``params`` to a MuJoCo MJCF XML document string.

    Args:
        params: The single source of truth for the mechanism.
        timestep_s: Default integrator timestep written into ``<option>``.

    Returns:
        A well-formed MJCF XML string loadable via
        :func:`mujoco.MjModel.from_xml_string`. The model has ``nq == nv == 2``
        and two position actuators-free torque actuators (``motor``) on the
        shoulder and wrist joints.

    Raises:
        TypeError: If ``params`` is not a :class:`GolfModelParams`.
        ValueError: If ``timestep_s`` is not strictly positive.
    """
    require(isinstance(params, GolfModelParams), "params must be a GolfModelParams")
    require(timestep_s > 0.0, "timestep_s must be > 0")

    dp = params.to_double_pendulum_parameters()
    m1 = params.upper.mass_kg
    l1 = params.upper.length_m
    lc1 = dp.upper_segment.center_of_mass_distance
    i1c = params.upper.effective_inertia_about_com

    m2 = dp.lower_segment.total_mass
    l2 = params.lower.length_m
    lc2 = dp.lower_segment.center_of_mass_distance
    i2c = dp.lower_segment.inertia_about_com

    gravity = params.projected_gravity
    d1 = params.damping_shoulder
    d2 = params.damping_wrist

    # Links hang along -y at q=0; COMs sit below each proximal joint.
    upper_inertial = _inertial(mass=m1, com_y=-lc1, inertia=i1c)
    lower_inertial = _inertial(mass=m2, com_y=-lc2, inertia=i2c)

    return f"""<mujoco model="golf_double_pendulum">
  <compiler angle="radian" coordinate="local" inertiafromgeom="false"/>
  <option timestep="{timestep_s:.10g}" gravity="0 {-gravity:.10g} 0" integrator="RK4"/>
  <default>
    <joint type="hinge" axis="0 0 1" limited="false"/>
  </default>
  <worldbody>
    <body name="{UPPER_BODY}" pos="0 0 0">
      <joint name="{SHOULDER_JOINT}" pos="0 0 0" damping="{d1:.10g}"/>
      {upper_inertial}
      <geom type="capsule" fromto="0 0 0 0 {-l1:.10g} 0" size="0.02" mass="0"/>
      <body name="{LOWER_BODY}" pos="0 {-l1:.10g} 0">
        <joint name="{WRIST_JOINT}" pos="0 0 0" damping="{d2:.10g}"/>
        {lower_inertial}
        <geom type="capsule" fromto="0 0 0 0 {-l2:.10g} 0" size="0.015" mass="0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="{escape(SHOULDER_JOINT)}_motor" joint="{SHOULDER_JOINT}" gear="1"/>
    <motor name="{escape(WRIST_JOINT)}_motor" joint="{WRIST_JOINT}" gear="1"/>
  </actuator>
</mujoco>
"""


def _inertial(*, mass: float, com_y: float, inertia: float) -> str:
    """Return an ``<inertial>`` element with an isotropic inertia tensor.

    Only the ``z`` component affects ``M(q)`` for a ``z``-axis hinge; an
    isotropic ``diaginertia`` is always positive-definite and keeps MuJoCo's
    inertia validation happy while preserving the correct ``Izz``.
    """
    inertia_str = f"{inertia:.12g}"
    return (
        f'<inertial pos="0 {com_y:.10g} 0" mass="{mass:.10g}" '
        f'diaginertia="{inertia_str} {inertia_str} {inertia_str}"/>'
    )
