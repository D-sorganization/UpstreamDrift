"""Unit tests for MJX tracking plant and differentiable rollout (#11039)."""

from __future__ import annotations

from typing import Any

import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from src.shared.python.core.contracts import ContractViolationError
from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    GroundPlane,
)
from src.shared.python.motion_matching.jax_contact import (
    WeldGains,
    sphere_ground_contact_jax,
)
from src.shared.python.motion_matching.knot_gradient_optimiser import (
    knot_basis,
    knot_grid,
)
from src.engines.physics_engines.mujoco.python.motion_matching.mjx_tracking_plant import (
    TrackingPlantSpec,
    build_tracking_plant,
    computed_torque,
    initial_state,
    markers,
    reference_derivatives,
    sphere_wrenches,
    substep_tables,
)

pytestmark = pytest.mark.unit

TOY_TRACKING_XML = """
<mujoco model="toy_tracking">
  <option timestep="0.005" integrator="Euler"/>
  <default>
    <geom contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <body name="pelvis" pos="0 0 1">
      <joint name="root_z" type="slide" axis="0 0 1"/>
      <geom name="pelvis_geom" type="sphere" size="0.05" mass="1.0"/>
      <site name="root_marker" pos="0 0 0.05"/>
      <body name="thigh" pos="0 0 -0.2">
        <joint name="joint_1" type="hinge" axis="0 1 0"/>
        <geom name="thigh_geom" type="cylinder" size="0.03 0.1" mass="1.0"/>
        <body name="shin" pos="0 0 -0.2">
          <joint name="joint_2" type="hinge" axis="0 1 0"/>
          <geom name="shin_geom" type="cylinder" size="0.03 0.1" mass="1.0"/>
          <site name="foot_sphere" pos="0 0 -0.1"/>
          <site name="shin_marker" pos="0 0 -0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

TOY_TRACKING_NOGRAV_XML = """
<mujoco model="toy_tracking_nograv">
  <option timestep="0.005" integrator="Euler" gravity="0 0 0"/>
  <default>
    <geom contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <body name="pelvis" pos="0 0 1">
      <joint name="root_z" type="slide" axis="0 0 1"/>
      <geom name="pelvis_geom" type="sphere" size="0.05" mass="1.0"/>
      <site name="root_marker" pos="0 0 0.05"/>
      <body name="thigh" pos="0 0 -0.2">
        <joint name="joint_1" type="hinge" axis="0 1 0"/>
        <geom name="thigh_geom" type="cylinder" size="0.03 0.1" mass="1.0"/>
        <body name="shin" pos="0 0 -0.2">
          <joint name="joint_2" type="hinge" axis="0 1 0"/>
          <geom name="shin_geom" type="cylinder" size="0.03 0.1" mass="1.0"/>
          <site name="foot_sphere" pos="0 0 -0.1"/>
          <site name="shin_marker" pos="0 0 -0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

TOY_WELD_XML = """
<mujoco model="toy_weld">
  <option timestep="0.005" integrator="Euler"/>
  <default>
    <geom contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <body name="body_a" pos="0 0 1">
      <joint name="joint_a" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.05" mass="1.0"/>
      <site name="site_a" pos="0.1 0 0"/>
    </body>
    <body name="body_b" pos="0.1 0 1">
      <joint name="joint_b" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.05" mass="1.0"/>
      <site name="site_b" pos="0 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""

TOY_EQUALITY_XML = """
<mujoco model="toy_equality">
  <option timestep="0.005" integrator="Euler"/>
  <worldbody>
    <body name="body_a" pos="0 0 1">
      <joint name="joint_a" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.05" mass="1.0"/>
      <site name="site_a" pos="0.1 0 0"/>
    </body>
    <body name="body_b" pos="0.1 0 1">
      <joint name="joint_b" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.05" mass="1.0"/>
      <site name="site_b" pos="0 0 0"/>
    </body>
  </worldbody>
  <equality>
    <weld site1="site_a" site2="site_b"/>
  </equality>
</mujoco>
"""


def _sample_contact() -> ContactParameters:
    return ContactParameters(
        stiffness_n_m=10000.0,
        dissipation_s_m=0.1,
        static_friction=0.8,
        dynamic_friction=0.6,
        viscous_friction=0.01,
        transition_velocity_m_s=0.01,
    )


def _sample_spec(*, ground_h: float = 0.0) -> TrackingPlantSpec:
    return TrackingPlantSpec(
        rate_hz=100.0,
        substeps=2,
        qpos_adr=(0, 1, 2),
        dof_adr=(0, 1, 2),
        root_mask=(True, False, False),
        root_vertical_index=0,
        sphere_site_ids=(1,),  # site 1 is foot_sphere (site 0 is root_marker)
        sphere_body_ids=(3,),  # body 3 is shin
        sphere_radii_m=(0.05,),
        marker_body_ids=(1, 3),  # pelvis, shin
        marker_local_m=((0.0, 0.0, 0.05), (0.0, 0.0, -0.05)),
        omega_rad_s=60.0,
        zeta=1.0,
        contact=_sample_contact(),
        ground=GroundPlane(normal=(0.0, 0.0, 1.0), height_m=ground_h),
        closure=None,
    )


def test_computed_torque_identity() -> None:
    """After one forward pass with computed torque, actuated qacc == wanted to 1e-8."""
    model = mujoco.MjModel.from_xml_string(TOY_TRACKING_XML)
    spec = _sample_spec()
    mm = mjx.put_model(model)
    d = mjx.make_data(mm)

    q = jnp.array([1.0, 0.3, -0.4])
    v = jnp.array([0.1, -0.2, 0.5])
    d = mjx.forward(mm, d.replace(qpos=q, qvel=v))

    wanted_act = jnp.array([2.5, -4.0])
    tau = computed_torque(mm, spec, d, wanted_act)

    root_dofs = np.asarray(spec.root_dofs)
    act_dofs = np.asarray(spec.act_dofs)
    assert np.allclose(np.asarray(tau[root_dofs]), 0.0, atol=1e-12)

    d_next = mjx.forward(mm, d.replace(qfrc_applied=tau))
    qacc_act = np.asarray(d_next.qacc[act_dofs])
    wanted_np = np.asarray(wanted_act)
    err = np.max(np.abs(qacc_act - wanted_np))
    assert err < 1e-8


def test_contact_wiring() -> None:
    """Wrench added to sphere body equals sphere_ground_contact_jax plus CoM moment to 1e-12."""
    model = mujoco.MjModel.from_xml_string(TOY_TRACKING_XML)
    spec = _sample_spec(ground_h=0.0)
    mm = mjx.put_model(model)
    d = mjx.make_data(mm)

    # Set state so the foot sphere is penetrating below ground plane z=0
    # Pelvis pos=1.0, thigh offset=-0.2, shin offset=-0.2, site pos=-0.1 -> site z = 0.5
    # To place site at z = -0.02 (with radius 0.05, bottom is -0.07):
    # root_z = -0.52
    q = jnp.array([-0.52, 0.0, 0.0])
    v = jnp.array([-0.1, 0.2, -0.3])
    d = mjx.forward(mm, d.replace(qpos=q, qvel=v))

    site_id = spec.sphere_site_ids[0]
    body_id = spec.sphere_body_ids[0]
    radius = spec.sphere_radii_m[0]

    centre = d.site_xpos[site_id]
    jacp, _ = mjx.jac(mm, d, centre, body_id)
    vel = jacp.T @ d.qvel

    f_norm, f_fric = sphere_ground_contact_jax(
        centre,
        vel,
        radius,
        spec.ground,
        spec.contact,
    )
    expected_force = f_norm + f_fric
    ground_normal = jnp.asarray(spec.ground.normal)
    point = centre - ground_normal * radius
    expected_torque = jnp.cross(point - d.xipos[body_id], expected_force)
    expected_wrench = np.concatenate([expected_force, expected_torque])

    xfrc = sphere_wrenches(mm, spec, d)
    actual_wrench = np.asarray(xfrc[body_id])
    assert np.allclose(actual_wrench, expected_wrench, atol=1e-12)


def test_initial_state_static_penetration() -> None:
    """Lowest sphere ends at static penetration m*g/(k*n_spheres) to 1e-9."""
    model = mujoco.MjModel.from_xml_string(TOY_TRACKING_XML)
    spec = _sample_spec(ground_h=0.0)
    plant = build_tracking_plant(model, spec)

    mass_kg = float(np.sum(model.body_mass))
    opt = model.opt
    gravity = opt.gravity
    g_eff = float(-np.dot(gravity, spec.ground.normal))  # positive downward onto ground
    k_n = spec.contact.stiffness_n_m
    n_spheres = len(spec.sphere_site_ids)
    expected_penetration = mass_kg * g_eff / (k_n * n_spheres)

    q0 = np.array([0.0, 0.1, -0.1])
    v0 = np.array([0.0, 0.0, 0.0])
    d0 = plant.initial_state(q0, v0, mass_kg)

    site_id = spec.sphere_site_ids[0]
    radius = spec.sphere_radii_m[0]
    centre_z = float(d0.site_xpos[site_id][2])
    bottom_z = centre_z - radius
    actual_penetration = -bottom_z
    assert abs(actual_penetration - expected_penetration) < 1e-9


def test_tracking_no_contact() -> None:
    """With ground far below and smooth reference, marker RMS < 1 mm over 20 frames."""
    model = mujoco.MjModel.from_xml_string(TOY_TRACKING_NOGRAV_XML)
    # Ground far below: no contact forces
    spec = _sample_spec(ground_h=-100.0)
    plant = build_tracking_plant(model, spec)

    n_frames = 20
    dt_frame = 1.0 / spec.rate_hz
    times = np.linspace(0.0, (n_frames - 1) * dt_frame, n_frames)

    # Smooth trajectory for root and joints
    q_ref = np.zeros((n_frames, 3))
    q_ref[:, 0] = 1.0  # root_z
    q_ref[:, 1] = 0.05 * np.sin(2.0 * np.pi * 1.0 * times)
    q_ref[:, 2] = -0.05 * np.cos(2.0 * np.pi * 1.0 * times)

    v_ref, a_ref = reference_derivatives(q_ref, times)
    q_sub = substep_tables(q_ref, spec.substeps)
    v_sub = substep_tables(v_ref, spec.substeps)
    a_sub = substep_tables(a_ref, spec.substeps)

    d0 = mjx.make_data(plant.mjx_model)
    d0 = mjx.forward(
        plant.mjx_model,
        d0.replace(
            qpos=jnp.asarray(q_ref[0]),
            qvel=jnp.asarray(v_ref[0]),
        ),
    )

    markers_sim = plant.rollout(d0, q_sub, v_sub, a_sub)

    # Compute target markers directly from kinematics of q_ref
    target_markers = []
    d_kin = mjx.make_data(plant.mjx_model)
    for k in range(n_frames):
        d_k = mjx.forward(
            plant.mjx_model,
            d_kin.replace(qpos=jnp.asarray(q_ref[k]), qvel=jnp.zeros(3)),
        )
        target_markers.append(np.asarray(markers(plant.mjx_model, spec, d_k)))
    targets = np.stack(target_markers, axis=0)

    err = np.asarray(markers_sim) - targets
    marker_rms_m = float(np.sqrt(np.mean(err**2)))
    assert marker_rms_m < 1e-3  # below 1 mm


def test_gradient_matches_finite_difference() -> None:
    """jax.grad of marker cost w.r.t 3-knot correction matches central FD to 1e-4 relative."""
    model = mujoco.MjModel.from_xml_string(TOY_TRACKING_NOGRAV_XML)
    spec = _sample_spec(ground_h=-100.0)
    plant = build_tracking_plant(model, spec)

    n_frames = 5
    dt_frame = 1.0 / spec.rate_hz
    times = np.linspace(0.0, (n_frames - 1) * dt_frame, n_frames)

    knots = knot_grid(times, spacing_s=times[-1] / 2.0)
    assert len(knots) == 3
    basis = knot_basis(times, knots)  # shape (5, 3)

    q_nom = np.zeros((n_frames, 3))
    q_nom[:, 0] = 1.0
    q_nom[:, 1] = 0.05 * np.sin(np.pi * times / times[-1])
    q_nom[:, 2] = -0.05 * np.cos(np.pi * times / times[-1])

    act_spec = np.asarray(spec.act_indices)
    basis_j = jnp.asarray(basis)
    q_nom_j = jnp.asarray(q_nom)

    # Build targets
    v_nom, a_nom = reference_derivatives(q_nom, times)
    q_sub_nom = substep_tables(q_nom, spec.substeps)
    v_sub_nom = substep_tables(v_nom, spec.substeps)
    a_sub_nom = substep_tables(a_nom, spec.substeps)
    d0 = mjx.make_data(plant.mjx_model)
    d0 = mjx.forward(
        plant.mjx_model,
        d0.replace(
            qpos=jnp.asarray(q_nom[0]),
            qvel=jnp.asarray(v_nom[0]),
        ),
    )
    target_markers = plant.rollout(d0, q_sub_nom, v_sub_nom, a_sub_nom)

    def cost(delta: Any) -> Any:
        # delta shape: (3 knots, 2 actuated coordinates)
        correction = basis_j @ delta  # (5, 2)
        q = q_nom_j.at[:, act_spec].add(correction)
        v, a = reference_derivatives(q, times)
        q_s = substep_tables(q, spec.substeps)
        v_s = substep_tables(v, spec.substeps)
        a_s = substep_tables(a, spec.substeps)
        m = plant.rollout(d0, q_s, v_s, a_s)
        return jnp.sum((m - target_markers) ** 2)

    delta0 = jnp.array([[0.01, -0.01], [0.02, 0.01], [-0.01, 0.02]], dtype=jnp.float64)
    grad_jax = jax.grad(cost)(delta0)

    # Central finite differences
    eps = 1e-6
    grad_fd = np.zeros_like(delta0)
    for i in range(delta0.shape[0]):
        for j in range(delta0.shape[1]):
            d_plus = delta0.at[i, j].add(eps)
            d_minus = delta0.at[i, j].add(-eps)
            c_plus = float(cost(d_plus))
            c_minus = float(cost(d_minus))
            grad_fd[i, j] = (c_plus - c_minus) / (2.0 * eps)

    grad_jax_np = np.asarray(grad_jax)
    denom = np.maximum(np.abs(grad_jax_np), 1e-5)
    rel_err = np.max(np.abs(grad_jax_np - grad_fd) / denom)
    assert rel_err < 1e-4


def test_refusal_model_with_equality() -> None:
    """Model with <equality> constraint is refused."""
    model = mujoco.MjModel.from_xml_string(TOY_EQUALITY_XML)
    spec = _sample_spec()
    with pytest.raises(ValueError, match="equality"):
        build_tracking_plant(model, spec)


def test_refusal_out_of_range_indices() -> None:
    """Out-of-range sphere and marker IDs are refused."""
    model = mujoco.MjModel.from_xml_string(TOY_TRACKING_XML)

    # Out of range sphere site
    bad_sphere_spec = TrackingPlantSpec(
        rate_hz=100.0,
        substeps=2,
        qpos_adr=(0, 1, 2),
        dof_adr=(0, 1, 2),
        root_mask=(True, False, False),
        root_vertical_index=0,
        sphere_site_ids=(999,),
        sphere_body_ids=(3,),
        sphere_radii_m=(0.05,),
        marker_body_ids=(1, 3),
        marker_local_m=((0.0, 0.0, 0.05), (0.0, 0.0, -0.05)),
        omega_rad_s=60.0,
        zeta=1.0,
        contact=_sample_contact(),
        ground=GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0),
    )
    with pytest.raises(ValueError, match="Sphere site ID"):
        build_tracking_plant(model, bad_sphere_spec)

    # Out of range marker body
    bad_marker_spec = TrackingPlantSpec(
        rate_hz=100.0,
        substeps=2,
        qpos_adr=(0, 1, 2),
        dof_adr=(0, 1, 2),
        root_mask=(True, False, False),
        root_vertical_index=0,
        sphere_site_ids=(1,),
        sphere_body_ids=(3,),
        sphere_radii_m=(0.05,),
        marker_body_ids=(999, 3),
        marker_local_m=((0.0, 0.0, 0.05), (0.0, 0.0, -0.05)),
        omega_rad_s=60.0,
        zeta=1.0,
        contact=_sample_contact(),
        ground=GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0),
    )
    with pytest.raises(ValueError, match="Marker body ID"):
        build_tracking_plant(model, bad_marker_spec)


def test_weld_variant() -> None:
    """Model with two bodies and weld spring-damper runs without error."""
    model = mujoco.MjModel.from_xml_string(TOY_WELD_XML)
    gains = WeldGains(k=1000.0, c=10.0, rot_k=50.0, rot_c=1.0)
    spec = TrackingPlantSpec(
        rate_hz=100.0,
        substeps=2,
        qpos_adr=(0, 1),
        dof_adr=(0, 1),
        root_mask=(True, False),
        root_vertical_index=0,
        sphere_site_ids=(),
        sphere_body_ids=(),
        sphere_radii_m=(),
        marker_body_ids=(1, 2),
        marker_local_m=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        omega_rad_s=60.0,
        zeta=1.0,
        contact=_sample_contact(),
        ground=GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0),
        closure=(0, 1, gains),
    )
    plant = build_tracking_plant(model, spec)
    d0 = mjx.make_data(plant.mjx_model)
    q_sub = jnp.zeros((3, 2, 2))
    v_sub = jnp.zeros((3, 2, 2))
    a_sub = jnp.zeros((3, 2, 2))
    m = plant.rollout(d0, q_sub, v_sub, a_sub)
    assert m.shape == (3, 2, 3)


def test_from_package_parser() -> None:
    """TrackingPlantSpec.from_package parses exported format correctly."""
    meta = {
        "rate_hz": 200.0,
        "timestep_s": 0.0025,
        "coordinate_order": ["root_z", "joint_1", "joint_2"],
        "controller": {"omega_rad_s": 60.0, "zeta": 1.0},
        "contact": _sample_contact().as_document(),
        "closure": {"site_a": 0, "site_b": 1},
    }
    pkg = {
        "qpos_adr": np.array([0, 1, 2]),
        "dof_adr": np.array([0, 1, 2]),
        "root_mask": np.array([True, False, False]),
        "sphere_site_ids": np.array([1]),
        "sphere_body_ids": np.array([3]),
        "sphere_radii_m": np.array([0.05]),
        "marker_body_ids": np.array([1, 3]),
        "marker_local_m": np.array([[0.0, 0.0, 0.05], [0.0, 0.0, -0.05]]),
        "ground_normal": np.array([0.0, 0.0, 1.0]),
        "ground_height_m": np.array(0.0),
    }
    weld_gains = WeldGains(k=1000.0, c=10.0, rot_k=50.0, rot_c=1.0)
    spec = TrackingPlantSpec.from_package(
        meta, pkg, substeps=2, root_vertical_index=0, weld_gains=weld_gains
    )

    assert spec.rate_hz == 200.0
    assert spec.substeps == 2
    assert spec.root_vertical_index == 0
    assert spec.closure == (0, 1, weld_gains)
    assert len(spec.sphere_site_ids) == 1
    assert len(spec.marker_body_ids) == 2


def test_from_package_refuses_closure_without_weld_gains() -> None:
    """A package that declares a grip closure must not silently lose the weld."""
    meta = {
        "rate_hz": 200.0,
        "controller": {"omega_rad_s": 60.0, "zeta": 1.0},
        "contact": _sample_contact().as_document(),
        "closure": {"site_a": 0, "site_b": 1},
    }
    pkg = {
        "qpos_adr": np.array([0, 1, 2]),
        "dof_adr": np.array([0, 1, 2]),
        "root_mask": np.array([True, False, False]),
        "sphere_site_ids": np.array([1]),
        "sphere_body_ids": np.array([3]),
        "sphere_radii_m": np.array([0.05]),
        "marker_body_ids": np.array([1]),
        "marker_local_m": np.array([[0.0, 0.0, 0.05]]),
        "ground_normal": np.array([0.0, 0.0, 1.0]),
        "ground_height_m": np.array(0.0),
    }
    with pytest.raises(ContractViolationError, match="weld_gains"):
        TrackingPlantSpec.from_package(meta, pkg, substeps=2, root_vertical_index=0)


def test_build_tracking_plant_leaves_caller_model_timestep_unchanged() -> None:
    """build_tracking_plant sets the substep timestep on its own model copy."""
    model = mujoco.MjModel.from_xml_string(TOY_TRACKING_XML)
    original_timestep = float(model.opt.timestep)
    spec = _sample_spec()
    plant = build_tracking_plant(model, spec)

    assert float(model.opt.timestep) == original_timestep
    expected = 1.0 / (spec.rate_hz * spec.substeps)
    assert float(plant.model.opt.timestep) == pytest.approx(expected, rel=1e-12)
