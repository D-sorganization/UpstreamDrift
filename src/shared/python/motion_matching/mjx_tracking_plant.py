"""MJX tracking plant and differentiable rollout for motion matching (#11039)."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from src.shared.python.core.contracts import PreconditionError, require
from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    GroundPlane,
)
from src.shared.python.motion_matching.jax_contact import (
    SiteState,
    WeldGains,
    sphere_ground_contact_jax,
    weld_wrench_jax,
)


@dataclass(frozen=True)
class TrackingPlantSpec:
    """Specification of an MJX tracking plant.

    Parameters:
        rate_hz: Control and reference rate in Hz.
        substeps: Number of integration substeps per frame.
        qpos_adr: Indices into MuJoCo qpos for coordinates in spec order.
        dof_adr: Indices into MuJoCo qvel/qacc for coordinates in spec order.
        root_mask: True for unactuated root coordinates, False for actuated.
        root_vertical_index: Spec coordinate index of root vertical slide.
        sphere_site_ids: Site IDs for ground-contact foot spheres.
        sphere_body_ids: Body IDs corresponding to sphere sites.
        sphere_radii_m: Radii of contact spheres in metres.
        marker_body_ids: Body IDs on which tracking markers are mounted.
        marker_local_m: Local 3D offsets of markers on their bodies.
        omega_rad_s: Natural frequency of tracking PD controller (rad/s).
        zeta: Damping ratio of tracking PD controller.
        contact: Hunt-Crossley and friction parameters.
        ground: Ground plane normal and height.
        closure: Optional (site_a, site_b, WeldGains) for grip closure reaction.
    """

    rate_hz: float
    substeps: int
    qpos_adr: tuple[int, ...]
    dof_adr: tuple[int, ...]
    root_mask: tuple[bool, ...]
    root_vertical_index: int
    sphere_site_ids: tuple[int, ...]
    sphere_body_ids: tuple[int, ...]
    sphere_radii_m: tuple[float, ...]
    marker_body_ids: tuple[int, ...]
    marker_local_m: tuple[tuple[float, float, float], ...]
    omega_rad_s: float
    zeta: float
    contact: ContactParameters
    ground: GroundPlane
    closure: tuple[int, int, WeldGains] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rate_hz", float(self.rate_hz))
        object.__setattr__(self, "substeps", int(self.substeps))
        object.__setattr__(self, "qpos_adr", tuple(int(x) for x in self.qpos_adr))
        object.__setattr__(self, "dof_adr", tuple(int(x) for x in self.dof_adr))
        object.__setattr__(self, "root_mask", tuple(bool(x) for x in self.root_mask))
        object.__setattr__(self, "root_vertical_index", int(self.root_vertical_index))
        object.__setattr__(
            self, "sphere_site_ids", tuple(int(x) for x in self.sphere_site_ids)
        )
        object.__setattr__(
            self, "sphere_body_ids", tuple(int(x) for x in self.sphere_body_ids)
        )
        object.__setattr__(
            self, "sphere_radii_m", tuple(float(x) for x in self.sphere_radii_m)
        )
        object.__setattr__(
            self, "marker_body_ids", tuple(int(x) for x in self.marker_body_ids)
        )
        object.__setattr__(
            self,
            "marker_local_m",
            tuple(
                (float(pt[0]), float(pt[1]), float(pt[2])) for pt in self.marker_local_m
            ),
        )
        object.__setattr__(self, "omega_rad_s", float(self.omega_rad_s))
        object.__setattr__(self, "zeta", float(self.zeta))
        closure = self.closure
        if closure is not None:
            object.__setattr__(
                self,
                "closure",
                (int(closure[0]), int(closure[1]), closure[2]),
            )

        require(self.rate_hz > 0.0, "rate_hz must be positive", self.rate_hz)
        require(self.substeps >= 1, "substeps must be >= 1", self.substeps)
        n_coords = len(self.qpos_adr)
        require(n_coords > 0, "qpos_adr must not be empty")
        require(len(self.dof_adr) == n_coords, "dof_adr length mismatch")
        require(len(self.root_mask) == n_coords, "root_mask length mismatch")
        r_idx = self.root_vertical_index
        require(
            0 <= r_idx < n_coords,
            "root_vertical_index out of bounds",
            r_idx,
        )
        require(
            self.root_mask[r_idx],
            "root_vertical_index must point to a root coordinate",
            r_idx,
        )

        n_spheres = len(self.sphere_site_ids)
        require(
            len(self.sphere_body_ids) == n_spheres,
            "sphere_body_ids length mismatch",
        )
        require(
            len(self.sphere_radii_m) == n_spheres,
            "sphere_radii_m length mismatch",
        )
        for r in self.sphere_radii_m:
            require(r > 0.0, "sphere radius must be positive", r)

        n_markers = len(self.marker_body_ids)
        require(
            len(self.marker_local_m) == n_markers,
            "marker_local_m length mismatch",
        )
        for pt in self.marker_local_m:
            require(len(pt) == 3, "marker local offset must be 3-vector", len(pt))

        require(
            self.omega_rad_s > 0.0,
            "omega_rad_s must be positive",
            self.omega_rad_s,
        )
        require(self.zeta >= 0.0, "zeta must be nonnegative", self.zeta)

        require(
            isinstance(self.contact, ContactParameters),
            "contact must be ContactParameters",
        )
        require(isinstance(self.ground, GroundPlane), "ground must be GroundPlane")
        if self.closure is not None:
            cl_a, cl_b, gains = self.closure
            require(cl_a >= 0 and cl_b >= 0, "closure site IDs must be nonnegative")
            require(isinstance(gains, WeldGains), "closure gains must be WeldGains")

    @property
    def act_indices(self) -> tuple[int, ...]:
        """Indices of actuated coordinates in spec ordering."""
        return tuple(i for i, is_root in enumerate(self.root_mask) if not is_root)

    @property
    def root_indices(self) -> tuple[int, ...]:
        """Indices of root (unactuated) coordinates in spec ordering."""
        return tuple(i for i, is_root in enumerate(self.root_mask) if is_root)

    @property
    def act_dofs(self) -> tuple[int, ...]:
        """MuJoCo dof indices of actuated coordinates."""
        return tuple(self.dof_adr[i] for i in self.act_indices)

    @property
    def root_dofs(self) -> tuple[int, ...]:
        """MuJoCo dof indices of root (unactuated) coordinates."""
        return tuple(self.dof_adr[i] for i in self.root_indices)

    @classmethod
    def from_package(
        cls,
        meta: Mapping[str, Any],
        pkg: Mapping[str, Any],
        *,
        substeps: int,
        root_vertical_index: int,
        weld_gains: WeldGains | None = None,
    ) -> TrackingPlantSpec:
        """Parse spec from exported package metadata and array dict.

        ``substeps`` and ``root_vertical_index`` are required: the package does
        not record them, and guessing the vertical coordinate would silently
        preload the wrong root dof. A package that declares a grip closure must
        be given ``weld_gains``: the exporter strips the weld equality, so
        dropping the closure would let the grip come apart unnoticed.
        """
        rate_hz = float(meta["rate_hz"])
        sub_steps = int(substeps)

        qpos_adr = tuple(int(x) for x in pkg["qpos_adr"])
        dof_adr = tuple(int(x) for x in pkg["dof_adr"])
        root_mask = tuple(bool(x) for x in pkg["root_mask"])

        r_idx = int(root_vertical_index)

        ctrl = meta["controller"]
        omega_rad_s = float(ctrl["omega_rad_s"])
        zeta = float(ctrl["zeta"])

        c_doc = meta["contact"]
        if isinstance(c_doc, ContactParameters):
            contact = c_doc
        else:
            contact = ContactParameters(
                stiffness_n_m=float(c_doc["stiffness_n_m"]),
                dissipation_s_m=float(c_doc["dissipation_s_m"]),
                static_friction=float(c_doc["static_friction"]),
                dynamic_friction=float(c_doc["dynamic_friction"]),
                viscous_friction=float(c_doc["viscous_friction"]),
                transition_velocity_m_s=float(c_doc["transition_velocity_m_s"]),
            )

        normal_raw = np.asarray(pkg["ground_normal"], dtype=float).flatten()
        normal = (
            float(normal_raw[0]),
            float(normal_raw[1]),
            float(normal_raw[2]),
        )
        height_m = float(np.asarray(pkg["ground_height_m"], dtype=float).item())
        ground = GroundPlane(normal=normal, height_m=height_m)

        closure: tuple[int, int, WeldGains] | None = None
        cl = meta.get("closure")
        if cl is not None:
            # Unconditional (not ``require``): dropping a declared grip weld
            # changes the physics even when contract checking is switched off.
            if weld_gains is None:
                raise PreconditionError(
                    "package declares a grip closure; weld_gains are required",
                    parameter="weld_gains",
                )
            closure = (int(cl["site_a"]), int(cl["site_b"]), weld_gains)

        sphere_site_ids = tuple(int(x) for x in pkg["sphere_site_ids"])
        sphere_body_ids = tuple(int(x) for x in pkg["sphere_body_ids"])
        sphere_radii_m = tuple(float(x) for x in pkg["sphere_radii_m"])

        marker_body_ids = tuple(int(x) for x in pkg["marker_body_ids"])
        marker_local_m = tuple(
            (float(pt[0]), float(pt[1]), float(pt[2])) for pt in pkg["marker_local_m"]
        )

        return cls(
            rate_hz=rate_hz,
            substeps=sub_steps,
            qpos_adr=qpos_adr,
            dof_adr=dof_adr,
            root_mask=root_mask,
            root_vertical_index=r_idx,
            sphere_site_ids=sphere_site_ids,
            sphere_body_ids=sphere_body_ids,
            sphere_radii_m=sphere_radii_m,
            marker_body_ids=marker_body_ids,
            marker_local_m=marker_local_m,
            omega_rad_s=omega_rad_s,
            zeta=zeta,
            contact=contact,
            ground=ground,
            closure=closure,
        )

    def validate_against_model(self, model: mujoco.MjModel) -> None:
        """Verify model compatibility, equality absence, and index bounds."""
        neq = model.neq
        require(
            neq == 0,
            f"Model carries {neq} equality constraints; MJX constraint solve is not "
            "reverse-differentiable. Remove equalities.",
        )
        nsite = model.nsite
        nbody = model.nbody
        nq = model.nq
        nv = model.nv

        for site_id in self.sphere_site_ids:
            require(
                0 <= site_id < nsite,
                f"Sphere site ID {site_id} is out of range [0, {nsite})",
            )
        for body_id in self.sphere_body_ids:
            require(
                0 <= body_id < nbody,
                f"Sphere body ID {body_id} is out of range [0, {nbody})",
            )
        for body_id in self.marker_body_ids:
            require(
                0 <= body_id < nbody,
                f"Marker body ID {body_id} is out of range [0, {nbody})",
            )
        closure = self.closure
        if closure is not None:
            site_a, site_b, _ = closure
            require(
                0 <= site_a < nsite,
                f"Closure site A {site_a} is out of range [0, {nsite})",
            )
            require(
                0 <= site_b < nsite,
                f"Closure site B {site_b} is out of range [0, {nsite})",
            )
        for adr in self.qpos_adr:
            require(0 <= adr < nq, f"qpos address {adr} is out of range [0, {nq})")
        for adr in self.dof_adr:
            require(0 <= adr < nv, f"dof address {adr} is out of range [0, {nv})")


@dataclass(frozen=True)
class TrackingPlant:
    """Compiled MJX tracking plant for differentiable rollout."""

    model: mujoco.MjModel
    mjx_model: mjx.Model
    spec: TrackingPlantSpec
    rollout: Callable[[mjx.Data, jax.Array, jax.Array, jax.Array], jax.Array]
    rollout_diagnostic: Callable[
        [mjx.Data, jax.Array, jax.Array, jax.Array],
        tuple[jax.Array, jax.Array, jax.Array],
    ]
    initial_state: Callable[
        [jax.Array | np.ndarray, jax.Array | np.ndarray, float], mjx.Data
    ]
    markers: Callable[[mjx.Data], jax.Array]


def reference_derivatives(
    q: np.ndarray | jax.Array,
    times: np.ndarray | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute central finite-difference velocities and accelerations along time axis.

    Parameters:
        q: Reference trajectory of shape (frames, coordinates).
        times: Monotonically increasing time stamps of shape (frames,).

    Returns:
        Tuple of (v, a) of identical shape to q.
    """
    times_arr = np.asarray(times)
    require(times_arr.ndim == 1, "times must be 1-D", times_arr.ndim)
    require(times_arr.size >= 2, "times must have >= 2 points", times_arr.size)
    dt_arr = np.diff(times_arr)
    require(bool(np.all(dt_arr > 0.0)), "times must be strictly increasing")
    dt = float(np.mean(dt_arr))
    require(dt > 0.0, "time step must be positive", dt)

    q_j = jnp.asarray(q)
    require(q_j.shape[0] == times_arr.size, "q frame count must match times")
    v = jnp.gradient(q_j, axis=0) / dt
    a = jnp.gradient(v, axis=0) / dt
    return v, a


def substep_tables(
    x: np.ndarray | jax.Array,
    substeps: int,
) -> jax.Array:
    """Linearly interpolate array across integration substeps within each frame.

    Parameters:
        x: Array of shape (frames, ...).
        substeps: Number of integration substeps per frame (>= 1).

    Returns:
        Array of shape (frames, substeps, ...).
    """
    require(substeps >= 1, "substeps must be >= 1", substeps)
    x_j = jnp.asarray(x)
    require(x_j.ndim >= 1, "x must have at least 1 dimension", x_j.ndim)

    fractions = jnp.arange(substeps, dtype=x_j.dtype) / float(substeps)
    nxt = jnp.concatenate([x_j[1:], x_j[-1:]], axis=0)
    delta = nxt - x_j

    target_shape = (1, substeps) + (1,) * (x_j.ndim - 1)
    frac_b = fractions.reshape(target_shape)
    x_exp = jnp.expand_dims(x_j, axis=1)
    delta_exp = jnp.expand_dims(delta, axis=1)
    return x_exp + frac_b * delta_exp


def markers(
    mjx_model: mjx.Model,
    spec: TrackingPlantSpec,
    d: mjx.Data,
) -> jax.Array:
    """World positions of tracking markers (n_markers, 3)."""
    marker_bodies = spec.marker_body_ids
    if len(marker_bodies) == 0:
        return jnp.zeros((0, 3), dtype=d.qvel.dtype)
    bodies_arr = jnp.asarray(marker_bodies)
    local_arr = jnp.asarray(spec.marker_local_m, dtype=d.qvel.dtype)
    xmat = d.xmat
    xpos = d.xpos
    rot = xmat[bodies_arr]
    pos = xpos[bodies_arr]
    return pos + jnp.einsum("mij,mj->mi", rot, local_arr)


def weld_wrenches(
    mjx_model: mjx.Model,
    spec: TrackingPlantSpec,
    d: mjx.Data,
    xfrc: jax.Array,
) -> jax.Array:
    """Equal-and-opposite spatial reaction wrenches between weld closure sites."""
    closure = spec.closure
    if closure is None:
        return xfrc
    site_a, site_b, weld_gains = closure
    site_bodyid = mjx_model.site_bodyid
    body_a = int(site_bodyid[site_a])
    body_b = int(site_bodyid[site_b])

    site_xpos = d.site_xpos
    site_xmat = d.site_xmat
    p_a = site_xpos[site_a]
    p_b = site_xpos[site_b]
    r_a = site_xmat[site_a]
    r_b = site_xmat[site_b]

    jp_a, jr_a = mjx.jac(mjx_model, d, p_a, body_a)
    jp_b, jr_b = mjx.jac(mjx_model, d, p_b, body_b)
    qvel = d.qvel
    v_a = jp_a.T @ qvel
    v_b = jp_b.T @ qvel
    w_a = jr_a.T @ qvel
    w_b = jr_b.T @ qvel

    xipos = d.xipos
    com_a = xipos[body_a]
    com_b = xipos[body_b]
    state_a = SiteState(p=p_a, v=v_a, r=r_a, w=w_a, com=com_a)
    state_b = SiteState(p=p_b, v=v_b, r=r_b, w=w_b, com=com_b)

    wa, wb = weld_wrench_jax(state_a, state_b, gains=weld_gains)
    return xfrc.at[body_b].add(wb).at[body_a].add(wa)


def sphere_wrenches(
    mjx_model: mjx.Model,
    spec: TrackingPlantSpec,
    d: mjx.Data,
) -> jax.Array:
    """World wrenches (nbody, 6) of shared ground contact plus closure welds."""
    nbody = mjx_model.nbody
    qvel = d.qvel
    xfrc = jnp.zeros((nbody, 6), dtype=qvel.dtype)
    sphere_site_ids = spec.sphere_site_ids
    sphere_body_ids = spec.sphere_body_ids
    sphere_radii_m = spec.sphere_radii_m

    ground = spec.ground
    contact = spec.contact
    normal = ground.normal
    n_ground = jnp.asarray(normal, dtype=qvel.dtype)

    site_xpos = d.site_xpos
    xipos = d.xipos

    for i in range(len(sphere_site_ids)):
        site = int(sphere_site_ids[i])
        body = int(sphere_body_ids[i])
        radius = float(sphere_radii_m[i])

        centre = site_xpos[site]
        jacp, _ = mjx.jac(mjx_model, d, centre, body)
        velocity = jacp.T @ qvel

        f_norm, f_fric = sphere_ground_contact_jax(
            centre,
            velocity,
            radius,
            ground,
            contact,
        )
        force = f_norm + f_fric
        point = centre - n_ground * radius
        torque = jnp.cross(point - xipos[body], force)
        wrench = jnp.concatenate([force, torque])
        xfrc = xfrc.at[body].add(wrench)

    return weld_wrenches(mjx_model, spec, d, xfrc)


def computed_torque(
    mjx_model: mjx.Model,
    spec: TrackingPlantSpec,
    d: mjx.Data,
    wanted_act: jax.Array,
) -> jax.Array:
    """Generalised force for actuated dofs leaving root free (MuJoCo dof order)."""
    m_full = mjx.full_m(mjx_model, d)
    smooth = d.qfrc_smooth + d.qfrc_constraint

    act_dofs = spec.act_dofs
    rt_dofs = spec.root_dofs
    nv = mjx_model.nv

    act_arr = jnp.asarray(act_dofs)
    rt_arr = jnp.asarray(rt_dofs)

    a = jnp.zeros(nv, dtype=d.qvel.dtype).at[act_arr].set(wanted_act)
    if len(rt_dofs) > 0:
        idx_rr = jnp.ix_(rt_arr, rt_arr)
        m_rr = m_full[idx_rr]
        idx_ra = jnp.ix_(rt_arr, act_arr)
        m_ra = m_full[idx_ra]
        b_root = smooth[rt_arr] - m_ra @ wanted_act
        a_root = jnp.linalg.solve(m_rr, b_root)
        a = a.at[rt_arr].set(a_root)

    tau = m_full @ a - smooth
    if len(rt_dofs) > 0:
        tau = tau.at[rt_arr].set(0.0)
    return tau


def substep(
    mjx_model: mjx.Model,
    spec: TrackingPlantSpec,
    d: mjx.Data,
    ref: tuple[jax.Array, jax.Array, jax.Array],
) -> mjx.Data:
    """Single integration substep with computed torque and contact/weld wrenches."""
    q_r, v_r, a_r = ref
    d = mjx.com_pos(mjx_model, mjx.kinematics(mjx_model, d))
    nv = mjx_model.nv
    xfrc = sphere_wrenches(mjx_model, spec, d)
    zero_qfrc = jnp.zeros(nv, dtype=d.qvel.dtype)
    d = mjx.forward(mjx_model, d.replace(xfrc_applied=xfrc, qfrc_applied=zero_qfrc))

    qpos_adr = jnp.asarray(spec.qpos_adr)
    dof_adr = jnp.asarray(spec.dof_adr)
    q = d.qpos[qpos_adr]
    v = d.qvel[dof_adr]

    zeta = spec.zeta
    omega = spec.omega_rad_s
    act_indices = jnp.asarray(spec.act_indices)

    feedback = a_r + 2.0 * zeta * omega * (v_r - v) + (omega**2) * (q_r - q)
    wanted = feedback[act_indices]
    tau = computed_torque(mjx_model, spec, d, wanted)
    return mjx.step(mjx_model, d.replace(qfrc_applied=tau))


def frame(
    mjx_model: mjx.Model,
    spec: TrackingPlantSpec,
    d: mjx.Data,
    refs: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[mjx.Data, tuple[jax.Array, jax.Array, jax.Array]]:
    """Advance simulation by one frame (substeps integrations)."""
    substeps = spec.substeps

    def body(i: jax.Array, dd: mjx.Data) -> mjx.Data:
        q_ref_i = refs[0][i]
        v_ref_i = refs[1][i]
        a_ref_i = refs[2][i]
        return substep(mjx_model, spec, dd, (q_ref_i, v_ref_i, a_ref_i))

    d = jax.lax.fori_loop(0, substeps, body, d)
    d = mjx.forward(mjx_model, d)
    m = markers(mjx_model, spec, d)
    qvel = d.qvel
    peak_vel = jnp.max(jnp.abs(qvel))
    qpos_adr = jnp.asarray(spec.qpos_adr)
    qpos = d.qpos
    q_spec = qpos[qpos_adr]
    return d, (m, peak_vel, q_spec)


def rollout(
    mjx_model: mjx.Model,
    spec: TrackingPlantSpec,
    d0: mjx.Data,
    q_sub: jax.Array,
    v_sub: jax.Array,
    a_sub: jax.Array,
) -> jax.Array:
    """Roll out trajectory and return marker positions of shape (frames, n_markers, 3)."""
    return rollout_diagnostic(mjx_model, spec, d0, q_sub, v_sub, a_sub)[0]


def rollout_diagnostic(
    mjx_model: mjx.Model,
    spec: TrackingPlantSpec,
    d0: mjx.Data,
    q_sub: jax.Array,
    v_sub: jax.Array,
    a_sub: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Roll out trajectory and return (markers, peak_qvel, qpos_spec)."""
    step_fn = jax.checkpoint(lambda dd, r: frame(mjx_model, spec, dd, r))
    _, out = jax.lax.scan(step_fn, d0, (q_sub, v_sub, a_sub))
    return out


def initial_state(
    mjx_model: mjx.Model,
    spec: TrackingPlantSpec,
    q0: jax.Array | np.ndarray,
    v0: jax.Array | np.ndarray,
    mass_kg: float,
) -> mjx.Data:
    """Construct mjx.Data at first frame with lowest sphere preloaded to static penetration."""
    opt = mjx_model.opt
    gravity = opt.gravity
    g_vec = jnp.asarray(gravity)

    ground = spec.ground
    normal = ground.normal
    n_ground = jnp.asarray(normal, dtype=g_vec.dtype)
    g_eff = -jnp.dot(g_vec, n_ground)

    d = mjx.make_data(mjx_model)
    qpos_adr = jnp.asarray(spec.qpos_adr)
    dof_adr = jnp.asarray(spec.dof_adr)
    nq = mjx_model.nq
    nv = mjx_model.nv

    q = (
        jnp.zeros(nq, dtype=g_vec.dtype)
        .at[qpos_adr]
        .set(jnp.asarray(q0, dtype=g_vec.dtype))
    )
    v = (
        jnp.zeros(nv, dtype=g_vec.dtype)
        .at[dof_adr]
        .set(jnp.asarray(v0, dtype=g_vec.dtype))
    )
    d = mjx.forward(mjx_model, d.replace(qpos=q, qvel=v))

    n_spheres = len(spec.sphere_site_ids)
    if n_spheres > 0:
        sphere_sites = jnp.asarray(spec.sphere_site_ids)
        sphere_radii = jnp.asarray(spec.sphere_radii_m, dtype=g_vec.dtype)
        ground_h = ground.height_m

        site_xpos = d.site_xpos
        sphere_pos = site_xpos[sphere_sites]
        heights = sphere_pos @ n_ground - sphere_radii - ground_h

        contact = spec.contact
        k_n = contact.stiffness_n_m
        depth = mass_kg * g_eff / (k_n * n_spheres)
        lowest = jnp.min(heights) + depth

        root_v_idx = spec.root_vertical_index
        vert_adr = spec.qpos_adr[root_v_idx]
        q = q.at[vert_adr].add(-lowest)
        d = mjx.forward(mjx_model, d.replace(qpos=q))

    return d


def build_tracking_plant(
    model: mujoco.MjModel, spec: TrackingPlantSpec
) -> TrackingPlant:
    """Wire and compile JAX/MJX differentiable tracking plant.

    The plant works on a copy of ``model`` whose timestep is set to
    ``1 / (rate_hz * substeps)``; the caller's model is not modified.
    """
    spec.validate_against_model(model)
    model = copy.deepcopy(model)
    opt = model.opt
    rate_hz = spec.rate_hz
    substeps = spec.substeps
    opt.timestep = 1.0 / (rate_hz * substeps)
    mm = mjx.put_model(model)

    jitted_rollout = jax.jit(
        lambda d0, q_s, v_s, a_s: rollout(mm, spec, d0, q_s, v_s, a_s)
    )
    jitted_diag = jax.jit(
        lambda d0, q_s, v_s, a_s: rollout_diagnostic(mm, spec, d0, q_s, v_s, a_s)
    )
    jitted_init = jax.jit(lambda q0, v0, mass: initial_state(mm, spec, q0, v0, mass))
    jitted_markers = jax.jit(lambda d: markers(mm, spec, d))

    return TrackingPlant(
        model=model,
        mjx_model=mm,
        spec=spec,
        rollout=jitted_rollout,
        rollout_diagnostic=jitted_diag,
        initial_state=jitted_init,
        markers=jitted_markers,
    )
