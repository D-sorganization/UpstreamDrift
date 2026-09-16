"""Differentiable trajectory optimisation of the tracked reference with MuJoCo
MJX (FB-5 / MM-7b, #10109).

Runs in the MJX environment (JAX plus MuJoCo >= 3.13; see HANDOFF) on the
package written by ``export_mjx_package.py``. The plant is a JAX port of the
shared replay: the run's MJCF, the Hunt-Crossley plus regularised-Coulomb
sphere contact law (same parameters, applied as world wrenches at the
calcanei), the dual-grip weld (MJX soft constraint) and the computed-torque
tracking controller with the root free. The rollout is a ``lax.scan`` over
capture frames (several integrator steps each) that emits the marker
positions, so the cost is the marker error of the *replayed* motion against
the capture, plus a small regulariser on the reference change. The decision
variables are knot values of a correction added to the actuated coordinates
of the tracked reference (linear interpolation between knots); Adam on the
gradient through the whole rollout.

    python mjx_trajectory_optimisation.py --run anthro_driver --iterations 40
    -> <run>/mjx_optimised_reference.npz (q, frames x coordinates, spec order)
       <run>/mjx_optimisation_receipt.json

``--iterations 0`` only replays the unmodified reference in MJX, which is the
port check against the shared simulator's receipt.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, NamedTuple

import xml.etree.ElementTree as ET  # serialisation only; parsing is defused

import defusedxml.ElementTree as DET
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

LOG = logging.getLogger("mjx_opt")


def load_package(run: Path) -> tuple[dict, dict, mujoco.MjModel]:
    """Package plus the model with every equality removed: MJX's constraint
    solver is an iterative loop JAX cannot reverse-differentiate, so the grip
    weld is applied here as a stiff spring-damper wrench instead."""
    meta = json.loads((run / "mjx_package.json").read_text())
    pkg = dict(np.load(run / "mjx_package.npz"))
    root = DET.fromstring((run / "mjx_package.xml").read_text())
    for equality in root.findall("equality"):
        root.remove(equality)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    # The rigid weld of the shared simulator couples the near-massless hand
    # standoff to the club; with a spring weld those dofs need an armature
    # (rotor inertia) floor or they explode. Applied to every dof.
    model.dof_armature[:] = np.maximum(model.dof_armature, ARMATURE_KG_M2)
    return meta, pkg, model


def knot_basis(times: np.ndarray, spacing_s: float) -> np.ndarray:
    """Linear-interpolation basis (frames, knots) with knots every ``spacing_s``."""
    knots = np.arange(times[0], times[-1] + spacing_s, spacing_s)
    basis = np.zeros((times.size, knots.size))
    for k, t in enumerate(times):
        j = min(int((t - knots[0]) / spacing_s), knots.size - 2)
        w = (t - knots[j]) / spacing_s
        basis[k, j] = 1.0 - w
        basis[k, j + 1] = w
    return basis


ARMATURE_KG_M2 = 5e-3
WELD_STIFFNESS_N_M = 2.0e5
WELD_DAMPING_N_S_M = 400.0
WELD_ROT_STIFFNESS_N_M_RAD = 2.0e3
WELD_ROT_DAMPING_N_M_S = 4.0


def contact_force_jax(
    centre: jnp.ndarray,
    velocity: jnp.ndarray,
    radius: float | jnp.ndarray,
    params: Any,
    ground_normal: jnp.ndarray | None = None,
    ground_height_m: float = 0.0,
) -> jnp.ndarray:
    """Evaluate the shared ground contact force (normal + friction) in JAX.

    Pure JAX formulation of Hunt-Crossley compliant normal contact and
    regularised Coulomb friction, matching
    ``src.shared.python.motion_matching.contact_law.sphere_ground_contact``.

    Parameters:
        centre: 3-vector sphere center in world coordinates (m).
        velocity: 3-vector sphere linear velocity in world coordinates (m/s).
        radius: Sphere radius (m).
        params: Contact parameters providing stiffness_n_m, dissipation_s_m,
            static_friction, dynamic_friction, viscous_friction, and
            transition_velocity_m_s (either ContactParameters instance, dict,
            or object with these attributes).
        ground_normal: Optional 3-vector unit normal (defaults to z-up [0, 0, 1]).
        ground_height_m: Optional ground plane height (defaults to 0.0).

    Returns:
        3-vector total contact force (normal + friction) in world frame (N).
    """
    c = jnp.asarray(centre)
    v = jnp.asarray(velocity)
    r = float(radius)
    if c.shape != (3,) or v.shape != (3,):
        raise ValueError("centre and velocity must be 3-vectors")
    if not (r > 0.0):
        raise ValueError("Sphere radius must be positive")

    if isinstance(params, dict):
        k_n = float(params["stiffness_n_m"])
        d_n = float(params["dissipation_s_m"])
        mu_s = float(params["static_friction"])
        mu_d = float(params["dynamic_friction"])
        mu_v = float(params["viscous_friction"])
        v_t = float(params["transition_velocity_m_s"])
    else:
        k_n = float(params.stiffness_n_m)
        d_n = float(params.dissipation_s_m)
        mu_s = float(params.static_friction)
        mu_d = float(params.dynamic_friction)
        mu_v = float(params.viscous_friction)
        v_t = float(params.transition_velocity_m_s)

    if k_n <= 0.0 or v_t <= 0.0:
        raise ValueError("Stiffness and transition velocity must be positive")
    if d_n < 0.0 or mu_d < 0.0 or mu_v < 0.0:
        raise ValueError("Dissipation and friction coefficients must be nonnegative")
    if mu_s < mu_d:
        raise ValueError("Static friction must not be below dynamic friction")

    if ground_normal is None:
        n_ground = jnp.array([0.0, 0.0, 1.0], dtype=c.dtype)
    else:
        n_ground = jnp.asarray(ground_normal, dtype=c.dtype)
        n_ground = n_ground / jnp.linalg.norm(n_ground)

    signed = c @ n_ground - ground_height_m - r
    pen = jnp.maximum(0.0, -signed)
    rate_in = -(n_ground @ v)
    magnitude = jnp.maximum(0.0, k_n * pen * (1.0 + d_n * rate_in))
    magnitude = jnp.where(pen > 0.0, magnitude, 0.0)
    tangential = v - n_ground * (n_ground @ v)
    speed = jnp.sqrt(jnp.sum(tangential**2) + 1e-12)
    ratio = speed / v_t
    mu = mu_d + (mu_s - mu_d) * jnp.exp(-(ratio**2)) + mu_v * speed
    friction = -tangential / speed * mu * magnitude * jnp.tanh(ratio)
    return n_ground * magnitude + friction


class SiteState(NamedTuple):
    """World-frame state of one weld site: position, velocity, body rotation, angular velocity, body centre of mass."""

    p: jnp.ndarray
    v: jnp.ndarray
    r: jnp.ndarray
    w: jnp.ndarray
    com: jnp.ndarray


class WeldGains(NamedTuple):
    """Gains for the weld spring-damper reaction."""

    k: float = WELD_STIFFNESS_N_M
    c: float = WELD_DAMPING_N_S_M
    rot_k: float = WELD_ROT_STIFFNESS_N_M_RAD
    rot_c: float = WELD_ROT_DAMPING_N_M_S


def weld_wrench_jax(
    a: SiteState,
    b: SiteState,
    gains: WeldGains = WeldGains(),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Equal and opposite spatial wrenches on body a and body b.

    Computes the spring-damper reaction holding closure site b on site a.
    Returns (wrench_a, wrench_b) where each is a 6-vector (force, torque at CoM).
    """
    rel = a.r @ b.r.T  # rotation taking b's axes onto a's
    rotvec = 0.5 * jnp.array(
        [rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]
    )
    force_on_b = gains.k * (a.p - b.p) + gains.c * (a.v - b.v)
    torque_on_b = gains.rot_k * rotvec + gains.rot_c * (a.w - b.w)
    wrench_b = jnp.concatenate(
        [force_on_b, torque_on_b + jnp.cross(b.p - b.com, force_on_b)]
    )
    wrench_a = jnp.concatenate(
        [-force_on_b, -torque_on_b + jnp.cross(a.p - a.com, -force_on_b)]
    )
    return wrench_a, wrench_b


def compute_knot_mask(
    knot_times: np.ndarray | jnp.ndarray,
    horizon: float,
) -> jnp.ndarray:
    """Mask of active knots within the cost horizon (knots > horizon are 0)."""
    if horizon <= 0.0:
        raise ValueError("Horizon must be positive")
    kt = jnp.asarray(knot_times)
    return jnp.asarray((kt <= horizon).astype(jnp.float32))


def build_reference(
    q_track: jnp.ndarray | np.ndarray,
    act_indices: jnp.ndarray | np.ndarray,
    basis: jnp.ndarray | np.ndarray,
    delta: jnp.ndarray,
    knot_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Tracked reference plus the knot correction; masked knots are frozen."""
    q = jnp.asarray(q_track)
    act = jnp.asarray(act_indices)
    b = jnp.asarray(basis)
    d = delta if knot_mask is None else delta * knot_mask[:, None]
    return q.at[:, act].add(b @ d)


def build(
    meta: dict,
    pkg: dict,
    model: mujoco.MjModel,
    substeps: int,
    weld_k: float = WELD_STIFFNESS_N_M,
    weld_c: float = WELD_DAMPING_N_S_M,
) -> dict:
    """Constants and the jitted rollout for this package."""
    rate = float(meta["rate_hz"])
    model.opt.timestep = 1.0 / (rate * substeps)
    mm = mjx.put_model(model)
    contact = meta["contact"]
    n_ground = jnp.asarray(pkg["ground_normal"] / np.linalg.norm(pkg["ground_normal"]))
    ground_h = float(pkg["ground_height_m"])
    qpos_adr = jnp.asarray(pkg["qpos_adr"])
    dof_adr = jnp.asarray(pkg["dof_adr"])
    root_mask = jnp.asarray(pkg["root_mask"])
    act = jnp.where(~jnp.asarray(pkg["root_mask"]))[0]  # spec indices
    rt = jnp.where(jnp.asarray(pkg["root_mask"]))[0]
    act_dof = jnp.asarray(pkg["dof_adr"])[act]  # MuJoCo dof indices
    rt_dof = jnp.asarray(pkg["dof_adr"])[rt]
    sphere_sites = jnp.asarray(pkg["sphere_site_ids"])
    sphere_bodies = jnp.asarray(pkg["sphere_body_ids"])
    sphere_radii = jnp.asarray(pkg["sphere_radii_m"])
    marker_bodies = jnp.asarray(pkg["marker_body_ids"])
    marker_local = jnp.asarray(pkg["marker_local_m"])
    omega = float(meta["controller"]["omega_rad_s"])
    zeta = float(meta["controller"]["zeta"])
    k_n = float(contact["stiffness_n_m"])
    d_n = float(contact["dissipation_s_m"])
    mu_s = float(contact["static_friction"])
    mu_d = float(contact["dynamic_friction"])
    mu_v = float(contact["viscous_friction"])
    v_t = float(contact["transition_velocity_m_s"])
    nbody = model.nbody
    nv = model.nv
    site_a = int(meta["closure"]["site_a"])
    site_b = int(meta["closure"]["site_b"])
    body_a = int(model.site_bodyid[site_a])
    body_b = int(model.site_bodyid[site_b])

    weld_gains = WeldGains(
        k=weld_k,
        c=weld_c,
        rot_k=WELD_ROT_STIFFNESS_N_M_RAD,
        rot_c=WELD_ROT_DAMPING_N_M_S,
    )

    def weld_wrenches(d: mjx.Data, xfrc: jnp.ndarray) -> jnp.ndarray:
        """Stiff spring-damper holding closure site b on site a (the dual-grip
        weld the shared simulator solves rigidly)."""
        p_a, p_b = d.site_xpos[site_a], d.site_xpos[site_b]
        r_a, r_b = d.site_xmat[site_a], d.site_xmat[site_b]
        jp_a, jr_a = mjx.jac(mm, d, p_a, body_a)
        jp_b, jr_b = mjx.jac(mm, d, p_b, body_b)
        v_a, v_b = jp_a.T @ d.qvel, jp_b.T @ d.qvel
        w_a, w_b = jr_a.T @ d.qvel, jr_b.T @ d.qvel
        wa, wb = weld_wrench_jax(
            SiteState(p=p_a, v=v_a, r=r_a, w=w_a, com=d.xipos[body_a]),
            SiteState(p=p_b, v=v_b, r=r_b, w=w_b, com=d.xipos[body_b]),
            gains=weld_gains,
        )
        return xfrc.at[body_b].add(wb).at[body_a].add(wa)

    def sphere_wrenches(d: mjx.Data) -> jnp.ndarray:
        """World wrenches (nbody, 6) of the shared contact law at state ``d``."""
        xfrc = jnp.zeros((nbody, 6))
        static = list(
            zip(
                pkg["sphere_site_ids"],
                pkg["sphere_body_ids"],
                pkg["sphere_radii_m"],
                strict=True,
            )
        )

        def one(site: int, body: int, radius: float, acc: jnp.ndarray) -> jnp.ndarray:
            centre = d.site_xpos[site]
            jacp, _ = mjx.jac(mm, d, centre, body)
            velocity = jacp.T @ d.qvel
            force = contact_force_jax(
                centre, velocity, radius, contact, n_ground, ground_h
            )
            point = centre - n_ground * radius
            torque = jnp.cross(point - d.xipos[body], force)
            return acc.at[body].add(jnp.concatenate([force, torque]))

        for site, body, radius in static:
            xfrc = one(int(site), int(body), float(radius), xfrc)
        return weld_wrenches(d, xfrc)

    def computed_torque(d: mjx.Data, wanted_act: jnp.ndarray) -> jnp.ndarray:
        """Generalised applied force giving the actuated dofs ``wanted_act``
        with the root free (MuJoCo dof order)."""
        m_full = mjx.full_m(mm, d)
        smooth = d.qfrc_smooth + d.qfrc_constraint  # everything but qfrc_applied
        a = jnp.zeros(nv).at[act_dof].set(wanted_act)
        m_rr = m_full[jnp.ix_(rt_dof, rt_dof)]
        m_ra = m_full[jnp.ix_(rt_dof, act_dof)]
        a_root = jnp.linalg.solve(m_rr, smooth[rt_dof] - m_ra @ wanted_act)
        a = a.at[rt_dof].set(a_root)
        tau = m_full @ a - smooth
        return tau.at[rt_dof].set(0.0)

    def markers(d: mjx.Data) -> jnp.ndarray:
        rot = d.xmat[marker_bodies]
        return d.xpos[marker_bodies] + jnp.einsum("mij,mj->mi", rot, marker_local)

    def substep(
        d: mjx.Data, ref: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> mjx.Data:
        q_r, v_r, a_r = ref
        d = mjx.com_pos(mm, mjx.kinematics(mm, d))  # poses for the wrenches
        # qfrc_smooth includes qfrc_applied, so the previous torque must be
        # cleared before the forward pass the computed torque reads.
        d = mjx.forward(
            mm, d.replace(xfrc_applied=sphere_wrenches(d), qfrc_applied=jnp.zeros(nv))
        )
        q = d.qpos[qpos_adr]
        v = d.qvel[dof_adr]
        wanted = (a_r + 2.0 * zeta * omega * (v_r - v) + omega**2 * (q_r - q))[act]
        tau_spec = computed_torque(d, wanted)  # dof order already (scalar joints)
        return mjx.step(mm, d.replace(qfrc_applied=tau_spec))

    def frame(d: mjx.Data, refs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
        def body(i: jnp.ndarray, dd: mjx.Data) -> mjx.Data:
            return substep(dd, (refs[0][i], refs[1][i], refs[2][i]))

        d = jax.lax.fori_loop(0, substeps, body, d)
        d = mjx.forward(mm, d)
        return d, (markers(d), jnp.max(jnp.abs(d.qvel)), d.qpos[qpos_adr])

    frame = jax.checkpoint(frame)

    def rollout(
        d0: mjx.Data, q_sub: jnp.ndarray, v_sub: jnp.ndarray, a_sub: jnp.ndarray
    ):
        """``q_sub`` (frames, substeps, coords) in spec order; returns markers (frames, m, 3)."""
        _, out = jax.lax.scan(frame, d0, (q_sub, v_sub, a_sub))
        return out[0]

    def rollout_diagnostic(
        d0: mjx.Data, q_sub: jnp.ndarray, v_sub: jnp.ndarray, a_sub: jnp.ndarray
    ):
        _, out = jax.lax.scan(frame, d0, (q_sub, v_sub, a_sub))
        return out

    return {
        "mm": mm,
        "model": model,
        "rollout": jax.jit(rollout),
        "rollout_diagnostic": jax.jit(rollout_diagnostic),
        "qpos_adr": qpos_adr,
        "dof_adr": dof_adr,
        "root_mask": root_mask,
        "act": act,
        "sphere_sites": sphere_sites,
        "sphere_radii": sphere_radii,
        "n_ground": n_ground,
        "ground_h": ground_h,
        "k_n": k_n,
        "substeps": substeps,
        "markers": markers,
    }


def initial_state(
    sim: dict, q0_spec: np.ndarray, v0_spec: np.ndarray, mass: float, spheres: int
) -> mjx.Data:
    """Data at the first frame with the feet preloaded (lowest sphere at the
    static penetration of an evenly shared weight)."""
    mm = sim["mm"]
    d = mjx.make_data(mm)
    q = jnp.zeros(mm.nq).at[sim["qpos_adr"]].set(jnp.asarray(q0_spec))
    v = jnp.zeros(mm.nv).at[sim["dof_adr"]].set(jnp.asarray(v0_spec))
    d = mjx.forward(mm, d.replace(qpos=q, qvel=v))
    heights = (
        d.site_xpos[sim["sphere_sites"]] @ sim["n_ground"]
        - sim["sphere_radii"]
        - sim["ground_h"]
    )
    depth = mass * 9.81 / (sim["k_n"] * spheres)
    lowest = jnp.min(heights) + depth
    # root vertical slide is the third spec coordinate (world z for these documents)
    q = q.at[sim["qpos_adr"][2]].add(-lowest)
    return mjx.forward(mm, d.replace(qpos=q))


def main() -> None:
    jax.config.update("jax_enable_x64", False)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--substeps", type=int, default=6)
    parser.add_argument("--knot-spacing", type=float, default=0.04)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--regularisation", type=float, default=1e-3)
    parser.add_argument("--name", default="mjx_optimised_reference")
    parser.add_argument("--weld-stiffness", type=float, default=WELD_STIFFNESS_N_M)
    parser.add_argument("--weld-damping", type=float, default=WELD_DAMPING_N_S_M)
    parser.add_argument(
        "--horizon",
        type=float,
        default=1.65,
        help="frames beyond this time (s) are left out of the cost: the soft-weld "
        "plant departs from the rigid-weld one in the last follow-through",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="also save the reference every N iterations as <name>_iterN.npz so "
        "the shared-law plant can select the iteration that transfers best",
    )
    parser.add_argument(
        "--init",
        type=Path,
        default=None,
        help="warm-start the knot correction from a previous optimised reference npz",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="forward rollout only: first non-finite frame, peak joint speed, replay error",
    )
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    run = args.run if args.run.is_absolute() else here / args.run
    meta, pkg, model = load_package(run)
    sim = build(meta, pkg, model, args.substeps, args.weld_stiffness, args.weld_damping)
    times = pkg["time_s"]
    q_track = pkg["q_track"]
    # Invalid markers are NaN in the capture; a masked `where` still leaks a
    # NaN gradient from the untaken branch, so they are zeroed here.
    targets = jnp.asarray(np.nan_to_num(pkg["targets_m"]))
    in_horizon = (times <= args.horizon)[:, None]
    valid = jnp.asarray(
        pkg["valid"] & np.isfinite(pkg["targets_m"]).all(axis=2) & in_horizon
    )
    count = float(valid.sum())
    basis = jnp.asarray(knot_basis(times, args.knot_spacing))
    act = np.where(~pkg["root_mask"])[0]
    n_knots = basis.shape[1]
    spheres = int(pkg["sphere_site_ids"].size)
    mass = float(meta["mass_kg"])

    knot_times = np.arange(times[0], times[-1] + args.knot_spacing, args.knot_spacing)
    knot_mask = compute_knot_mask(knot_times[:n_knots], args.horizon)

    def reference(delta: jnp.ndarray) -> jnp.ndarray:
        """Tracked reference plus the knot correction; knots beyond the cost
        horizon are frozen so the uncosted tail keeps the original reference."""
        return build_reference(q_track, act, basis, delta, knot_mask)

    def replay(delta: jnp.ndarray) -> jnp.ndarray:
        q_np = reference(delta)
        v_np = jnp.gradient(q_np, axis=0) / float(np.mean(np.diff(times)))
        a_np = jnp.gradient(v_np, axis=0) / float(np.mean(np.diff(times)))
        fractions = jnp.arange(args.substeps) / args.substeps

        def table(x: jnp.ndarray) -> jnp.ndarray:
            nxt = jnp.vstack([x[1:], x[-1:]])
            return x[:, None, :] + fractions[None, :, None] * (nxt - x)[:, None, :]

        d0 = initial_state(sim, q_np[0], v_np[0], mass, spheres)
        return sim["rollout"](d0, table(q_np), table(v_np), table(a_np))

    def cost(delta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        m = replay(delta)
        err2 = jnp.sum((m - targets) ** 2, axis=2)
        marker_cost = jnp.sum(jnp.where(valid, err2, 0.0)) / count
        reg = args.regularisation * jnp.mean(delta**2)
        return marker_cost + reg, marker_cost

    if args.diagnose:
        q_np = reference(jnp.zeros((n_knots, act.size)))
        dt = float(np.mean(np.diff(times)))
        v_np = jnp.gradient(q_np, axis=0) / dt
        a_np = jnp.gradient(v_np, axis=0) / dt
        fractions = jnp.arange(args.substeps) / args.substeps

        def table(x: jnp.ndarray) -> jnp.ndarray:
            nxt = jnp.vstack([x[1:], x[-1:]])
            return x[:, None, :] + fractions[None, :, None] * (nxt - x)[:, None, :]

        d0 = initial_state(sim, q_np[0], v_np[0], mass, spheres)
        t0 = time.perf_counter()
        m, speed, q_sim = sim["rollout_diagnostic"](
            d0, table(q_np), table(v_np), table(a_np)
        )
        m, speed, q_sim = np.asarray(m), np.asarray(speed), np.asarray(q_sim)
        finite = np.isfinite(m).all(axis=(1, 2))
        first_bad = int(np.argmin(finite)) if not finite.all() else -1
        err = np.sqrt(np.sum((m - np.asarray(targets)) ** 2, axis=2))
        ok = np.asarray(valid) & finite[:, None]
        rms = float(np.sqrt(np.mean(err[ok] ** 2))) if ok.any() else float("nan")
        root_err = np.linalg.norm(q_sim[:, :3] - q_track[:, :3], axis=1)
        np.savez(run / "mjx_diagnose.npz", markers_m=m, q_sim=q_sim, peak_qvel=speed)
        LOG.info(
            "diagnose: rollout %.0f s; first non-finite frame %d (t=%.3f s); replay markers over finite frames %.1f mm",
            time.perf_counter() - t0,
            first_bad,
            times[first_bad] if first_bad >= 0 else float("nan"),
            rms * 1e3,
        )
        for k in range(0, len(times), 36):
            LOG.info(
                "  t=%.2f: peak |qvel| %.1f rad/s, root err %.0f mm, marker rms %.0f mm",
                times[k],
                speed[k],
                root_err[k] * 1e3,
                float(np.sqrt(np.mean(err[k][np.asarray(valid)[k]] ** 2))) * 1e3
                if finite[k]
                else float("nan"),
            )
        return

    value_and_grad = jax.jit(jax.value_and_grad(cost, has_aux=True))
    delta = jnp.zeros((n_knots, act.size))
    if args.init is not None:
        previous = np.load(args.init)
        if float(previous["knot_spacing_s"]) != args.knot_spacing:
            raise ValueError("--init was optimised with a different knot spacing")
        delta = jnp.asarray(previous["delta_knots"])
    t0 = time.perf_counter()
    (total, marker_cost), grad = value_and_grad(delta)
    LOG.info(
        "port check: MJX replay of the unmodified reference %.1f mm (shared simulator %.1f mm); first gradient in %.0f s",
        float(jnp.sqrt(marker_cost)) * 1e3, meta["baseline"]["replay_marker_rms_m"] * 1e3, time.perf_counter() - t0,
    )  # fmt: skip
    history = [
        {
            "iteration": 0,
            "replay_marker_rms_m": float(jnp.sqrt(marker_cost)),
            "total_cost": float(total),
        }
    ]
    best = (float(marker_cost), delta)
    # Adam
    m1 = jnp.zeros_like(delta)
    m2 = jnp.zeros_like(delta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(1, args.iterations + 1):
        m1 = b1 * m1 + (1 - b1) * grad
        m2 = b2 * m2 + (1 - b2) * grad**2
        step = (
            args.learning_rate * (m1 / (1 - b1**k)) / (jnp.sqrt(m2 / (1 - b2**k)) + eps)
        )
        delta = delta - step
        (total, marker_cost), grad = value_and_grad(delta)
        rms = float(jnp.sqrt(marker_cost))
        finite = bool(jnp.isfinite(total))
        history.append(
            {
                "iteration": k,
                "replay_marker_rms_m": rms,
                "total_cost": float(total),
                "delta_max_rad": float(jnp.abs(delta).max()),
            }
        )
        if args.snapshot_every and k % args.snapshot_every == 0:
            np.savez(
                run / f"{args.name}_iter{k}.npz",
                q=np.asarray(reference(delta)),
                delta_knots=np.asarray(delta),
                knot_spacing_s=args.knot_spacing,
            )
        if not bool(jnp.isfinite(grad).all()):
            LOG.info("gradient not finite at iteration %d; stopping", k)
            break
        LOG.info(
            "iteration %d: replay markers %.1f mm, delta max %.2f deg, %s",
            k,
            rms * 1e3,
            float(jnp.degrees(jnp.abs(delta).max())),
            "ok" if finite else "NON-FINITE",
        )
        if not finite:
            break
        if float(marker_cost) < best[0]:
            best = (float(marker_cost), delta)
            np.savez(  # every improvement is saved so a stopped run keeps its best
                run / f"{args.name}.npz",
                q=np.asarray(reference(delta)),
                delta_knots=np.asarray(delta),
                knot_spacing_s=args.knot_spacing,
            )
    q_best = np.asarray(reference(best[1]))
    np.savez(
        run / f"{args.name}.npz",
        q=q_best,
        delta_knots=np.asarray(best[1]),
        knot_spacing_s=args.knot_spacing,
    )
    receipt = {
        "run": run.name,
        "settings": vars(args) | {"run": str(run)},
        "knots": int(n_knots),
        "actuated_coordinates": int(act.size),
        "port_check_replay_marker_rms_m": history[0]["replay_marker_rms_m"],
        "shared_simulator_replay_marker_rms_m": meta["baseline"]["replay_marker_rms_m"],
        "best_replay_marker_rms_m": float(np.sqrt(best[0])),
        "history": history,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    (run / "mjx_optimisation_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    LOG.info(
        "best MJX replay %.1f mm after %d iterations; wrote %s",
        float(np.sqrt(best[0])) * 1e3,
        args.iterations,
        run / f"{args.name}.npz",
    )


if __name__ == "__main__":
    main()
