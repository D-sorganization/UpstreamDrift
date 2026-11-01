"""MuJoCo contact grip integration tests with ContactTopology (#13306, HRG-GS-G8).

Acceptance criteria:
- Static check: native contact force per pair matches the shared law to <= 5 %
  at matched penetration, at two penetrations (1x preload and 2x preload),
  for each style (overlap, interlock, ten_finger).
- Dynamic parity versus the Pinocchio topology reference via grip_contact.parity,
  within PEAK_TOLERANCE (5 %) and RMS_TOLERANCE (2 %) for overlap, interlock,
  and ten_finger (hold plus a short swing).
- No spurious contact: in the generated model, no active contact involves two
  hand bodies or a hand and its welded parent over all steps.
- Existing MuJoCo grip tests stay green, and the old pad path's MJCF is
  byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.grip_contact import (
    ClubDynamics,
    CoordinateSwing,
    GripInterface,
    load_coordinate_swing,
)
from src.shared.python.grip_contact.contact_topology import (
    ContactSite,
    ContactTopology,
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.pad_contact import build_pad_model, grip_axis
from src.shared.python.grip_contact.net_wrench_parity import (
    failing_net_wrench,
    net_club_wrench_errors,
)
from src.shared.python.grip_contact.pair_contact import (
    ContactCapsule,
    capsule_line_nodes,
)
from src.shared.python.grip_contact.parity import (
    PEAK_TOLERANCE,
    RMS_TOLERANCE,
    GripKineticsSeries,
    QuantityError,
    parity_errors,
)
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    build_topology_model,
    topology_pair_wrench,
    topology_side_wrench,
)
from src.shared.python.motion_matching.contact_kernel import friction_secant_gain
from src.shared.python.motion_matching.grip_contract import GripSpec

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
SPEC_PATH = MODELS / "full_body_spec_anthro_driver.json"
SQUEEZE = 1100.0
STATIC_FORCE_TOL = 0.05


def _mujoco_sim_module() -> Any:
    try:
        import mujoco  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"mujoco not installed: {exc}")
    from src.engines.physics_engines.mujoco.python import grip_contact_sim

    return grip_contact_sim


def _pinocchio_sim_module() -> Any:
    try:
        import pinocchio  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"pinocchio not installed: {exc}")
    from src.engines.physics_engines.pinocchio.python import grip_contact_sim

    return grip_contact_sim


def _scale_topology_penetration(
    topo: ContactTopology, preload_scale: float
) -> ContactTopology:
    """Shift hand-grip sites radially so matched penetration is preload_scale * delta_0."""
    if abs(preload_scale - 1.0) < 1e-12:
        return topo
    new_sites: list[ContactSite] = []
    for s in topo.sites:
        cx, cy, cz = s.centre_m
        radial = math.hypot(cy, cz)
        penetration_0 = (topo.grip_radius_m + s.radius_m) - radial
        if radial > 1e-12 and penetration_0 > 0.0:
            new_radial = (topo.grip_radius_m + s.radius_m) - (
                preload_scale * penetration_0
            )
            factor = new_radial / radial
            s = replace(s, centre_m=(cx, cy * factor, cz * factor))
        new_sites.append(s)
    return replace(topo, sites=tuple(new_sites))


def _setup(
    style: GripStyle = GripStyle.OVERLAP,
    trigger_finger: bool = False,
    preload_scale: float = 1.0,
) -> tuple[bytes, dict[str, Any], list[str], Any, GripInterface, TopologyContactModel]:
    spec_bytes = SPEC_PATH.read_bytes()
    spec = json.loads(spec_bytes)
    names = spec["coordinate_order"]
    swing = load_coordinate_swing(
        FIXTURES / "swing_q_driver.npz",
        FIXTURES / "address_poses.json",
        "driver",
        names,
    )
    interface = GripInterface.from_spec(spec)
    gspec = GripSpec(style=style, trigger_finger=trigger_finger)
    topo = _scale_topology_penetration(contact_topology(gspec), preload_scale)
    topo_model = build_topology_model(interface, topo, squeeze_per_hand_n=SQUEEZE)
    return spec_bytes, spec, names, swing, interface, topo_model


def _reference_old_pad_mjcf(
    club: ClubDynamics,
    gravity: np.ndarray,
    timestep_s: float,
    interface: GripInterface,
    pads: Any,
    solref: dict[tuple[str, int], tuple[float, float]],
    friction_time_s: float | None = None,
    sides: tuple[str, ...] = ("L", "R"),
) -> str:
    """Reference generator for the legacy PadLayout MJCF string."""
    if friction_time_s is None:
        friction_time_s = 2.0 * timestep_s
    i = club.inertia_com_kg_m2
    full = (i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2])
    axis, axis_point = grip_axis(interface, pads)
    lo, hi = pads.cylinder.axial_range_m
    a0, a1 = axis_point + lo * axis, axis_point + hi * axis
    mu = pads.law.static_friction
    torsional = mu * pads.patch_radius_m

    def _nums(values: Any) -> str:
        return " ".join(format(float(v), ".17g") for v in values)

    bodies, pairs = [], []
    for side in sides:
        geoms = []
        for k, pos in enumerate(pads.layout.positions_grip_frame(side)):
            geoms.append(
                f'<geom name="pad_{side}{k}" type="sphere" size="{pads.layout.pad_radius_m:.17g}" '
                f'pos="{_nums(pos)}" contype="0" conaffinity="0" mass="0"/>'
            )
            ref = solref[(side, k)]
            pairs.append(
                f'<pair geom1="pad_{side}{k}" geom2="grip" condim="6" '
                f'friction="{mu:.17g} {mu:.17g} {torsional:.17g} 1e-5 1e-5" '
                f'solref="{ref[0]:.17g} {ref[1]:.17g}" '
                f'solreffriction="{friction_time_s:.17g} 1" '
                f'solimp="0.95 0.95 0.001 0.5 2"/>'
            )
        bodies.append(
            f'<body name="hand_{side}"><freejoint name="hand_{side}_free"/>'
            f'<inertial pos="0 0 0" mass="100.0" diaginertia="10 10 10"/>'
            + "".join(geoms)
            + "</body>"
        )
    return (
        '<mujoco model="grip_contact"><compiler inertiafromgeom="false"/>'
        f'<option timestep="{timestep_s:.17g}" integrator="Euler" '
        f'cone="elliptic" impratio="10" gravity="{_nums(gravity)}"/>'
        '<worldbody><body name="club"><freejoint name="club_free"/>'
        f'<inertial pos="{_nums(club.com_m)}" mass="{club.mass_kg:.17g}" '
        f'fullinertia="{_nums(full)}"/>'
        f'<geom name="grip" type="cylinder" fromto="{_nums(a0)} {_nums(a1)}" '
        f'size="{pads.layout.grip_radius_m:.17g}" contype="0" conaffinity="0" mass="0"/>'
        "</body>" + "".join(bodies) + "</worldbody>"
        "<contact>" + "".join(pairs) + "</contact></mujoco>"
    )


def test_old_pad_path_mjcf_is_byte_identical() -> None:
    """Old pad path's MJCF is byte-identical before and after calibration."""
    module = _mujoco_sim_module()
    spec_bytes = SPEC_PATH.read_bytes()
    spec = json.loads(spec_bytes)
    names = spec["coordinate_order"]
    swing = load_coordinate_swing(
        FIXTURES / "swing_q_driver.npz",
        FIXTURES / "address_poses.json",
        "driver",
        names,
    )
    interface = GripInterface.from_spec(spec)
    pads = build_pad_model(interface, SQUEEZE)
    sim = module.ClubInHands(spec_bytes, names, interface, pads)

    expected_pre = _reference_old_pad_mjcf(
        sim.club,
        sim.gravity,
        sim.timestep_s,
        sim.interface,
        sim.pads,
        sim.solref,
        sim.friction_time_s,
        sim.pad_sides,
    )
    actual_pre = module.contact_mjcf(
        sim.club,
        sim.gravity,
        sim.timestep_s,
        sim.interface,
        sim.pads,
        sim.solref,
        sim.friction_time_s,
        sim.pad_sides,
    )
    assert actual_pre == expected_pre
    # One digest for every platform and BLAS kernel (#13720): the damping design
    # and pad ring trig use platform-independent arithmetic.
    assert (
        hashlib.sha256(actual_pre.encode("utf-8")).hexdigest()
        == "48d14dd9138eee8319f74293d8235e8be5e586accb8bd78d35faa64fca597d48"
    )

    q0 = np.asarray(swing.q[0], float)
    sim.calibrate(q0)
    expected_post = _reference_old_pad_mjcf(
        sim.club,
        sim.gravity,
        sim.timestep_s,
        sim.interface,
        sim.pads,
        sim.solref,
        sim.friction_time_s,
        sim.pad_sides,
    )
    actual_post = module.contact_mjcf(
        sim.club,
        sim.gravity,
        sim.timestep_s,
        sim.interface,
        sim.pads,
        sim.solref,
        sim.friction_time_s,
        sim.pad_sides,
    )
    assert actual_post == expected_post
    assert (
        hashlib.sha256(actual_post.encode("utf-8")).hexdigest()
        == hashlib.sha256(expected_post.encode("utf-8")).hexdigest()
    )


@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_topology_mjcf_primitives_and_exclusions(style: GripStyle) -> None:
    """Sphere geoms per physical site (capsules as line-contact nodes), explicit pairs."""
    module = _mujoco_sim_module()
    spec_bytes, _, names, _, interface, topo_model = _setup(style=style)
    sim = module.ClubInHands(
        spec_bytes, names, interface, pads=None, topology_model=topo_model
    )
    m = sim.model
    mj = sim._mj

    phys_pairs = topo_model.topology.hand_grip_pairs()
    grip_geom_id = int(m.geom("grip").id)
    hand_body_ids = {int(m.body("hand_L").id), int(m.body("hand_R").id)}

    xml = module.contact_mjcf(
        sim.club,
        sim.gravity,
        sim.timestep_s,
        sim.interface,
        None,
        sim.solref,
        sim.friction_time_s,
        topology_model=topo_model,
    )
    assert '<exclude body1="hand_L" body2="hand_R"/>' in xml

    expected_pairs = 0
    for p in phys_pairs:
        site = topo_model.topology.site(p.site_a)
        if site.primitive == "sphere":
            geom_names = [site.name]
        else:  # capsule: shared axial line-contact nodes (#13308)
            p0, p1 = site.endpoints_m
            count = len(
                capsule_line_nodes(
                    ContactCapsule(np.asarray(p0), np.asarray(p1), site.radius_m)
                )
            )
            geom_names = [
                module.line_node_geom_name(site.name, j) for j in range(count)
            ]
            assert f'name="{site.name}"' not in xml
        expected_pairs += len(geom_names)
        expected_body = (
            int(m.body("hand_L").id)
            if site.hand == "lead"
            else int(m.body("hand_R").id)
        )
        for name in geom_names:
            assert f'<pair geom1="{name}" geom2="grip" condim="6"' in xml
            geom_id = int(m.geom(name).id)
            assert int(m.geom_contype[geom_id]) == 0
            assert int(m.geom_conaffinity[geom_id]) == 0
            assert int(m.geom_bodyid[geom_id]) == expected_body
            assert int(m.geom_type[geom_id]) == int(mj.mjtGeom.mjGEOM_SPHERE)
    assert m.npair == expected_pairs

    # Every explicit pair must link a physical hand site and the grip cylinder
    for pair_idx in range(m.npair):
        g1 = int(m.pair_geom1[pair_idx])
        g2 = int(m.pair_geom2[pair_idx])
        b1 = int(m.geom_bodyid[g1])
        b2 = int(m.geom_bodyid[g2])
        assert grip_geom_id in (g1, g2)
        other_body = b2 if g1 == grip_geom_id else b1
        assert other_body in hand_body_ids
        assert int(m.pair_dim[pair_idx]) == 6


@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_static_pair_force_matches_shared_law_at_1x_and_2x_preload(
    style: GripStyle,
) -> None:
    """Static check: native pair force (mj_contactForce rotated to world) matches shared law to <= 5 % at 1x and 2x preload."""
    module = _mujoco_sim_module()
    spec_bytes, _, names, swing, interface, topo_model_1x = _setup(
        style=style, preload_scale=1.0
    )
    q0 = np.asarray(swing.q[0], float)

    sim = module.ClubInHands(
        spec_bytes, names, interface, pads=None, topology_model=topo_model_1x
    )
    cal_err = sim.calibrate(q0)
    assert cal_err <= STATIC_FORCE_TOL
    calibrated_solref = dict(sim.solref)

    for preload_scale in (1.0, 2.0):
        _, _, _, _, _, topo_model = _setup(style=style, preload_scale=preload_scale)
        probe_sim = module.ClubInHands(
            spec_bytes, names, interface, pads=None, topology_model=topo_model
        )
        # Reuse the solref calibrated at 1x preload to verify both 1x and 2x penetration
        probe_sim.solref = dict(calibrated_solref)
        probe_sim._rebuild_keep_state()

        hands = probe_sim.hold_pose(q0)
        club = probe_sim.club_state_now()
        frames = {s: club.frame(probe_sim._offsets[s]) for s in ("L", "R")}

        # Run native mj_forward and read mj_contactForce rotated to world per pair
        mj, m, d = probe_sim._mj, probe_sim.model, probe_sim.data
        mj.mj_forward(m, d)
        res = np.zeros(6, dtype=float)
        native_force: dict[tuple[str, int], float] = {}
        native_world_vec: dict[tuple[str, int], np.ndarray] = {}
        native_depth: dict[tuple[str, int], float] = {}
        for i in range(d.ncon):
            c = d.contact[i]
            hit = probe_sim._pad_geom.get(int(c.geom1))
            if hit is None:
                continue
            mj.mj_contactForce(m, d, i, res)
            frame = np.asarray(c.frame, dtype=float).reshape(3, 3)
            native_force[hit] = native_force.get(hit, 0.0) + float(res[0])
            native_world_vec[hit] = (
                native_world_vec.get(hit, np.zeros(3, dtype=float)) + frame.T @ res[:3]
            )
            native_depth[hit] = max(native_depth.get(hit, 0.0), float(-c.dist))

        max_rel_err = 0.0
        for side in ("L", "R"):
            sw = topology_side_wrench(topo_model, side, hands[side], frames[side])
            for k, pair in enumerate(topo_model.physical_grip_pairs(side)):
                key = (side, k)
                assert key in native_force, f"missing contact for {pair.name}"
                pw = topology_pair_wrench(topo_model, pair, hands[side], frames[side])
                assert pw.normal_force_mag_n == pytest.approx(
                    sw.pair_wrenches[pair.name].normal_force_mag_n, rel=1e-12
                )
                f_mj = native_force[key]
                f_law = pw.normal_force_mag_n
                d_mj = native_depth[key]
                d_law = pw.penetration_m
                assert d_mj == pytest.approx(d_law, rel=1e-3, abs=1e-6)
                rel_err = abs(f_mj - f_law) / f_law
                vec_err = float(
                    np.linalg.norm(native_world_vec[key] - pw.force_b_n)
                ) / float(np.linalg.norm(pw.force_b_n))
                max_rel_err = max(max_rel_err, rel_err)
                assert rel_err <= STATIC_FORCE_TOL, (
                    f"{style.value} {preload_scale}x preload pair {pair.name}: "
                    f"f_mj={f_mj:.4f} N vs f_law={f_law:.4f} N (rel_err={rel_err:.4%})"
                )
                assert vec_err <= STATIC_FORCE_TOL, (
                    f"{style.value} {preload_scale}x preload pair {pair.name}: "
                    f"world force vector err={vec_err:.4%}"
                )
        # Ensure we are measuring native unconstrained mj_contactForce, not the analytic law itself
        assert max_rel_err > 1e-6


@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_no_spurious_contacts_and_pair_trace_recorder_fast(style: GripStyle) -> None:
    """Fast 0.1 s check: no hand-hand or parent-child contacts; PairTraceRecorder populated."""
    module = _mujoco_sim_module()
    spec_bytes, _, names, swing, interface, topo_model = _setup(style=style)
    q0 = np.asarray(swing.q[0], float)

    sim = module.ClubInHands(
        spec_bytes, names, interface, pads=None, topology_model=topo_model
    )
    sim.calibrate(q0)
    m = sim.model
    d = sim.data
    hand_bodies = {int(m.body("hand_L").id), int(m.body("hand_R").id)}
    club_body = int(m.body("club").id)

    sim.hold_pose(q0)
    steps_per_sample = int(round(0.002 / sim.timestep_s))
    samples = [sim.sample()]
    probe_sim_steps = 10  # 10 samples of 2 ms (2,000 steps) to check every step
    for _ in range(probe_sim_steps):
        probe_sim_step = True
        if probe_sim_step:
            sim._set_step_scaling(True)
        for _ in range(steps_per_sample):
            t = float(d.time)
            sim.place_hands(t, velocity_time=t + 0.5 * sim.timestep_s)
            sim._mj.mj_step(m, d)
            for i in range(d.ncon):
                c = d.contact[i]
                b1 = int(m.geom_bodyid[int(c.geom1)])
                b2 = int(m.geom_bodyid[int(c.geom2)])
                assert not (b1 in hand_bodies and b2 in hand_bodies), (
                    f"spurious hand-hand contact between bodies {b1} and {b2}"
                )
                assert (
                    int(m.body_parentid[b1]) != b2 and int(m.body_parentid[b2]) != b1
                ), f"spurious parent-child contact between bodies {b1} and {b2}"
                assert {b1, b2} <= (hand_bodies | {club_body})
                assert club_body in (b1, b2)
        samples.append(sim.sample())

    times = np.arange(probe_sim_steps + 1) * 0.002
    run = module.build_run(
        times, samples, {"scenario": "static hold"}, topology_model=topo_model
    )
    assert run.pair_normal_force_n is not None
    assert run.pair_tangential_force_n is not None
    assert run.channel_reasons is not None
    assert run.hand_states is not None
    assert run.club_states is not None

    for pair in topo_model.topology.hand_grip_pairs():
        fn = run.pair_normal_force_n[pair.name]
        ft = run.pair_tangential_force_n[pair.name]
        assert len(fn) == len(run.series.time_s)
        assert len(ft) == len(run.series.time_s)
        assert np.all(np.isfinite(fn))
        assert np.all(np.isfinite(ft))
        assert np.all(fn >= 0.0)

    for pair in topo_model.topology.hand_hand_pairs():
        fn = run.pair_normal_force_n[pair.name]
        ft = run.pair_tangential_force_n[pair.name]
        assert np.all(np.isnan(fn))
        assert np.all(np.isnan(ft))
        assert pair.name in run.channel_reasons

    # Also verify simulate_grip_contact entrypoint populates traces on a short swing step
    swing_run = module.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=0.004,
        topology_model=topo_model,
    )
    assert swing_run.pair_normal_force_n is not None
    assert len(swing_run.series.time_s) == 3


def _hold_swing(swing: CoordinateSwing, duration_s: float) -> CoordinateSwing:
    n = int(round(duration_s / 0.002)) + 1
    return CoordinateSwing(
        swing.names,
        np.arange(n) * 0.002,
        np.repeat(swing.q[:1], n, axis=0),
        swing.sha256,
    )


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_dynamic_net_wrench_parity_hold_vs_pinocchio(style: GripStyle) -> None:
    """Determinate net club wrench (force and moment about CoM) on 0.1 s hold vs Pinocchio."""
    mj_mod = _mujoco_sim_module()
    pin_mod = _pinocchio_sim_module()
    spec_bytes, spec, _, swing, interface, topo_model = _setup(style=style)
    club = ClubDynamics.from_spec(spec)
    duration = 0.1
    hold_sw = _hold_swing(swing, duration)

    mj_run = mj_mod.simulate_grip_contact(
        spec_bytes,
        hold_sw,
        interface=interface,
        t_end_s=duration,
        hold=True,
        topology_model=topo_model,
    )
    pin_run = pin_mod.simulate_grip_contact(
        spec_bytes,
        hold_sw,
        interface=interface,
        t_end_s=duration,
        hold=True,
        topology_model=topo_model,
    )
    errors = net_club_wrench_errors(mj_run.series, pin_run.series, club, interface)
    assert not failing_net_wrench(errors), failing_net_wrench(errors)


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_dynamic_net_wrench_parity_short_swing_vs_pinocchio(style: GripStyle) -> None:
    """Determinate net club wrench (force and moment about CoM) on 0.1 s swing vs Pinocchio."""
    mj_mod = _mujoco_sim_module()
    pin_mod = _pinocchio_sim_module()
    spec_bytes, spec, _, swing, interface, topo_model = _setup(style=style)
    club = ClubDynamics.from_spec(spec)
    duration = 0.1

    mj_run = mj_mod.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        topology_model=topo_model,
    )
    pin_run = pin_mod.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        hold=False,
        topology_model=topo_model,
    )
    errors = net_club_wrench_errors(mj_run.series, pin_run.series, club, interface)
    assert not failing_net_wrench(errors), failing_net_wrench(errors)


def test_capsule_line_contact_continuous_and_matches_shared_law_ten_finger() -> None:
    """Native node contacts follow the continuous shared line law (#13308).

    The old capsule geom snapped its contact 8 mm (overlap centre to endpoint)
    once the finger tilted past ``sin^2(theta) = 1e-10`` at ~1.18 ms; the
    node spheres keep the normal-force-weighted contact centroid on the shared
    law's contact point before and after that tilt, with no geometry switch.
    """
    module = _mujoco_sim_module()
    spec_bytes, _, names, swing, interface, topo_model = _setup(
        style=GripStyle.TEN_FINGER
    )
    q0 = np.asarray(swing.q[0], float)
    sim = module.ClubInHands(
        spec_bytes, names, interface, pads=None, topology_model=topo_model
    )
    sim.calibrate(q0)
    sim.hold_pose(q0)
    pair_little = next(
        p
        for p in topo_model.physical_grip_pairs("R")
        if p.site_a == "trail_little_finger"
    )
    k_little = topo_model.physical_grip_pairs("R").index(pair_little)
    sizes0 = sim.model.geom_size.copy()

    def native_vs_shared() -> tuple[float, np.ndarray]:
        hands = sim.place_hands(float(sim.data.time))
        sim._mj.mj_forward(sim.model, sim.data)
        frame_r = sim.club_state_now().frame(sim._offsets["R"])
        shared = topology_pair_wrench(topo_model, pair_little, hands["R"], frame_r)
        res = np.zeros(6)
        load = 0.0
        moment = np.zeros(3)
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            if sim._pad_geom.get(int(c.geom1)) != ("R", k_little):
                continue
            sim._mj.mj_contactForce(sim.model, sim.data, i, res)
            load += float(res[0])
            moment += float(res[0]) * np.asarray(c.pos, float)
        assert load > 0.0
        centroid = moment / load
        return float(np.linalg.norm(centroid - shared.contact_point_m)), centroid

    err_0, c_0 = native_vs_shared()
    sim.advance(1)  # |theta| ~ 1e-8 rad: below the old 1e-5 rad threshold
    err_1, c_1 = native_vs_shared()
    sim.advance(200)  # t ~ 2 ms: past the old threshold crossing at ~1.18 ms
    err_2, c_2 = native_vs_shared()
    assert max(err_0, err_1, err_2) < 5.0e-4
    # No 8 mm snap: the contact drifts with the club, far less than 1 mm.
    assert float(np.linalg.norm(c_2 - c_1)) < 1.0e-3
    np.testing.assert_array_equal(sim.model.geom_size, sizes0)


def test_capsule_node_friction_matches_shared_line_law_ten_finger() -> None:
    """Node friction rows share one contact's friction gain (#13308).

    MuJoCo's regularised friction gain is per contact row, so ``N`` node rows
    with the single-contact ``solreffriction`` time constant resisted slip
    about ``N`` times as strongly (37.9 N versus the shared law's 19.5 N on
    the trail little finger at 10 ms).  Each node row therefore uses ``N``
    times that time constant, and the pair's tangential force tracks the law.
    """
    module = _mujoco_sim_module()
    spec_bytes, _, names, swing, interface, topo_model = _setup(
        style=GripStyle.TEN_FINGER
    )
    q0 = np.asarray(swing.q[0], float)
    sim = module.ClubInHands(
        spec_bytes, names, interface, pads=None, topology_model=topo_model
    )
    nodes_per_site: dict[tuple[str, int], int] = {}
    for key in sim._pad_geom.values():
        nodes_per_site[key] = nodes_per_site.get(key, 0) + 1

    def pair_nodes(i: int) -> int:
        g1, g2 = int(sim.model.pair_geom1[i]), int(sim.model.pair_geom2[i])
        return nodes_per_site[sim._pad_geom.get(g1) or sim._pad_geom[g2]]

    single = {
        float(sim.model.pair_solreffriction[i, 0])
        for i in range(sim.model.npair)
        if pair_nodes(i) == 1
    }
    assert len(single) == 1
    (base_time,) = single
    for i in range(sim.model.npair):
        assert sim.model.pair_solreffriction[i, 0] == pytest.approx(
            pair_nodes(i) * base_time
        )
    assert max(nodes_per_site.values()) == 5

    sim.calibrate(q0)
    sim.hold_pose(q0)
    pair_little = next(
        p
        for p in topo_model.physical_grip_pairs("R")
        if p.site_a == "trail_little_finger"
    )
    k_little = topo_model.physical_grip_pairs("R").index(pair_little)
    sim.advance(1000)  # t = 10 ms: the club pitch has loaded the finger in shear
    hands = sim.place_hands(float(sim.data.time))
    sim._mj.mj_forward(sim.model, sim.data)
    frame_r = sim.club_state_now().frame(sim._offsets["R"])
    shared = topology_pair_wrench(topo_model, pair_little, hands["R"], frame_r)
    res = np.zeros(6)
    native_ft = 0.0
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        if sim._pad_geom.get(int(c.geom1)) != ("R", k_little):
            continue
        sim._mj.mj_contactForce(sim.model, sim.data, i, res)
        native_ft += float(np.hypot(res[1], res[2]))
    assert shared.tangential_force_mag_n > 5.0
    assert native_ft == pytest.approx(shared.tangential_force_mag_n, rel=0.10)


def test_scheduled_node_friction_rows_follow_shared_law_ten_finger() -> None:
    """Every loaded node friction row carries the shared law's force (#13306).

    A fixed ``solreffriction`` time gives each row one viscous gain, so rows
    at other loads and slip speeds missed the regularised law by up to 17 %
    after 10 ms of the ten-finger hold.  ``advance`` schedules each row's
    time constant to the law's secant gain, and switching step scaling off
    restores the static (calibration) time constants exactly.
    """
    module = _mujoco_sim_module()
    spec_bytes, _, names, swing, interface, topo_model = _setup(
        style=GripStyle.TEN_FINGER
    )
    q0 = np.asarray(swing.q[0], float)
    sim = module.ClubInHands(
        spec_bytes, names, interface, pads=None, topology_model=topo_model
    )
    sim.calibrate(q0)
    sim.hold_pose(q0)
    static_time = sim.model.pair_solreffriction[:, 0].copy()
    static_impratio = float(sim.model.opt.impratio)
    sim.advance(1000)
    d = sim.data
    ratios = []
    for i in range(d.ncon):
        c = d.contact[i]
        if c.efc_address < 0 or int(c.geom1) not in sim._pad_geom:
            continue
        a = int(c.efc_address)
        speed = float(np.hypot(d.efc_vel[a + 1], d.efc_vel[a + 2]))
        law_ft = friction_secant_gain(speed, topo_model.law) * d.efc_force[a] * speed
        if law_ft > 0.05:
            ratios.append(
                float(np.hypot(d.efc_force[a + 1], d.efc_force[a + 2])) / law_ft
            )
    assert len(ratios) >= 10
    np.testing.assert_allclose(ratios, 1.0, rtol=0.02)
    assert not np.array_equal(sim.model.pair_solreffriction[:, 0], static_time)

    sim._set_step_scaling(False)
    np.testing.assert_array_equal(sim.model.pair_solreffriction[:, 0], static_time)
    assert sim.model.opt.impratio == pytest.approx(static_impratio, rel=1e-12)


def test_resting_sample_carries_zero_law_friction_ten_finger() -> None:
    """The at-rest t = 0 sample carries the shared law's zero friction (#13560).

    Before any step MuJoCo's soft friction rows answered gravity with a
    regulariser force of 2-11 mN per pair, a net moment of 16 mN m that the
    balanced reference preload (#13555) does not have; it set the ten-finger
    hold's net-moment RMS to 2.27 %.  Sampling restores the pair ``condim``.
    """
    module = _mujoco_sim_module()
    spec_bytes, _, names, swing, interface, topo_model = _setup(
        style=GripStyle.TEN_FINGER
    )
    q0 = np.asarray(swing.q[0], float)
    sim = module.ClubInHands(
        spec_bytes, names, interface, pads=None, topology_model=topo_model
    )
    sim.calibrate(q0)
    sim.hold_pose(q0)
    condim = sim.model.pair_dim.copy()
    smp = sim.sample()
    for side in ("L", "R"):
        wrench = smp.side_wrenches[side]
        for pair in wrench.pair_wrenches.values():
            assert pair.tangential_force_mag_n == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_array_equal(sim.model.pair_dim, condim)
    sim.advance(10)  # the friction schedule reads the restored rows


def test_two_pad_hyperstatic_penalty_vs_mujoco_cone_divergence() -> None:
    """Minimal 2-pad hyperstatic test (#13306) proving why MuJoCo cone QP diverges from penalty split."""
    import mujoco

    from src.shared.python.grip_contact.bushing_law import BushingState
    from src.shared.python.grip_contact.pair_contact import (
        ContactCapsule,
        ContactSphere,
        sphere_capsule,
    )
    from src.shared.python.motion_matching.contact_law import ContactParameters

    law = ContactParameters(
        stiffness_n_m=171095.0080515298,
        dissipation_s_m=0.38266566243,
        static_friction=0.9,
        dynamic_friction=0.7,
        viscous_friction=0.0,
        transition_velocity_m_s=1.0e-3,
    )
    delta0, r_grip, r_pad = 0.00025, 0.014, 0.008
    x_pad = r_grip + r_pad - delta0
    m_club, m_hand = 0.313, 100.0
    a_hat = (1.0 / m_club) + (1.0 / m_hand)
    d0 = 0.001
    k_sol = (1.0 - d0) * a_hat * law.stiffness_n_m
    b_sol = k_sol * delta0 * law.dissipation_s_m

    sphere_l = ContactSphere(center_m=np.array([-x_pad, 0.0, 0.03]), radius_m=r_pad)
    sphere_r = ContactSphere(center_m=np.array([+x_pad, 0.0, -0.03]), radius_m=r_pad)
    grip_cap = ContactCapsule(
        p0_m=np.array([0.0, 0.0, -0.15]),
        p1_m=np.array([0.0, 0.0, +0.15]),
        radius_m=r_grip,
    )
    hand_st = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))

    def _eval_mj(
        v_club: np.ndarray, condim: int, cone: str, imp: float, tf: float
    ) -> tuple[float, float, float]:
        xml = (
            f'<mujoco><compiler inertiafromgeom="false"/>'
            f'<option timestep="1e-5" integrator="Euler" cone="{cone}" impratio="{imp}" gravity="0 0 0">'
            f'<flag multiccd="disable" refsafe="disable" warmstart="disable"/></option>'
            f'<worldbody><body name="club"><freejoint/>'
            f'<inertial pos="0 0 0" mass="{m_club}" diaginertia="0.05 0.05 0.0004"/>'
            f'<geom name="grip" type="cylinder" fromto="0 0 -0.15 0 0 0.15" size="{r_grip}" contype="0" conaffinity="0"/></body>'
            f'<body name="hand_L"><freejoint/><inertial pos="0 0 0" mass="{m_hand}" diaginertia="10 10 10"/>'
            f'<geom name="pad_L" type="sphere" pos="{-x_pad} 0 0.03" size="{r_pad}" contype="0" conaffinity="0"/></body>'
            f'<body name="hand_R"><freejoint/><inertial pos="0 0 0" mass="{m_hand}" diaginertia="10 10 10"/>'
            f'<geom name="pad_R" type="sphere" pos="{x_pad} 0 -0.03" size="{r_pad}" contype="0" conaffinity="0"/></body></worldbody>'
            f'<contact><exclude body1="hand_L" body2="hand_R"/>'
            f'<pair geom1="pad_L" geom2="grip" condim="{condim}" friction="0.9 0.9 0 0 0" '
            f'solref="-{k_sol:.17g} -{b_sol:.17g}" solreffriction="{tf:.17g} 1" solimp="{d0} {d0} 0.001 0.5 2"/>'
            f'<pair geom1="pad_R" geom2="grip" condim="{condim}" friction="0.9 0.9 0 0 0" '
            f'solref="-{k_sol:.17g} -{b_sol:.17g}" solreffriction="{tf:.17g} 1" solimp="{d0} {d0} 0.001 0.5 2"/></contact></mujoco>'
        )
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        d.qvel[0:3] = v_club
        mujoco.mj_forward(m, d)
        res_l, res_r = np.zeros(6), np.zeros(6)
        mujoco.mj_contactForce(m, d, 0, res_l)
        mujoco.mj_contactForce(m, d, 1, res_r)
        return (
            float(res_l[0]),
            float(np.hypot(res_l[1], res_l[2])),
            float(np.linalg.norm(res_l[:3])),
        )

    # State 0 (v=0): elliptic matches static normal force (42.774 N), pyramidal biases it (+29.6%)
    v0 = np.zeros(3)
    pw0 = sphere_capsule(
        sphere_l, grip_cap, hand_st, BushingState(np.eye(3), v0, v0, v0), law
    )
    fn0_ell, _, _ = _eval_mj(v0, 3, "elliptic", 0.95, 1.9e-5)
    fn0_pyr, _, _ = _eval_mj(v0, 3, "pyramidal", 0.95, 1.9e-5)
    assert fn0_ell == pytest.approx(pw0.normal_force_mag_n, rel=1e-3)
    assert (fn0_pyr - pw0.normal_force_mag_n) / pw0.normal_force_mag_n > 0.25

    # State 2 (v_z = -3 mm/s = 3*v_tr): penalty keeps fn=42.774 N, whereas MuJoCo elliptic cone
    # QCQP projection inflates normal squeeze to >70 N (+65.9%) and |f_L| by +83.1%
    v2 = np.array([0.0, 0.0, -0.003])
    pw2 = sphere_capsule(
        sphere_l, grip_cap, hand_st, BushingState(np.eye(3), v0, v2, v0), law
    )
    fn2_ell, ft2_ell, f2_ell = _eval_mj(v2, 3, "elliptic", 0.95, 1.9e-5)
    assert pw2.normal_force_mag_n == pytest.approx(42.773752, rel=1e-5)
    assert (fn2_ell - pw2.normal_force_mag_n) / pw2.normal_force_mag_n > 0.60
    assert (f2_ell - float(np.linalg.norm(pw2.force_b_n))) / float(
        np.linalg.norm(pw2.force_b_n)
    ) > 0.80
    assert ft2_ell == pytest.approx(0.9 * fn2_ell, rel=1e-4)


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "style",
    list(GripStyle),
)
def test_dynamic_parity_hold_vs_pinocchio(style: GripStyle) -> None:
    """0.1 s hold full PARITY_QUANTITIES (including indeterminate per-hand split) vs Pinocchio."""
    mj_mod = _mujoco_sim_module()
    pin_mod = _pinocchio_sim_module()
    spec_bytes, _, _, swing, interface, topo_model = _setup(style=style)
    duration = 0.1
    hold_sw = _hold_swing(swing, duration)

    mj_run = mj_mod.simulate_grip_contact(
        spec_bytes,
        hold_sw,
        interface=interface,
        t_end_s=duration,
        hold=True,
        topology_model=topo_model,
    )
    pin_run = pin_mod.simulate_grip_contact(
        spec_bytes,
        hold_sw,
        interface=interface,
        t_end_s=duration,
        hold=True,
        topology_model=topo_model,
    )
    errors = parity_errors(mj_run.series, pin_run.series)
    failing = {
        k: v.to_dict()
        for k, v in errors.items()
        if not (
            v.passes()
            and v.peak_error <= PEAK_TOLERANCE
            and v.rms_error <= RMS_TOLERANCE
        )
    }
    assert not failing, failing


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "style",
    list(GripStyle),
)
def test_dynamic_parity_short_swing_vs_pinocchio(style: GripStyle) -> None:
    """Short 0.1 s swing full PARITY_QUANTITIES (including indeterminate per-hand split) vs Pinocchio."""
    mj_mod = _mujoco_sim_module()
    pin_mod = _pinocchio_sim_module()
    spec_bytes, _, _, swing, interface, topo_model = _setup(style=style)
    duration = 0.1

    mj_run = mj_mod.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        topology_model=topo_model,
    )
    pin_run = pin_mod.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        hold=False,
        topology_model=topo_model,
    )
    errors = parity_errors(mj_run.series, pin_run.series)
    failing = {
        k: v.to_dict()
        for k, v in errors.items()
        if not (
            v.passes()
            and v.peak_error <= PEAK_TOLERANCE
            and v.rms_error <= RMS_TOLERANCE
        )
    }
    assert not failing, failing
