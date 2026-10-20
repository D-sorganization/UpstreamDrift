"""OpenSim contact grip integration tests with ContactTopology (#13305, HRG-GS-G7).

Acceptance criteria:
- For each style (overlap, interlock, ten_finger): a hold and a short swing replay,
  with the same fixture as the Pinocchio topology test.
- Parity of the per-side total normal force versus the Pinocchio topology reference,
  via grip_contact.parity, within EXISTING PEAK_TOLERANCE (5 %) and RMS_TOLERANCE (2 %).
- Every hand-hand pair is NaN with a reason.
- Existing OpenSim grip tests stay green, and the old pad path is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.helpers.opensim_support import require_opensim

require_opensim()

from src.engines.physics_engines.opensim.python.grip_contact_osim_sim import (
    ContactGripSimulator,
    calibrated_site_foundation,
    pad_normal_force_n,
    simulate_grip_contact,
    write_calibration_mesh,
)
from src.engines.physics_engines.pinocchio.python import (
    grip_contact_sim as pin_sim,
)
from src.shared.python.grip_contact import (
    GripInterface,
    load_coordinate_swing,
)
from src.shared.python.grip_contact.bushing_law import BushingState
from src.shared.python.grip_contact.contact_topology import (
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.parity import (
    PEAK_TOLERANCE,
    RMS_TOLERANCE,
    QuantityError,
)
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    build_topology_model,
    topology_pair_wrench,
)
from src.shared.python.motion_matching.grip_contract import GripSpec

# Full-model runs integrate OpenSim and Pinocchio stiff contact (~20-60 s on a
# loaded host) and carry @pytest.mark.slow; the single-pair sliding probe runs
# in the default fast unit lane.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.requires_opensim,
]

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
SPEC_PATH = MODELS / "full_body_spec_anthro_driver.json"
SQUEEZE = 1100.0


def _setup(
    style: GripStyle = GripStyle.OVERLAP, trigger_finger: bool = False
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
    topo = contact_topology(gspec)
    topo_model = build_topology_model(interface, topo, squeeze_per_hand_n=SQUEEZE)
    return spec_bytes, spec, names, swing, interface, topo_model


def test_single_pair_sliding_friction_matches_shared_law(tmp_path: Path) -> None:
    """Single topology sphere on grip cylinder matches shared law normal stiffness and sliding friction within 5%."""
    osim = require_opensim()
    _, _, _, _, _, topo_model = _setup(style=GripStyle.OVERLAP)
    site = topo_model.topology.site("lead_palm_proximal")
    grip_r = topo_model.cylinder.radius_m
    ef = calibrated_site_foundation(topo_model, site.radius_m, grip_r, count=1)
    mesh = write_calibration_mesh(ef, tmp_path)

    # 1. Normal force and normal tangent stiffness at operating preload
    d0 = ef.preload_penetration_m
    f0 = pad_normal_force_n(ef, mesh, d0)
    assert f0 == pytest.approx(ef.force_per_pad_n, rel=1e-2)
    h = 5.0e-5
    tangent = (
        pad_normal_force_n(ef, mesh, d0 + h) - pad_normal_force_n(ef, mesh, d0 - h)
    ) / (2.0 * h)
    assert tangent == pytest.approx(topo_model.law.stiffness_n_m, rel=0.05)

    # 2. Imposed tangential slip velocity sweep (1e-4 to 1.0 m/s) along cylinder axis
    model = osim.Model()
    model.setGravity(osim.Vec3(0.0, 0.0, 0.0))
    body = osim.Body("pad", 0.5, osim.Vec3(0), osim.Inertia(1e-3))
    model.addBody(body)
    joint = osim.FreeJoint(
        "free",
        model.getGround(),
        osim.Vec3(0),
        osim.Vec3(0),
        body,
        osim.Vec3(0),
        osim.Vec3(0),
    )
    model.addJoint(joint)
    model.addContactGeometry(
        osim.ContactSphere(site.radius_m, osim.Vec3(0), body, "pad")
    )
    model.addContactGeometry(
        osim.ContactMesh(
            str(mesh), osim.Vec3(0), osim.Vec3(0), model.getGround(), "grip"
        )
    )
    force = osim.ElasticFoundationForce()
    force.setName("pad_force")
    force.addGeometry("pad")
    force.addGeometry("grip")
    force.setStiffness(ef.stiffness_n_m3)
    force.setDissipation(ef.dissipation_s_m)
    force.setStaticFriction(ef.static_friction)
    force.setDynamicFriction(ef.dynamic_friction)
    force.setViscousFriction(ef.viscous_friction)
    force.setTransitionVelocity(ef.transition_velocity_m_s)
    model.addForce(force)

    state = model.initSystem()
    coords = model.updCoordinateSet()
    coords.get("free_coord_4").setValue(state, grip_r + site.radius_m - d0)

    speeds = np.geomspace(1.0e-4, 1.0, 17)
    ft_ref_list: list[float] = []
    ft_osim_list: list[float] = []
    club_state = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))
    for v_slip in speeds:
        coords.get("free_coord_5").setSpeedValue(state, float(v_slip))
        model.realizeDynamics(state)
        rec = force.getRecordValues(state)
        ft_osim_list.append(abs(float(rec.get(2))))
        hand_state = BushingState(
            np.eye(3), np.zeros(3), np.array([float(v_slip), 0.0, 0.0]), np.zeros(3)
        )
        pw = topology_pair_wrench(
            topo_model, "lead_palm_proximal_grip", hand_state, club_state
        )
        ft_ref_list.append(pw.tangential_force_mag_n)

    ft_ref = np.asarray(ft_ref_list, dtype=float)
    ft_osim = np.asarray(ft_osim_list, dtype=float)
    peak_ref = float(np.max(ft_ref))
    peak_cand = float(np.max(ft_osim))
    peak_err = abs(peak_cand - peak_ref) / peak_ref
    rms_err = float(np.sqrt(np.mean((ft_osim - ft_ref) ** 2))) / peak_ref
    assert peak_err <= PEAK_TOLERANCE, (
        f"Single-pair sliding peak error {peak_err * 100:.3f}% > {PEAK_TOLERANCE * 100:.1f}%"
    )
    assert rms_err <= PEAK_TOLERANCE, (
        f"Single-pair sliding RMS error {rms_err * 100:.3f}% > {PEAK_TOLERANCE * 100:.1f}%"
    )


def test_capsule_site_zero_static_mesh_bias_and_cop_reporting() -> None:
    """Capsule sites have ring-aligned meshes (zero static tangential bias at t=0) and report CoP."""
    spec_bytes, _, names, swing, interface, topo_model = _setup(
        style=GripStyle.TEN_FINGER
    )
    q_hold = np.tile(np.asarray(swing.q[0], float), (5, 1))
    sim = ContactGripSimulator(
        spec_bytes,
        names,
        np.asarray(swing.time_s[:5], float),
        q_hold,
        interface=interface,
        topology_model=topo_model,
    )
    sim._model.realizeDynamics(sim._state)  # noqa: SLF001
    sample_l = sim._side_sample(sim._state, "left")  # noqa: SLF001
    sample_r = sim._side_sample(sim._state, "right")  # noqa: SLF001
    sw_l = sample_l["side_wrench"]
    sw_r = sample_r["side_wrench"]
    pw_cap = sw_r.pair_wrenches["ten_finger_little_finger_grip"]

    # 1. Zero static tangential force bias (< 1e-4 N) and exact preload normal force at t=0
    assert pw_cap.tangential_force_mag_n < 1.0e-4, (
        f"Capsule site has spurious static tangential force {pw_cap.tangential_force_mag_n:.6f} N at t=0"
    )
    target_fn = topo_model.law.stiffness_n_m * 0.00025
    assert pw_cap.normal_force_mag_n == pytest.approx(target_fn, abs=1.0e-3)
    f_net_0 = float(np.linalg.norm(sw_l.force_n + sw_r.force_n))
    assert f_net_0 < 1.0e-3, (
        f"Static net force residual at t=0 is {f_net_0:.6f} N (expected < 1e-3 N)"
    )

    # 2. Reported contact point sits at capsule axial midpoint (-36.9582 mm), not distal endpoint p1 (-28.9582 mm)
    r_club = sample_r["club_bstate"].rotation
    p_club = sample_r["club_bstate"].position_m
    cp_in_club = r_club.T @ (pw_cap.contact_point_m - p_club)
    site_cap = topo_model.topology.site("trail_little_finger")
    from src.shared.python.grip_contact.topology_contact import map_site_to_hand_frame

    mapped_cap = map_site_to_hand_frame(topo_model.topology, site_cap, "R")
    assert cp_in_club[0] == pytest.approx(mapped_cap.centre_m[0], abs=1.0e-4)


def _check_pair_traces(run: Any, topo_model: TopologyContactModel) -> None:
    """Verify physical pairs are finite and hand-hand pairs are NaN with reasons."""
    assert run.pair_normal_force_n is not None
    assert run.pair_tangential_force_n is not None
    assert run.channel_reasons is not None

    for pair in topo_model.topology.hand_grip_pairs():
        fn_trace = run.pair_normal_force_n[pair.name]
        ft_trace = run.pair_tangential_force_n[pair.name]
        assert len(fn_trace) == len(run.series.time_s)
        assert len(ft_trace) == len(run.series.time_s)
        assert np.all(np.isfinite(fn_trace))
        assert np.all(np.isfinite(ft_trace))
        assert np.all(fn_trace >= 0.0)

    for pair in topo_model.topology.hand_hand_pairs():
        fn_trace = run.pair_normal_force_n[pair.name]
        ft_trace = run.pair_tangential_force_n[pair.name]
        assert np.all(np.isnan(fn_trace))
        assert np.all(np.isnan(ft_trace))
        assert pair.name in run.channel_reasons


def _check_parity_versus_pinocchio(
    run_osim: Any, run_pin: Any, label: str
) -> dict[str, QuantityError]:
    """Check per-side total normal force parity versus Pinocchio."""
    errors = {}
    for side in ("L", "R"):
        x_ref = np.asarray(run_pin.squeeze_n[side], dtype=float)
        x_cand = np.asarray(run_osim.squeeze_n[side], dtype=float)
        peak_ref = float(np.max(np.abs(x_ref)))
        peak_cand = float(np.max(np.abs(x_cand)))
        peak_err = abs(peak_cand - peak_ref) / peak_ref
        rms_err = float(np.sqrt(np.mean((x_cand - x_ref) ** 2))) / peak_ref
        qe = QuantityError(
            peak_reference=peak_ref,
            peak_candidate=peak_cand,
            peak_error=peak_err,
            rms_error=rms_err,
        )
        errors[side] = qe
        assert qe.passes(), (
            f"[{label}] side {side} parity failed: "
            f"peak_error={peak_err * 100:.3f}% (max {PEAK_TOLERANCE * 100}%), "
            f"rms_error={rms_err * 100:.3f}% (max {RMS_TOLERANCE * 100}%). "
            f"peak_osim={peak_cand:.3f}N, peak_pin={peak_ref:.3f}N"
        )
    return errors


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.requires_pinocchio
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_opensim_topology_static_hold_fast(style: GripStyle) -> None:
    """Fast check: 0.04 s static hold runs, pair traces valid, parity vs Pinocchio."""
    spec_bytes, _, names, swing, interface, topo_model = _setup(style=style)
    duration = 0.04

    run_pin = pin_sim.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        hold=True,
        topology_model=topo_model,
    )
    run_osim = simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        hold=True,
        topology_model=topo_model,
    )

    _check_pair_traces(run_osim, topo_model)
    _check_parity_versus_pinocchio(run_osim, run_pin, f"hold_fast_{style.value}")


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.requires_pinocchio
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_opensim_topology_swing_replay_fast(style: GripStyle) -> None:
    """Fast check: 0.04 s swing replay runs, pair traces valid, parity vs Pinocchio."""
    spec_bytes, _, names, swing, interface, topo_model = _setup(style=style)
    duration = 0.04

    run_pin = pin_sim.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        hold=False,
        topology_model=topo_model,
    )
    run_osim = simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        hold=False,
        topology_model=topo_model,
    )

    _check_pair_traces(run_osim, topo_model)
    _check_parity_versus_pinocchio(run_osim, run_pin, f"swing_fast_{style.value}")


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.requires_pinocchio
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_opensim_topology_static_hold_slow(style: GripStyle) -> None:
    """Slow check: 0.1 s static hold over all styles."""
    spec_bytes, _, names, swing, interface, topo_model = _setup(style=style)
    duration = 0.1

    run_pin = pin_sim.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        hold=True,
        topology_model=topo_model,
    )
    run_osim = simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        hold=True,
        topology_model=topo_model,
    )

    _check_pair_traces(run_osim, topo_model)
    _check_parity_versus_pinocchio(run_osim, run_pin, f"hold_slow_{style.value}")


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.requires_pinocchio
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_opensim_topology_swing_replay_slow(style: GripStyle) -> None:
    """Slow check: 0.1 s swing replay over all styles."""
    spec_bytes, _, names, swing, interface, topo_model = _setup(style=style)
    duration = 0.1

    run_pin = pin_sim.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        hold=False,
        topology_model=topo_model,
    )
    run_osim = simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=duration,
        hold=False,
        topology_model=topo_model,
    )

    _check_pair_traces(run_osim, topo_model)
    _check_parity_versus_pinocchio(run_osim, run_pin, f"swing_slow_{style.value}")
