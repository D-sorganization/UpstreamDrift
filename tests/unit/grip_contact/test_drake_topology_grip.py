"""Drake two-hand-body refactor and topology contact grip tests (#13307, HRG-GS-G9).

Acceptance criteria:
- Each hand has its own prescribed body ("lead_hand", "trail_hand") in Drake
  ClubInHands, driven by its own GripInterface hand frame (hand_frame_states per side).
- Regression test fails if any pad or site is attached to the wrong hand body.
- Pure helper computes hand-body assignment of every site without building a plant.
- Per-hand pair forces are recorded separately (lead pairs from lead_hand only,
  trail pairs from trail_hand only), and hand-hand pairs are NaN with a reason.
- Per-pair force traces equal topology_side_wrench re-evaluated at the recorded
  states to <= 1e-9 N.
- Parity versus the Pinocchio topology reference via grip_contact.parity is within
  PEAK_TOLERANCE (5 %) and RMS_TOLERANCE (2 %) per style (overlap, interlock,
  ten_finger) on a 0.04 s hold and a 0.04 s swing replay.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.grip_contact import (
    BushingState,
    GripInterface,
    load_coordinate_swing,
)
from src.shared.python.grip_contact.contact_topology import (
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.pad_contact import build_pad_model
from src.shared.python.grip_contact.parity import (
    PEAK_TOLERANCE,
    RMS_TOLERANCE,
    parity_errors,
)
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    build_topology_model,
    topology_side_wrench,
)
from src.shared.python.motion_matching.grip_contract import GripSpec

pytestmark = [pytest.mark.unit, pytest.mark.requires_drake, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
SPEC_PATH = MODELS / "full_body_spec_anthro_driver.json"
SQUEEZE = 1100.0
DURATION_S = 0.04


def _drake_sim_module() -> Any:
    try:
        import pydrake  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"pydrake not installed: {exc}")
    from src.engines.physics_engines.drake.python import grip_contact_sim

    return grip_contact_sim


def _pinocchio_sim_module() -> Any:
    try:
        import pinocchio  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"pinocchio not installed: {exc}")
    from src.engines.physics_engines.pinocchio.python import grip_contact_sim

    return grip_contact_sim


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


def _assert_traces_match_law(run: Any, topo_model: TopologyContactModel) -> None:
    """Every recorded pair trace equals topology_side_wrench re-evaluated (<= 1e-9 N)."""
    assert run.pair_normal_force_n is not None
    assert run.pair_tangential_force_n is not None
    assert run.hand_states is not None
    assert run.club_states is not None
    for k in range(len(run.series.time_s)):
        for side in ("L", "R"):
            re_eval = topology_side_wrench(
                topo_model, side, run.hand_states[side][k], run.club_states[side][k]
            )
            for pair in topo_model.physical_grip_pairs(side):
                pw = re_eval.pair_wrenches[pair.name]
                np.testing.assert_allclose(
                    run.pair_normal_force_n[pair.name][k],
                    pw.normal_force_mag_n,
                    atol=1e-9,
                )
                np.testing.assert_allclose(
                    run.pair_tangential_force_n[pair.name][k],
                    pw.tangential_force_mag_n,
                    atol=1e-9,
                )


def test_hand_body_assignment_pure_without_plant() -> None:
    """Pure helper computes the hand-body assignment of every site without a plant."""
    from src.engines.physics_engines.drake.python.grip_contact_sim import (
        HAND_BODY_NAMES,
        hand_body_name,
        hand_site_attachments,
    )

    assert hand_body_name("L") == "lead_hand"
    assert hand_body_name("lead") == "lead_hand"
    assert hand_body_name("R") == "trail_hand"
    assert hand_body_name("trail") == "trail_hand"
    assert HAND_BODY_NAMES == {"L": "lead_hand", "R": "trail_hand"}

    with pytest.raises(TypeError):
        hand_body_name(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        hand_body_name("both")
    with pytest.raises(ValueError):
        hand_site_attachments()

    _, _, _, _, interface, _ = _setup()
    pads = build_pad_model(interface, SQUEEZE)
    pad_atts = hand_site_attachments(pads=pads)
    assert len(pad_atts) == 2 * pads.layout.pad_count
    for att in pad_atts:
        expected_body = "lead_hand" if att.side == "L" else "trail_hand"
        assert att.hand_body_name == expected_body
        assert att.name.startswith(f"pad_{att.side}")

    for style in GripStyle:
        topo = contact_topology(GripSpec(style=style))
        topo_model = build_topology_model(interface, topo, squeeze_per_hand_n=SQUEEZE)
        site_atts = hand_site_attachments(topology_model=topo_model)
        assert len(site_atts) == len(topo.sites)
        by_name = {a.name: a for a in site_atts}
        for site in topo.sites:
            att = by_name[site.name]
            if site.hand == "lead":
                assert att.side == "L"
                assert att.hand_body_name == "lead_hand"
            else:
                assert att.side == "R"
                assert att.hand_body_name == "trail_hand"


def test_two_hand_bodies_and_wrong_hand_regression() -> None:
    """Each hand has its own body; attaching any pad or site to the wrong body fails."""
    module = _drake_sim_module()
    from pydrake.geometry import ProximityProperties, Sphere
    from pydrake.math import RigidTransform
    from pydrake.multibody.tree import FixedOffsetFrame

    _, spec, _, _, interface, topo_model = _setup(style=GripStyle.OVERLAP)
    pads = build_pad_model(interface, SQUEEZE)

    # 1. Pad plant has distinct lead_hand and trail_hand bodies and valid attachments
    sim_pads = module.ClubInHands(spec, interface, pads=pads)
    assert set(sim_pads.hands.keys()) == {"L", "R"}
    assert sim_pads.hands["L"].name() == "lead_hand"
    assert sim_pads.hands["R"].name() == "trail_hand"
    assert sim_pads.hands["L"].index() != sim_pads.hands["R"].index()
    sim_pads.validate_hand_attachments()

    # 2. Topology plant has distinct lead_hand and trail_hand bodies and valid attachments
    sim_topo = module.ClubInHands(spec, interface, topology_model=topo_model)
    assert sim_topo.hands["L"].name() == "lead_hand"
    assert sim_topo.hands["R"].name() == "trail_hand"
    sim_topo.validate_hand_attachments()

    # 3. Regression check: mutating a pad frame onto the wrong hand body must fail
    wrong_frame = FixedOffsetFrame(
        "pad_L0_wrong",
        sim_pads.hands["R"].body_frame(),
        RigidTransform(),
    )
    sim_pads.site_frames["pad_L0"] = wrong_frame
    with pytest.raises(ValueError, match="wrong hand body"):
        sim_pads.validate_hand_attachments()

    # 4. Regression check: mutating a topology site frame onto the wrong hand body must fail
    wrong_site_frame = FixedOffsetFrame(
        "trail_index_middle_wrong",
        sim_topo.hands["L"].body_frame(),
        RigidTransform(),
    )
    sim_topo.site_frames["trail_index_middle"] = wrong_site_frame
    with pytest.raises(ValueError, match="wrong hand body"):
        sim_topo.validate_hand_attachments()

    # 5. Regression check: collision geometry registered on the wrong hand body must fail
    sim_bad_geom = module.ClubInHands(spec, interface, pads=pads)
    # Swap a lead pad's recorded side in pad_ids to simulate wrong-hand geometry attachment
    first_gid = next(
        gid for gid, (side, _) in sim_bad_geom.pad_ids.items() if side == "L"
    )
    sim_bad_geom.pad_ids[first_gid] = ("R", 0)
    with pytest.raises(ValueError, match="wrong hand body"):
        sim_bad_geom.validate_hand_attachments()


def test_per_hand_pair_isolation_and_reaction_balance() -> None:
    """Moving only lead_hand changes only lead pairs; moving trail_hand changes only trail pairs."""
    module = _drake_sim_module()
    spec_bytes, spec, names, swing, interface, topo_model = _setup(
        style=GripStyle.OVERLAP
    )
    kin = module.WeldClubKinematics(spec_bytes, list(swing.names))
    sim = module.ClubInHands(spec, interface, topology_model=topo_model)
    q0 = np.asarray(swing.q[0], float)
    weld0 = kin.state(q0, np.zeros_like(q0))
    sim.set_hand(weld0)
    sim.place_club_at(weld0)

    # Evaluate baseline wrenches and applied spatial forces
    _ = sim.wrenches()
    base_wrenches = dict(sim.last_side_wrenches)
    hands_base, _ = sim.bushing_states()

    # Displace ONLY lead_hand ("L") by 0.2 mm radially (toe axis); keep trail_hand ("R") fixed
    shifted_L = BushingState(
        rotation=hands_base["L"].rotation,
        position_m=hands_base["L"].position_m + 2.0e-4 * hands_base["L"].rotation[:, 1],
        velocity_m_s=hands_base["L"].velocity_m_s,
        omega_rad_s=hands_base["L"].omega_rad_s,
    )
    sim.set_hand_frames({"L": shifted_L, "R": hands_base["R"]})
    _ = sim.wrenches()
    after_L = dict(sim.last_side_wrenches)

    # Lead pairs changed, trail pairs stayed identical to machine precision
    assert not np.allclose(after_L["L"].force_n, base_wrenches["L"].force_n)
    assert not np.allclose(
        after_L["L"].normal_force_n, base_wrenches["L"].normal_force_n
    )
    np.testing.assert_allclose(
        after_L["R"].force_n, base_wrenches["R"].force_n, atol=1e-12
    )
    for pair in topo_model.physical_grip_pairs("R"):
        np.testing.assert_allclose(
            after_L["R"].pair_wrenches[pair.name].normal_force_mag_n,
            base_wrenches["R"].pair_wrenches[pair.name].normal_force_mag_n,
            atol=1e-12,
        )

    # Check ExternallyAppliedSpatialForce action-reaction balance on club and hands
    _ = sim.club_derivatives()
    assert len(sim.last_applied_spatial_forces) == 4
    total_f = np.zeros(3)
    for sf in sim.last_applied_spatial_forces:
        total_f += np.asarray(sf.F_Bq_W.translational())
    np.testing.assert_allclose(total_f, np.zeros(3), atol=1e-12)


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
@pytest.mark.parametrize("hold", [True, False], ids=["hold", "swing"])
def test_drake_topology_parity_and_traces_vs_pinocchio(
    style: GripStyle, hold: bool
) -> None:
    """Drake topology hold & swing match Pinocchio within 5 % peak / 2 % RMS and law to <= 1e-9 N."""
    drake_mod = _drake_sim_module()
    pin_mod = _pinocchio_sim_module()
    spec_bytes, _, _, swing, interface, topo_model = _setup(style=style)

    drake_run = drake_mod.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=DURATION_S,
        hold=hold,
        topology_model=topo_model,
    )
    pin_run = pin_mod.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=DURATION_S,
        hold=hold,
        topology_model=topo_model,
    )

    # 1. Per-pair traces match topology_side_wrench re-evaluated at recorded states to <= 1e-9 N
    _assert_traces_match_law(drake_run, topo_model)

    # 2. Lead and trail physical pairs are finite and nonnegative; hand-hand pairs are NaN + reason
    assert drake_run.pair_normal_force_n is not None
    assert drake_run.pair_tangential_force_n is not None
    assert drake_run.channel_reasons is not None
    for side in ("L", "R"):
        pairs = topo_model.physical_grip_pairs(side)
        assert drake_run.normal_force_n[side].shape == (
            len(drake_run.series.time_s),
            len(pairs),
        )
        for idx, pair in enumerate(pairs):
            fn_trace = drake_run.pair_normal_force_n[pair.name]
            ft_trace = drake_run.pair_tangential_force_n[pair.name]
            assert np.all(np.isfinite(fn_trace))
            assert np.all(np.isfinite(ft_trace))
            assert np.all(fn_trace >= 0.0)
            np.testing.assert_allclose(
                drake_run.normal_force_n[side][:, idx], fn_trace, atol=1e-12
            )

    for pair in topo_model.topology.hand_hand_pairs():
        assert np.all(np.isnan(drake_run.pair_normal_force_n[pair.name]))
        assert np.all(np.isnan(drake_run.pair_tangential_force_n[pair.name]))
        assert pair.name in drake_run.channel_reasons

    # 3. Parity versus Pinocchio reference within PEAK_TOLERANCE (5 %) and RMS_TOLERANCE (2 %)
    errors = parity_errors(drake_run.series, pin_run.series)
    max_peak = max(e.peak_error for e in errors.values())
    max_rms = max(e.rms_error for e in errors.values())
    print(
        f"[parity] style={style.value} hold={hold}: "
        f"max_peak_error={max_peak * 100:.4f}% (tol {PEAK_TOLERANCE * 100:.1f}%), "
        f"max_rms_error={max_rms * 100:.4f}% (tol {RMS_TOLERANCE * 100:.1f}%)"
    )
    failing = {k: v.to_dict() for k, v in errors.items() if not v.passes()}
    assert not failing, f"Parity failed for style={style.value} hold={hold}: {failing}"
