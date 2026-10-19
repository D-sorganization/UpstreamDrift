"""Tests for separated LHand/RHand bodies and physical hand-hand contact in full-body-v2 (#13309, HRG-GS-G11).

Covers:
1. Opt-in v2 hand separation (``derive_full_body_v2_spec(..., separate_hands=True)`` and
   ``separate_v2_hands``):
   - Default ``v1`` and ``v2`` (``separate_hands=False``) bytes and SHA256 hashes stay
     immutable.
   - When ``separate_hands=True``: ``LHandBody`` (left wrist child), ``RHandStandoff``
     (right wrist child), and bare ``Clubface Vector`` (6-DOF free joint child) are
     distinct rigid bodies; total mass and neutral world poses are preserved; Pinocchio
     builds the model with zero bilateral weld constraints.
2. Physical hand-hand contact in ``topology_contact.py`` (``mode="separated_v2"``):
   - Default ``mode="contact"`` still rejects hand-hand pairs in ``topology_pair_wrench``
     and records ``NaN`` + ``"allocated"`` reason in ``PairTraceRecorder``.
   - In ``mode="separated_v2"``: evaluates ``trail_palm_lead_thumb`` and style-specific
     finger pairs (``overlap_little_finger_lead_index``,
     ``interlock_little_finger_lead_middle``, ``interlock_lead_index_wrap``) with the
     shared ``pair_contact`` kernels.
   - Physics invariants: zero force at separation, unilateral normal (``f_n >= 0``),
     Coulomb friction cone (``||f_t|| <= mu * f_n``), and exact action-reaction
     (``<= 1e-9 N`` force and ``<= 1e-9 N*m`` moment about a common world point)
     between lead and trail hands.
3. Identifiability note in docstrings, constants, and spec metadata: finger contact is
   not observable from markers (no finger markers); only the net club wrench is
   identified, while individual pair forces depend on retention and contact parameters.
4. Committed synthetic swing test (no private data) and private best-fit swing test
   (reading ``UD_BEST_FIT_BUNDLE`` and ``UD_BEST_FIT_SPEC`` from environment variables
   and skipping cleanly when unset):
   - Separated-hand ``v2`` net closure wrench reproduces ``v1`` closure wrench within
     5 % peak and 2 % RMS;
   - Per-pair normal-force contact/no-contact transitions occur in < 1 % of steps.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.biomechanics.club_inertia_partition import club_solids_partition
from src.shared.python.grip_contact import (
    BushingState,
    CoordinateSwing,
    GripInterface,
)
from src.shared.python.grip_contact.contact_topology import (
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.parity import (
    PEAK_TOLERANCE,
    RMS_TOLERANCE,
    parity_errors,
)
from src.shared.python.grip_contact.topology_contact import (
    HAND_HAND_IDENTIFIABILITY_NOTE,
    PairTraceRecorder,
    TopologyContactModel,
    TopologyHandHandWrench,
    build_topology_model,
    contact_transition_fraction,
    topology_hand_hand_pair_wrench,
    topology_hand_hand_wrenches,
    topology_pair_wrench,
    topology_side_wrench,
)
from src.shared.python.model_appearance.skinned_body import spec_body_poses
from src.shared.python.motion_matching.full_body_migration import TrunkMigrator
from src.shared.python.motion_matching.full_body_spec import canonical_sha256
from src.shared.python.motion_matching.full_body_v2 import (
    LEFT_HAND_BODY,
    derive_full_body_v2_spec,
    segment_boundary_locations,
    separate_v2_hands,
    validate_full_body_v2_spec,
)
from src.shared.python.motion_matching.grip_contract import GripSpec

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
ANTHRO_DRIVER_PATH = MODELS / "full_body_spec_anthro_driver.json"

CHATTER_TOLERANCE = 0.01
ACTION_REACTION_TOL_N = 1.0e-9


def _spec_frame_poses(spec: dict) -> dict[str, np.ndarray]:
    bposes = spec_body_poses(spec)
    return {
        f["name"]: bposes[f["body"]] @ np.asarray(f["placement"], dtype=float)
        for f in spec["frames"]
    }


@pytest.fixture(scope="module")
def v1_driver_spec() -> dict:
    """Load the committed anthropometric v1 driver spec."""
    return json.loads(ANTHRO_DRIVER_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def v2_default_spec(v1_driver_spec: dict) -> dict:
    """Derive the default (welded-hand) full-body-v2 spec."""
    locs = segment_boundary_locations(v1_driver_spec)
    return derive_full_body_v2_spec(
        v1_driver_spec, locs, provenance="HRG-GS-G11 test default v2"
    )


@pytest.fixture(scope="module")
def v2_separated_spec(v1_driver_spec: dict) -> dict:
    """Derive the opt-in separated-hand full-body-v2 spec."""
    locs = segment_boundary_locations(v1_driver_spec)
    return derive_full_body_v2_spec(
        v1_driver_spec,
        locs,
        provenance="HRG-GS-G11 test separated v2",
        separate_hands=True,
    )


@pytest.mark.unit
def test_default_v1_and_v2_specs_unchanged_without_opt_in(
    v1_driver_spec: dict, v2_default_spec: dict
) -> None:
    """v1 bytes and default v2 derivation stay immutable when separate_hands is False."""
    v1_sha_before = canonical_sha256(v1_driver_spec)
    locs = segment_boundary_locations(v1_driver_spec)
    v2_explicit_false = derive_full_body_v2_spec(
        v1_driver_spec,
        locs,
        provenance="HRG-GS-G11 test default v2",
        separate_hands=False,
    )
    assert canonical_sha256(v1_driver_spec) == v1_sha_before
    assert canonical_sha256(v2_explicit_false) == canonical_sha256(v2_default_spec)
    assert "grip_bushing" not in v2_default_spec
    assert "hand_separation" not in v2_default_spec


@pytest.mark.unit
def test_opt_in_separate_hands_creates_distinct_hand_and_club_bodies_without_weld(
    v2_default_spec: dict, v2_separated_spec: dict
) -> None:
    """Opt-in separate_hands=True separates LHandBody, RHandStandoff, and bare club."""
    assert v2_separated_spec["hand_separation"]["enabled"] is True
    assert v2_separated_spec["hand_separation"]["mode"] == "separated_v2"
    assert v2_separated_spec["hand_separation"]["left_hand_body"] == LEFT_HAND_BODY
    assert "grip_bushing" in v2_separated_spec

    # Direct separate_v2_hands(v2_default_spec) produces the same topology
    via_helper = separate_v2_hands(v2_default_spec)
    assert via_helper["hand_separation"] == v2_separated_spec["hand_separation"]
    validate_full_body_v2_spec(
        v2_separated_spec,
        {
            "schema_version": v2_separated_spec["upper_body_schema_version"],
            "qualification": v2_separated_spec["upper_body_qualification"],
            "gravity_m_s2": v2_separated_spec["gravity_m_s2"],
            "coordinate_order": list(
                v2_separated_spec["coordinate_order"][
                    : v2_separated_spec["upper_body_counts"]["coordinates"]
                ]
            ),
            "bodies": v2_separated_spec["bodies"][
                : v2_separated_spec["upper_body_counts"]["bodies"]
            ],
            "joints": v2_separated_spec["joints"][
                : v2_separated_spec["upper_body_counts"]["joints"]
            ],
            "frames": v2_separated_spec["frames"],
            "closure": v2_separated_spec["closure"],
        },
    )

    bodies = {b["name"]: b for b in v2_separated_spec["bodies"]}
    right_body = v2_separated_spec["closure"]["body_a"]
    club_body = v2_separated_spec["closure"]["body_b"]
    assert LEFT_HAND_BODY in bodies
    assert right_body in bodies
    assert club_body in bodies

    left_solids = {
        s["name"].rsplit("/", 1)[-1] for s in bodies[LEFT_HAND_BODY]["solids"]
    }
    right_solids = {s["name"].rsplit("/", 1)[-1] for s in bodies[right_body]["solids"]}
    club_solids = {s["name"].rsplit("/", 1)[-1] for s in bodies[club_body]["solids"]}
    assert left_solids == {"LHand", "LHandStandoff"}
    assert "RHand" in right_solids
    assert not (club_solids & {"LHand", "LHandStandoff", "RHand"})
    assert {"Clubhead", "Grip", "Rigid Shaft"}.issubset(club_solids)

    # Total mass is conserved to <= 1e-9 kg
    mass_default = sum(
        s["mass_kg"] for b in v2_default_spec["bodies"] for s in b["solids"]
    )
    mass_sep = sum(
        s["mass_kg"] for b in v2_separated_spec["bodies"] for s in b["solids"]
    )
    assert abs(mass_sep - mass_default) <= 1.0e-9

    # Bare club partition and GripInterface match the unseparated spec
    part_def = club_solids_partition(v2_default_spec)
    part_sep = club_solids_partition(v2_separated_spec)
    assert abs(part_sep.bare_club.mass_kg - part_def.bare_club.mass_kg) <= 1.0e-12
    np.testing.assert_allclose(
        part_sep.bare_club.com_m, part_def.bare_club.com_m, atol=1.0e-12
    )
    np.testing.assert_allclose(
        part_sep.bare_club.inertia_com_kg_m2,
        part_def.bare_club.inertia_com_kg_m2,
        atol=1.0e-12,
    )
    gi_def = GripInterface.from_spec(v2_default_spec)
    gi_sep = GripInterface.from_spec(v2_separated_spec)
    np.testing.assert_allclose(gi_sep.left.matrix(), gi_def.left.matrix(), atol=1.0e-12)
    np.testing.assert_allclose(
        gi_sep.right.matrix(), gi_def.right.matrix(), atol=1.0e-12
    )

    # Neutral (q = 0) world poses of all bodies and frames match to <= 1e-12 m
    poses_def = spec_body_poses(v2_default_spec)
    poses_sep = spec_body_poses(v2_separated_spec)
    for name, pose in poses_def.items():
        np.testing.assert_allclose(poses_sep[name], pose, atol=1.0e-12)
    np.testing.assert_allclose(
        poses_sep[LEFT_HAND_BODY], poses_def[club_body], atol=1.0e-12
    )
    fposes_def = _spec_frame_poses(v2_default_spec)
    fposes_sep = _spec_frame_poses(v2_separated_spec)
    for name, fpose in fposes_def.items():
        np.testing.assert_allclose(fposes_sep[name], fpose, atol=1.0e-12)


@pytest.mark.unit
@pytest.mark.requires_pinocchio
def test_pinocchio_separated_v2_model_has_no_weld_and_evaluates_hand_wrenches(
    v2_separated_spec: dict,
) -> None:
    """FullBodyPinocchioModel on a separated_v2 spec has 0 bilateral welds and finite dynamics."""
    pytest.importorskip("pinocchio")
    from src.engines.physics_engines.pinocchio.python.native_model import (
        FullBodyPinocchioModel,
    )

    model = FullBodyPinocchioModel(v2_separated_spec)
    assert len(model.constraints) == 0
    assert len(model.constraint_data) == 0

    coords = dict.fromkeys(model.coordinate_order, 0.0)
    rates = dict.fromkeys(model.coordinate_order, 0.0)
    efforts = dict.fromkeys(model.coordinate_order, 0.0)
    acc = model.accelerations(coords, rates, efforts)
    assert set(acc) == set(model.coordinate_order)
    assert all(np.isfinite(v) for v in acc.values())

    grip_state = model.separated_grip_wrenches(coords, rates)
    assert set(grip_state["hand_on_club"]) == {"L", "R"}
    hh = grip_state["hand_hand"]
    assert isinstance(hh, TopologyHandHandWrench)
    assert (
        np.linalg.norm(hh.force_on_lead_n + hh.force_on_trail_n)
        <= ACTION_REACTION_TOL_N
    )


@pytest.mark.unit
def test_default_contact_mode_still_rejects_hand_hand_pairs(
    v1_driver_spec: dict,
) -> None:
    """Default mode='contact' preserves the G6 contract rejecting hand-hand pairs."""
    assert not hasattr(TopologyContactModel, "from_Spec")
    assert not hasattr(TopologyContactModel, "hand_hand_pair_primitives")
    interface = GripInterface.from_spec(v1_driver_spec)
    topo = contact_topology(GripSpec(style=GripStyle.OVERLAP))
    model = build_topology_model(interface, topology=topo)
    st_l = BushingState(
        np.asarray(interface.left.rotation, float),
        np.asarray(interface.left.position_m, float),
        np.zeros(3),
        np.zeros(3),
    )
    st_r = BushingState(
        np.asarray(interface.right.rotation, float),
        np.asarray(interface.right.position_m, float),
        np.zeros(3),
        np.zeros(3),
    )
    hh_pair = topo.hand_hand_pairs()[0]
    with pytest.raises(ValueError, match="hand-hand"):
        topology_pair_wrench(model, hh_pair, st_l, st_r)

    recorder = PairTraceRecorder(model)
    club = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))
    frames = {"L": st_l, "R": st_r}
    law = {s: topology_side_wrench(model, s, frames[s], club) for s in ("L", "R")}
    recorder.record(law, frames, frames)
    res = recorder.result()
    assert np.all(np.isnan(res["pair_normal_force_n"][hh_pair.name]))
    assert "allocated" in res["channel_reasons"][hh_pair.name]


@pytest.mark.unit
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_separated_v2_hand_hand_physics_contracts(
    v1_driver_spec: dict, style: GripStyle
) -> None:
    """Zero force at separation, unilateral normal, friction cone, and action-reaction <= 1e-9 N."""
    interface = GripInterface.from_spec(v1_driver_spec)
    topo = contact_topology(GripSpec(style=style))
    model = build_topology_model(interface, topology=topo)

    st_l_neutral = BushingState(
        np.asarray(interface.left.rotation, float),
        np.asarray(interface.left.position_m, float),
        np.zeros(3),
        np.zeros(3),
    )
    st_r_neutral = BushingState(
        np.asarray(interface.right.rotation, float),
        np.asarray(interface.right.position_m, float),
        np.zeros(3),
        np.zeros(3),
    )

    # 1. Neutral pose with tangential relative velocity: active unilateral compression,
    # friction cone respected, and exact action-reaction between lead and trail hands.
    st_r_sliding = BushingState(
        st_r_neutral.rotation,
        st_r_neutral.position_m,
        np.array([0.15, -0.20, 0.05]),
        np.array([0.5, -0.3, 0.2]),
    )
    hh_active = topology_hand_hand_wrenches(
        model, st_l_neutral, st_r_sliding, mode="separated_v2"
    )
    assert isinstance(hh_active, TopologyHandHandWrench)
    assert set(hh_active.pairs) == {p.name for p in topo.hand_hand_pairs()}
    mu = float(model.parameters.static_friction)

    role_frames = {"lead": st_l_neutral, "trail": st_r_sliding}
    for pair in topo.hand_hand_pairs():
        pw = hh_active.pairs[pair.name]
        fa = role_frames[topo.site(pair.site_a).hand]
        fb = role_frames[topo.site(pair.site_b).hand]
        direct_pw = topology_hand_hand_pair_wrench(
            model, pair, fa, fb, mode="separated_v2"
        )
        np.testing.assert_allclose(pw.force_n, direct_pw.force_n, atol=1.0e-15)
        np.testing.assert_allclose(pw.moment_nm, direct_pw.moment_nm, atol=1.0e-15)
        assert pw.normal_force_n == pytest.approx(direct_pw.normal_force_n, abs=1.0e-15)
        assert pw.normal_force_n > 0.0, (
            f"{pair.name} must be in compression at neutral pose"
        )
        assert pw.tangential_force_n >= 0.0
        assert pw.tangential_force_n <= mu * pw.normal_force_n * (1.0 + 1.0e-9)

    # Equal and opposite forces on lead vs trail hands (<= 1e-9 N)
    assert (
        np.linalg.norm(hh_active.force_on_lead_n + hh_active.force_on_trail_n)
        <= ACTION_REACTION_TOL_N
    )
    # Total moment about any common world point (e.g. origin) cancels to <= 1e-9 N*m
    net_moment_world = (
        hh_active.moment_on_lead_nm
        + hh_active.moment_on_trail_nm
        + np.cross(st_l_neutral.position_m, hh_active.force_on_lead_n)
        + np.cross(st_r_sliding.position_m, hh_active.force_on_trail_n)
    )
    assert np.linalg.norm(net_moment_world) <= ACTION_REACTION_TOL_N

    # 2. Separated hands (+50 mm apart): zero force and zero moment on every pair
    st_r_separated = BushingState(
        st_r_neutral.rotation,
        st_r_neutral.position_m + np.array([0.05, 0.05, 0.05]),
        np.array([0.2, -0.1, 0.3]),
        np.zeros(3),
    )
    hh_sep = topology_hand_hand_wrenches(
        model, st_l_neutral, st_r_separated, mode="separated_v2"
    )
    for pair in topo.hand_hand_pairs():
        pw = hh_sep.pairs[pair.name]
        assert pw.normal_force_n == 0.0
        assert pw.tangential_force_n == 0.0
        np.testing.assert_allclose(pw.force_n, np.zeros(3), atol=1.0e-15)
        np.testing.assert_allclose(pw.moment_nm, np.zeros(3), atol=1.0e-15)
    np.testing.assert_allclose(hh_sep.force_on_lead_n, np.zeros(3), atol=1.0e-15)
    np.testing.assert_allclose(hh_sep.force_on_trail_n, np.zeros(3), atol=1.0e-15)
    np.testing.assert_allclose(hh_sep.moment_on_lead_nm, np.zeros(3), atol=1.0e-15)
    np.testing.assert_allclose(hh_sep.moment_on_trail_nm, np.zeros(3), atol=1.0e-15)

    # 3. PairTraceRecorder in mode="separated_v2" records finite physical values
    recorder = PairTraceRecorder(model, mode="separated_v2")
    frames = {"L": st_l_neutral, "R": st_r_neutral}
    law = {s: topology_side_wrench(model, s, frames[s], frames[s]) for s in ("L", "R")}
    recorder.record(law, {"L": st_l_neutral, "R": st_r_sliding}, frames)
    res = recorder.result()
    assert res["channel_reasons"] == {}
    for pair in topo.pairs:
        assert np.all(np.isfinite(res["pair_normal_force_n"][pair.name]))
        assert np.all(np.isfinite(res["pair_tangential_force_n"][pair.name]))
        assert np.all(np.isfinite(res["pair_contact_point_drift_m"][pair.name]))


@pytest.mark.unit
def test_identifiability_note_present_in_code_and_spec(
    v2_separated_spec: dict,
) -> None:
    """Identifiability note documents that finger contact is unobservable from markers."""
    note = HAND_HAND_IDENTIFIABILITY_NOTE.lower()
    assert "finger" in note and "marker" in note
    assert "net" in note and "wrench" in note
    assert "retention" in note and "contact" in note
    spec_note = v2_separated_spec["hand_separation"]["identifiability_note"]
    assert spec_note == HAND_HAND_IDENTIFIABILITY_NOTE


@pytest.mark.unit
def test_hrg_tex_g11_has_no_private_hashes_or_paths() -> None:
    """G11 subsection of human_realistic_golfer.tex contains no private capture hashes or paths."""
    import re

    tex_path = ROOT / "docs/research/human_realistic_golfer/human_realistic_golfer.tex"
    tex = tex_path.read_text(encoding="utf-8")
    start = tex.index(r"\subsection{Separated Hands and Physical Hand--Hand Contact")
    end = tex.index(r"\subsection{Model Grip Binding Contract", start)
    g11_sec = tex[start:end]
    assert (
        "the private best-fit driver capture (hash recorded in the private receipt)"
        in g11_sec
    )
    assert "bundle SHA256" not in g11_sec
    assert "spec SHA256" not in g11_sec
    assert re.search(r"\b[0-9a-f]{8,64}\b", g11_sec) is None
    assert re.search(r"/home/", g11_sec) is None


def _write_grip_receipt(
    out_path_str: str,
    *,
    time_s: np.ndarray,
    dt_s: float,
    net_force_err: object,
    net_moment_err: object,
    pair_normal_forces: dict[str, np.ndarray],
    max_action_reaction_error_n: float,
    bundle_sha256: str = "",
    spec_sha256: str = "",
) -> dict:
    """Build and optionally write the G11 acceptance receipt outside the repository."""
    n_samples = int(time_s.size)
    n_steps = max(0, n_samples - 1)
    pair_summary: dict[str, dict[str, float | int | bool]] = {}
    for pname, fn_arr in pair_normal_forces.items():
        arr = np.asarray(fn_arr, dtype=float)
        frac = contact_transition_fraction(arr)
        active = arr > 0.0
        transitions = (
            int(np.count_nonzero(active[1:] != active[:-1])) if arr.size >= 2 else 0
        )
        pair_summary[pname] = {
            "peak_normal_force_n": float(np.max(arr)),
            "transition_count": transitions,
            "step_count": n_steps,
            "transition_fraction": float(frac),
            "passes_chatter": bool(frac < CHATTER_TOLERANCE),
        }

    receipt = {
        "issue": 13309,
        "sample_count": n_samples,
        "step_count": n_steps,
        "dt_s": float(dt_s),
        "duration_s": float(time_s[-1] - time_s[0]),
        "bundle_sha256": bundle_sha256,
        "spec_sha256": spec_sha256,
        "net_force": net_force_err.to_dict(),  # type: ignore[attr-defined]
        "net_moment_com": net_moment_err.to_dict(),  # type: ignore[attr-defined]
        "max_action_reaction_error_n": float(max_action_reaction_error_n),
        "pairs": pair_summary,
    }
    if out_path_str:
        out_path = Path(out_path_str).resolve()
        if ROOT in out_path.parents or out_path == ROOT:
            raise ValueError(
                f"UD_GRIP_RECEIPT_OUT must point outside the repository ({ROOT}), got {out_path}"
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


@pytest.mark.unit
@pytest.mark.requires_pinocchio
@pytest.mark.timeout(600)
def test_synthetic_swing_separated_v2_closure_wrench_and_no_chatter(
    v1_driver_spec: dict, v2_separated_spec: dict, tmp_path: Path
) -> None:
    """Synthetic committed swing: separated-hand v2 matches v1 wrench within 5%/2% with <1% chatter."""
    pytest.importorskip("pinocchio")
    from src.engines.physics_engines.pinocchio.python.grip_bushing import (
        simulate_grip_bushing,
        simulate_separated_v2_grip,
    )
    from src.engines.physics_engines.pinocchio.python.native_model import (
        FullBodyPinocchioModel,
    )

    order1 = list(v1_driver_spec["coordinate_order"])
    m1 = FullBodyPinocchioModel(v1_driver_spec)
    q0_map = dict.fromkeys(order1, 0.0)
    arm_coords = [
        c
        for c in order1
        if c.startswith(("LS", "LE", "LF", "LW", "RS", "RE", "RF", "RW"))
    ]
    for _ in range(15):
        lin = m1.closure_position_linearization(q0_map)
        if float(np.linalg.norm(lin.position)) < 1.0e-13:
            break
        arm_idx = [lin.names.index(c) for c in arm_coords]
        dq = np.linalg.pinv(lin.jacobian[:, arm_idx]) @ lin.position
        for c, d in zip(arm_coords, dq, strict=True):
            q0_map[c] -= float(d)

    n_steps = 8
    time_s = np.arange(n_steps, dtype=float) * 0.001
    q0_row = np.array([q0_map[c] for c in order1], dtype=float)
    q_v1 = np.tile(q0_row, (n_steps, 1))
    # C2-smooth whole-body pelvis translation + axial rotation (zero initial velocity/accel)
    envelope = (10.0 * time_s) ** 3
    for idx, name in enumerate(order1):
        if name == "TranslationInputX":
            q_v1[:, idx] += 0.20 * envelope
        elif name == "TranslationInputZ":
            q_v1[:, idx] += 0.15 * envelope
        elif name in ("PelvisRotInputZ", " PelvisRotInputZ"):
            q_v1[:, idx] += 1.50 * envelope

    swing_v1 = CoordinateSwing(
        names=order1,
        time_s=time_s,
        q=q_v1,
        sha256="0" * 64,
    )
    interface = GripInterface.from_spec(v1_driver_spec)
    ref_series = simulate_grip_bushing(
        json.dumps(v1_driver_spec).encode("utf-8"),
        swing_v1,
        interface=interface,
    )

    migrator = TrunkMigrator(
        v1_driver_spec,
        derive_full_body_v2_spec(
            v1_driver_spec,
            segment_boundary_locations(v1_driver_spec),
            provenance="synthetic migration",
        ),
    )
    q_v2 = migrator.migrate_coordinates(q_v1, order1)
    swing_v2 = CoordinateSwing(
        names=list(migrator._v2_order),  # noqa: SLF001
        time_s=time_s,
        q=q_v2,
        sha256="1" * 64,
    )

    sep_run = simulate_separated_v2_grip(
        json.dumps(v2_separated_spec).encode("utf-8"),
        swing_v2,
        interface=interface,
        style=GripStyle.OVERLAP,
    )

    errors = parity_errors(sep_run.series, ref_series)
    m_err = sep_run.net_moment_com_error(ref_series)
    assert errors["net_force"].peak_error <= PEAK_TOLERANCE
    assert errors["net_force"].rms_error <= RMS_TOLERANCE
    assert m_err.peak_error <= PEAK_TOLERANCE
    assert m_err.rms_error <= RMS_TOLERANCE

    # No chatter: per-pair normal-force sign changes occur in < 1% of steps
    for pair_name, fn_series in sep_run.pair_normal_force_n.items():
        frac = contact_transition_fraction(fn_series)
        assert frac < CHATTER_TOLERANCE, (
            f"pair {pair_name} transition fraction {frac:.4f} >= {CHATTER_TOLERANCE}"
        )

    # Action-reaction <= 1e-9 N at every sample
    assert sep_run.max_action_reaction_error_n <= ACTION_REACTION_TOL_N

    receipt_file = tmp_path / "synthetic_receipt.json"
    receipt = _write_grip_receipt(
        str(receipt_file),
        time_s=time_s,
        dt_s=0.001,
        net_force_err=errors["net_force"],
        net_moment_err=m_err,
        pair_normal_forces=dict(sep_run.pair_normal_force_n),
        max_action_reaction_error_n=sep_run.max_action_reaction_error_n,
    )
    assert receipt_file.is_file()
    loaded = json.loads(receipt_file.read_text(encoding="utf-8"))
    assert loaded["sample_count"] == n_steps
    assert loaded["step_count"] == n_steps - 1
    assert loaded["net_force"]["passes"] is True
    assert loaded["net_moment_com"]["passes"] is True
    assert "trail_palm_lead_thumb" in receipt["pairs"]


@pytest.mark.unit
@pytest.mark.requires_pinocchio
@pytest.mark.timeout(1800)
def test_best_fit_swing_separated_v2_acceptance() -> None:
    """Private best-fit swing acceptance via UD_BEST_FIT_BUNDLE and UD_BEST_FIT_SPEC."""
    pytest.importorskip("pinocchio")
    bundle_env = os.environ.get("UD_BEST_FIT_BUNDLE", "").strip()
    spec_env = os.environ.get("UD_BEST_FIT_SPEC", "").strip()
    if not bundle_env or not spec_env:
        pytest.skip(
            "UD_BEST_FIT_BUNDLE and UD_BEST_FIT_SPEC environment variables are unset; "
            "skipping private best-fit swing acceptance test"
        )
    bundle_path = Path(bundle_env)
    spec_path = Path(spec_env)
    if not bundle_path.is_file() or not spec_path.is_file():
        pytest.skip(
            "Private capture files referenced by UD_BEST_FIT_BUNDLE / UD_BEST_FIT_SPEC "
            "are not present on disk"
        )

    import hashlib

    from src.engines.physics_engines.pinocchio.python.grip_bushing import (
        SeparatedHandsKinematics,
        _build_se3_hand_source,
        simulate_grip_bushing,
        simulate_separated_v2_grip,
    )
    from src.shared.python.grip_contact import CoordinateSpline

    bundle_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    spec_bytes = spec_path.read_bytes()
    spec_sha = hashlib.sha256(spec_bytes).hexdigest()

    v1_spec = json.loads(spec_bytes.decode("utf-8"))
    locs = segment_boundary_locations(v1_spec)
    v2_base = derive_full_body_v2_spec(
        v1_spec, locs, provenance="HRG-GS-G11 best-fit v2"
    )
    v2_sep = derive_full_body_v2_spec(
        v1_spec,
        locs,
        provenance="HRG-GS-G11 best-fit separated v2",
        separate_hands=True,
    )

    with np.load(bundle_path, allow_pickle=True) as data:
        q_v1 = np.asarray(data["reference_q"], dtype=float)
        manifest = json.loads(str(data["manifest"]))
    dt_s = float(manifest["dt_s"])
    time_s = np.arange(q_v1.shape[0], dtype=float) * dt_s
    order1 = list(v1_spec["coordinate_order"])

    migrator = TrunkMigrator(v1_spec, v2_base)
    q_v2 = migrator.migrate_coordinates(q_v1, order1)
    order2 = list(v2_base["coordinate_order"])

    swing_v1 = CoordinateSwing(names=order1, time_s=time_s, q=q_v1, sha256="0" * 64)
    swing_v2 = CoordinateSwing(names=order2, time_s=time_s, q=q_v2, sha256="1" * 64)
    interface = GripInterface.from_spec(v1_spec)

    v1_spec_bytes = json.dumps(v1_spec).encode("utf-8")
    v2_sep_bytes = json.dumps(v2_sep).encode("utf-8")
    ref_series = simulate_grip_bushing(
        v1_spec_bytes,
        swing_v1,
        interface=interface,
    )
    sep_run = simulate_separated_v2_grip(
        v2_sep_bytes,
        swing_v2,
        interface=interface,
        style=GripStyle.OVERLAP,
    )

    # Also evaluate interlock hand-hand pairs on the same separated-v2 hand trajectory
    # so the receipt records every hand-hand pair reported in the HRG table.
    kin = SeparatedHandsKinematics(v2_sep, order2)
    spline = CoordinateSpline(time_s, q_v2)
    hand_eval = _build_se3_hand_source(kin, spline, time_s)
    interlock_topo = contact_topology(GripSpec(style=GripStyle.INTERLOCK))
    interlock_model = build_topology_model(interface, topology=interlock_topo)
    all_pair_normal: dict[str, np.ndarray] = dict(sep_run.pair_normal_force_n)
    missing_hh = [
        p for p in interlock_topo.hand_hand_pairs() if p.name not in all_pair_normal
    ]
    for p in missing_hh:
        all_pair_normal[p.name] = np.zeros(time_s.size, dtype=float)
    for k, t_val in enumerate(time_s):
        hs = hand_eval(float(t_val))
        hh_il = topology_hand_hand_wrenches(
            interlock_model, hs["L"], hs["R"], mode="separated_v2"
        )
        for p in missing_hh:
            all_pair_normal[p.name][k] = hh_il.pairs[p.name].normal_force_n

    errs = parity_errors(sep_run.series, ref_series)
    m_err = sep_run.net_moment_com_error(ref_series)

    receipt_out = os.environ.get("UD_GRIP_RECEIPT_OUT", "").strip()
    _write_grip_receipt(
        receipt_out,
        time_s=time_s,
        dt_s=dt_s,
        net_force_err=errs["net_force"],
        net_moment_err=m_err,
        pair_normal_forces=all_pair_normal,
        max_action_reaction_error_n=sep_run.max_action_reaction_error_n,
        bundle_sha256=bundle_sha,
        spec_sha256=spec_sha,
    )

    assert errs["net_force"].peak_error <= PEAK_TOLERANCE, (
        f"net_force peak error {errs['net_force'].peak_error:.4%} > {PEAK_TOLERANCE:.1%}"
    )
    assert errs["net_force"].rms_error <= RMS_TOLERANCE, (
        f"net_force RMS error {errs['net_force'].rms_error:.4%} > {RMS_TOLERANCE:.1%}"
    )
    assert m_err.peak_error <= PEAK_TOLERANCE, (
        f"net_moment_com peak error {m_err.peak_error:.4%} > {PEAK_TOLERANCE:.1%}"
    )
    assert m_err.rms_error <= RMS_TOLERANCE, (
        f"net_moment_com RMS error {m_err.rms_error:.4%} > {RMS_TOLERANCE:.1%}"
    )

    for pair_name, fn_series in all_pair_normal.items():
        frac = contact_transition_fraction(fn_series)
        assert frac < CHATTER_TOLERANCE, (
            f"pair {pair_name} transition fraction {frac:.4%} >= {CHATTER_TOLERANCE:.1%}"
        )

    assert sep_run.max_action_reaction_error_n <= ACTION_REACTION_TOL_N
