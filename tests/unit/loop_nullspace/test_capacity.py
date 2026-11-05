"""Torque-capacity loader tests for NSX-04 (#12553)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.shared.python.loop_nullspace.capacity import (
    CapacitySubject,
    CapacityVector,
    capacity_for,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPECS = ROOT / "docs" / "development" / "full_body_models"
SPEC_FILES = ("full_body_spec_anthro_driver.json", "full_body_spec_anthro_iron7.json")
G = 9.80665


def test_every_unverified_slot_has_a_specific_reason() -> None:
    registry = json.loads(
        (ROOT / "src/config/loop_nullspace_registry.json").read_text("utf-8")
    )
    unresolved = [
        entry
        for entry in registry["torque_capacity"]
        if entry["verification_status"] != "VERIFIED"
    ]
    for entry in unresolved:
        label = f"{entry['joint_slot']}/{entry['direction']}"
        note = entry["source"]["note"]
        assert note.strip() and note != "No source read for this slot.", label
        assert entry["rationale"].strip(), label
        assert entry["assumed"] is True, label
    slots = tuple(dict.fromkeys(entry["joint_slot"] for entry in unresolved))
    if slots:
        vector = capacity_for(slots, CapacitySubject(78.0, 1.71))
        assert set(vector.assumed_slots) == set(slots)
        assert not vector.verified_slots
        assert vector.warnings and "ASSUMED" in vector.warnings[0]


def _spec(name: str) -> dict:
    return json.loads((SPECS / name).read_text(encoding="utf-8"))


@pytest.fixture(params=SPEC_FILES)
def spec(request: pytest.FixtureRequest) -> dict:
    return _spec(request.param)


def _subject(spec: dict) -> CapacitySubject:
    return CapacitySubject.from_spec_subject(spec["subject"])


def test_covers_all_44_coordinates_of_both_specs(spec: dict) -> None:
    order = tuple(spec["coordinate_order"])
    assert len(order) == 44
    vec = capacity_for(order, _subject(spec))
    assert isinstance(vec, CapacityVector)
    # Root translations are forces, not joint torques.
    assert len(vec.actuated_names) == 41
    assert all(not n.startswith("Translation") for n in vec.actuated_names)
    for arr in (vec.tau_max_negative_nm, vec.tau_max_positive_nm):
        assert arr.shape == (41,)
        assert (arr > 0).all() and bool(arr.dtype.kind == "f")
        assert all(math.isfinite(float(v)) for v in arr)
    assert [order[i] for i in vec.actuated_indices] == list(vec.actuated_names)


def test_result_is_frozen_and_read_only(spec: dict) -> None:
    vec = capacity_for(tuple(spec["coordinate_order"]), _subject(spec))
    with pytest.raises(AttributeError):
        vec.assumed_slots = ()  # type: ignore[misc]
    with pytest.raises(ValueError):
        vec.tau_max_positive_nm[0] = 1.0


def test_assumed_slots_are_surfaced_not_silent(spec: dict) -> None:
    vec = capacity_for(tuple(spec["coordinate_order"]), _subject(spec))
    assert vec.has_assumed
    assert set(vec.assumed_slots) | set(vec.verified_slots) == set(vec.actuated_names)
    assert not set(vec.assumed_slots) & set(vec.verified_slots)
    assert "TorsoInput" in vec.verified_slots
    assert "LEInput" in vec.verified_slots and "knee_angle_r" in vec.verified_slots
    # Neck and sagittal spine axes resolved from the model frames (NSX-04 review).
    for name in ("NeckInputX", "NeckInputY", "NeckInputZ", "SpineInputY", "LWInputY"):
        assert name in vec.verified_slots, name
    for name in ("SpineInputX", "LSInputX", "LSInputY", "LSInputZ", "LFInput"):
        assert name in vec.assumed_slots, name
    assert "hip_rotation_r" in vec.assumed_slots
    assert vec.warnings and "ASSUMED" in vec.warnings[0]


def test_missing_slots_raise_listing_each(spec: dict) -> None:
    order = tuple(spec["coordinate_order"]) + ("PhantomInputA", "PhantomInputB")
    with pytest.raises(ValueError) as exc:
        capacity_for(order, _subject(spec))
    msg = str(exc.value)
    assert "PhantomInputA" in msg and "PhantomInputB" in msg


def test_unactuated_coordinate_need_not_be_covered(spec: dict) -> None:
    order = tuple(spec["coordinate_order"]) + ("PhantomInputA",)
    vec = capacity_for(order, _subject(spec), actuated=spec["coordinate_order"][3:])
    assert "PhantomInputA" not in vec.actuated_names


def test_preconditions(spec: dict) -> None:
    subj = _subject(spec)
    order = tuple(spec["coordinate_order"])
    with pytest.raises(ValueError, match="empty"):
        capacity_for((), subj)
    with pytest.raises(ValueError, match="duplicate"):
        capacity_for(order + (order[5],), subj)
    with pytest.raises(ValueError, match="not in coordinate_order"):
        capacity_for(order, subj, actuated=("Nope",))
    with pytest.raises(ValueError, match="mass_kg"):
        CapacitySubject(mass_kg=-1.0, stature_m=1.7)
    with pytest.raises(ValueError, match="stature_m"):
        CapacitySubject(mass_kg=70.0, stature_m=float("nan"))
    with pytest.raises(ValueError, match="sex"):
        capacity_for(order, CapacitySubject(70.0, 1.7, sex="female"))
    with pytest.raises(TypeError):
        capacity_for(order, {"mass_kg": 70.0})  # type: ignore[arg-type]


def test_anderson_scaling_reproduces_cited_example() -> None:
    """Anderson 2007 Table 2 mean young male (72.8 kg, 174.8 cm) times Table 3
    C1 = 0.163 (knee extension) gives 0.163 * 72.8 * g * 1.748 N m."""
    expected = 0.163 * 72.8 * G * 1.748
    vec = capacity_for(
        ("knee_angle_r",), CapacitySubject(72.8, 1.748), actuated=("knee_angle_r",)
    )
    assert vec.tau_max_positive_nm[0] == pytest.approx(expected, rel=1e-9)
    # Extension torque sits inside one SD of the independent Leblanc 2015
    # men's isometric knee extension mean 235.6 (SD 63.4).
    assert abs(vec.tau_max_positive_nm[0] - 235.6) < 63.4


def test_anderson_scaling_is_proportional_to_weight_times_height() -> None:
    a = capacity_for(("hip_flexion_l",), CapacitySubject(60.0, 1.60))
    b = capacity_for(("hip_flexion_l",), CapacitySubject(90.0, 1.80))
    ratio = (90.0 * 1.80) / (60.0 * 1.60)
    assert b.tau_max_positive_nm[0] / a.tau_max_positive_nm[0] == pytest.approx(ratio)
    assert b.tau_max_negative_nm[0] / a.tau_max_negative_nm[0] == pytest.approx(ratio)


def test_unscaled_population_values_are_returned_verbatim() -> None:
    vec = capacity_for(("TorsoInput", "LEInput"), CapacitySubject(50.0, 1.5))
    assert vec.tau_max_negative_nm[0] == pytest.approx(68.7)
    assert vec.tau_max_positive_nm[0] == pytest.approx(68.7)
    # Elbow: negative = flexion 67.9, positive = extension 59.6 (Leblanc 2015).
    assert vec.tau_max_negative_nm[1] == pytest.approx(67.9)
    assert vec.tau_max_positive_nm[1] == pytest.approx(59.6)


def test_reference_subject_reproduces_stored_reference_values(spec: dict) -> None:
    ref = CapacitySubject(78.0, 1.71)
    vec = capacity_for(tuple(spec["coordinate_order"]), ref)
    reg = json.loads(
        (ROOT / "src" / "config" / "loop_nullspace_registry.json").read_text("utf-8")
    )
    stored = {
        (e["joint_slot"], e["direction"]): e["tau_max_nm"]
        for e in reg["torque_capacity"]
    }
    for i, name in enumerate(vec.actuated_names):
        assert vec.tau_max_negative_nm[i] == pytest.approx(
            stored[(name, "negative")], abs=0.01
        )
        assert vec.tau_max_positive_nm[i] == pytest.approx(
            stored[(name, "positive")], abs=0.01
        )


def test_force_velocity_parameters_exposed_for_lower_limb() -> None:
    vec = capacity_for(("knee_angle_r",), CapacitySubject(78.0, 1.71))
    fv = vec.force_velocity["knee_angle_r"]
    assert fv["positive"]["omega_75pct_rad_s"] == pytest.approx(1.517)
    assert fv["negative"]["omega_50pct_rad_s"] == pytest.approx(5.233)


def test_sagittal_spine_and_neck_signs_follow_frame_geometry() -> None:
    """Positive rotation about +Y (left-right axis, model faces +X) is forward
    tilt, so positive = flexion: trunk 325.0, neck 30; negative = extension."""
    vec = capacity_for(("SpineInputY", "NeckInputY"), CapacitySubject(78.0, 1.71))
    assert list(vec.tau_max_positive_nm) == pytest.approx([325.0, 30.0])
    assert list(vec.tau_max_negative_nm) == pytest.approx([422.8, 52.0])


# --- NSX-04 slice: lower-limb anatomical slots (#12553) -----------------------
_REGISTRY = ROOT / "src/config/loop_nullspace_registry.json"
_ANATOMICAL = ("hip_adduction", "hip_rotation", "subtalar_angle", "mtp_angle")
# (slot, direction) -> (citation id, cited Nm). Subtalar: da Fonseca 2025 Table 6,
# male mean peak torque (inversion 37.0, eversion 33.4). MTP flexion: Saeki 2021
# Results, first MTP 11.6 + second-fifth MTP 8.0 Nm.
_CITED = {
    **{(f"subtalar_angle_{s}", "positive"): ("CIT-DAFONSECA-2025", 37.0) for s in "lr"},
    **{(f"subtalar_angle_{s}", "negative"): ("CIT-DAFONSECA-2025", 33.4) for s in "lr"},
    **{(f"mtp_angle_{s}", "positive"): ("CIT-SAEKI-2021", 19.6) for s in "lr"},
}


def _torque_entries() -> list[dict]:
    return json.loads(_REGISTRY.read_text("utf-8"))["torque_capacity"]


@pytest.mark.parametrize(("slot", "direction"), sorted(_CITED))
def test_capacity_reproduces_cited_value(slot: str, direction: str) -> None:
    citation, nm = _CITED[(slot, direction)]
    entry = next(
        e
        for e in _torque_entries()
        if e["joint_slot"] == slot and e["direction"] == direction
    )
    assert entry["verification_status"] == "VERIFIED"
    assert entry["assumed"] is False
    assert entry["source"]["citation_id"] == citation
    assert entry["source"]["location"].startswith(("Table", "Results"))
    vec = capacity_for((slot,), CapacitySubject(78.0, 1.71), actuated=(slot,))
    got = (
        vec.tau_max_positive_nm if direction == "positive" else vec.tau_max_negative_nm
    )
    assert got[0] == pytest.approx(nm)


def test_every_anatomical_slot_is_verified_or_states_what_was_not_found() -> None:
    entries = [e for e in _torque_entries() if e["joint_slot"].startswith(_ANATOMICAL)]
    assert len(entries) == 16
    for e in entries:
        label = f"{e['joint_slot']}/{e['direction']}"
        if e["verification_status"] == "VERIFIED":
            assert e["source"]["citation_id"], label
            continue
        note = e["source"]["note"]
        assert note.startswith("Not verified:") and len(note) > 80, label


def test_subtalar_and_mtp_signs_follow_the_model_axes() -> None:
    """Positive subtalar = inversion and positive MTP = flexion, from the model FK
    (ankle positive = dorsiflexion is the control for the method)."""
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    xml, _ = export_full_body_mjcf((SPECS / SPEC_FILES[0]).read_bytes())
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    fwd, up = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)  # model faces +X, +Z up, +Y is left

    def axis(name: str):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return data.xaxis[jid].copy()

    import numpy as np

    for side, medial in (("r", (0.0, 1.0, 0.0)), ("l", (0.0, -1.0, 0.0))):
        ankle, sub, mtp = (
            axis(f"{n}_{side}") for n in ("ankle_angle", "subtalar_angle", "mtp_angle")
        )
        assert np.cross(ankle, fwd)[2] > 0.1  # dorsiflexion raises the toes
        assert np.cross(sub, -np.array(up)) @ medial > 0.1  # sole turns inward
        assert np.cross(mtp, fwd)[2] < -0.1  # flexion lowers the toes


# ---- v2 trunk (lumbar / thoracic) capacity slots --------------------------------
_V2_SPEC = SPECS / "full_body_spec_anthro_driver_v2.json"
_TRUNK_PREFIXES = ("lumbar_", "thoracic_")
#: Pan 2025 male median isometric trunk torque (Nm), by model sign: positive =
#: flexion / right lateral bend / left axial rotation (test_full_body_v2_spec).
_PAN = {
    "flexion": (118.1, 87.3),  # (negative = extension, positive = flexion)
    "lateral_bending": (71.3, 63.2),  # (negative = left, positive = right)
    "axial_rotation": (49.2, 47.7),  # (negative = right, positive = left)
}


def test_v2_spec_has_a_capacity_for_every_actuated_coordinate() -> None:
    """The v2 showpiece spec must build a capacity vector (was: 12 trunk slots missing)."""
    spec = json.loads(_V2_SPEC.read_text("utf-8"))
    vec = capacity_for(tuple(spec["coordinate_order"]), CapacitySubject(78.0, 1.71))
    assert [n for n in vec.actuated_names if n.startswith(_TRUNK_PREFIXES)]


@pytest.mark.parametrize("prefix", ["lumbar", "thoracic"])
@pytest.mark.parametrize("kind", sorted(_PAN))
def test_trunk_capacity_reproduces_pan_2025(prefix: str, kind: str) -> None:
    neg, pos = _PAN[kind]
    slot = f"{prefix}_{kind}"
    vec = capacity_for((slot,), CapacitySubject(78.0, 1.71), actuated=(slot,))
    assert vec.tau_max_negative_nm[0] == pytest.approx(neg)
    assert vec.tau_max_positive_nm[0] == pytest.approx(pos)
    for e in _torque_entries():
        if e["joint_slot"] == slot:
            assert e["source"]["citation_id"] == "CIT-PAN-2025"
            assert e["source"]["location"].startswith("Results")
            assert e["verification_status"] == "VERIFIED" and e["assumed"] is False


def test_physiological_envelope_builds_from_the_v2_spec() -> None:
    """The C3c blocker: from_spec on the v2 showpiece spec must succeed."""
    from src.shared.python.impact_optimization.physiological_envelope import (
        PhysiologicalEnvelope,
        load_limits,
    )

    spec = json.loads(_V2_SPEC.read_text("utf-8"))
    env = PhysiologicalEnvelope.from_spec(
        spec,
        CapacitySubject(78.0, 1.71),
        load_limits(),
        allow_unbounded_rom=frozenset(spec["coordinate_order"]),
    )
    trunk = [n for n in env.names if n.startswith(_TRUNK_PREFIXES)]
    assert len(trunk) == 6
    assert not set(trunk) & set(env.assumed_torque_slots)  # torque is cited
    assert set(trunk) <= set(env.assumed_velocity_slots)  # velocity is a stated prior
