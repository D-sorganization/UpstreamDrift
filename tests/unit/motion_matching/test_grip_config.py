"""Physical grip configuration as an immutable spec revision (HRG-GR2, #13203)."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
    export_full_body_mjcf,
)
from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.model_appearance.grip_pose import GripStyle
from src.shared.python.model_appearance.schema import physics_spec_sha256
from src.shared.python.motion_matching.execution.grip_rotation import (
    GRIP_PROVENANCE_SOURCES,
    apply_grip_config,
    grip_spec_of,
)
from src.shared.python.motion_matching.full_body_spec import canonical_sha256
from src.shared.python.motion_matching.grip_contract import (
    GripSpec,
    HandAngles,
    resolve_visual_pose,
)
from src.shared.python.motion_matching.pipeline.lane import fitted_grip
from src.shared.python.motion_matching.pipeline.receipt import (
    GroundSupportReceiptInputs,
    build_ground_support_receipt,
)
from src.shared.python.motion_matching.pipeline.receipt_schema import Receipt

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "docs/development/full_body_models"
DRIVER = MODELS / "full_body_spec_anthro_driver.json"
IRON = MODELS / "full_body_spec_anthro_iron7.json"
RECEIPT = MODELS / "evidence/ground_support/anthro_driver/receipt.json"


@pytest.fixture(scope="module")
def base() -> dict[str, Any]:
    return json.loads(DRIVER.read_text(encoding="utf-8"))


def _bytes(doc: dict[str, Any]) -> bytes:
    return json.dumps(doc).encode("utf-8")


def _compiled(doc: dict[str, Any]) -> Any:
    import mujoco

    xml, _ = export_full_body_mjcf(_bytes(doc))
    return mujoco.MjModel.from_xml_string(xml)


def _site_rotations(doc: dict[str, Any], *names: str) -> list[np.ndarray]:
    """World rotations of the named frame sites at q = 0."""
    import mujoco

    xml, meta = export_full_body_mjcf(_bytes(doc))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    return [
        data.site_xmat[model.site(meta["frame_sites"][n]).id].reshape(3, 3)
        for n in names
    ]


def _solid(doc: dict[str, Any], name: str) -> dict[str, Any]:
    for body in doc["bodies"]:
        for solid in body["solids"]:
            if solid["name"].endswith(name):
                return solid  # type: ignore[no-any-return]
    raise AssertionError(name)


def _wrist_y(doc: dict[str, Any]) -> float:
    joint = next(j for j in doc["joints"] if j["child"].endswith("Clubface Vector"))
    return float(joint["child_to_follower"][1][3])


def _cfg(**kwargs: Any) -> GripSpec:
    return GripSpec.preset("neutral", **kwargs)


PHYSICAL_CHANGES = [
    (
        "lead.alpha",
        lambda g: replace(g, lead=replace(g.lead, alpha_deg=g.lead.alpha_deg + 1.0)),
    ),
    (
        "lead.psi",
        lambda g: replace(g, lead=replace(g.lead, psi_deg=g.lead.psi_deg + 1.0)),
    ),
    (
        "lead.gamma",
        lambda g: replace(g, lead=replace(g.lead, gamma_deg=g.lead.gamma_deg + 1.0)),
    ),
    (
        "trail.alpha",
        lambda g: replace(g, trail=replace(g.trail, alpha_deg=g.trail.alpha_deg + 1.0)),
    ),
    (
        "trail.psi",
        lambda g: replace(g, trail=replace(g.trail, psi_deg=g.trail.psi_deg + 1.0)),
    ),
    (
        "trail.gamma",
        lambda g: replace(g, trail=replace(g.trail, gamma_deg=g.trail.gamma_deg + 1.0)),
    ),
    ("spacing", lambda g: replace(g, hand_spacing_m=g.hand_spacing_m + 0.001)),
    ("depth", lambda g: replace(g, lead_below_butt_m=g.lead_below_butt_m + 0.001)),
]


@pytest.mark.parametrize(
    ("label", "change"), PHYSICAL_CHANGES, ids=lambda x: x if isinstance(x, str) else ""
)
def test_one_degree_or_one_mm_changes_both_hashes(base, label, change) -> None:
    before = _bytes(base)
    ref = apply_grip_config(base, _cfg())
    out = apply_grip_config(base, change(_cfg()))
    assert out["spec_revision"]["parent_spec_canonical_sha256"] == canonical_sha256(
        base
    )
    assert out["spec_revision"]["version"] == 1
    assert canonical_sha256(out) != canonical_sha256(ref)
    assert _bytes(out) != _bytes(ref)
    assert physics_spec_sha256(out) != physics_spec_sha256(ref)
    assert _bytes(base) == before  # v1 bytes never change
    assert "spec_revision" not in base


def test_committed_spec_and_receipt_bytes_are_untouched(base) -> None:
    spec_bytes, receipt_bytes = DRIVER.read_bytes(), RECEIPT.read_bytes()
    apply_grip_config(json.loads(spec_bytes), GripSpec.preset("strong"))
    assert DRIVER.read_bytes() == spec_bytes
    assert RECEIPT.read_bytes() == receipt_bytes


def test_apply_is_idempotent_and_chains_revisions(base) -> None:
    once = apply_grip_config(base, GripSpec.preset("strong"))
    twice = apply_grip_config(once, GripSpec.preset("strong"))
    assert twice == once
    assert canonical_sha256(twice) == canonical_sha256(once)
    third = apply_grip_config(once, GripSpec.preset("weak"))
    assert third["spec_revision"]["version"] == 2
    assert third["spec_revision"]["parent_spec_canonical_sha256"] == canonical_sha256(
        once
    )


def test_visual_only_parameters_leave_the_compiled_physics_identical(base) -> None:
    a = apply_grip_config(base, _cfg(style=GripStyle.OVERLAP))
    b = apply_grip_config(
        base, _cfg(style=GripStyle.INTERLOCK, lead_thumb_offset_deg=30.0)
    )
    assert _bytes(a) != _bytes(b)  # the visual hints differ ...
    assert physics_spec_sha256(a) == physics_spec_sha256(b)  # ... the physics does not
    ma, mb = _compiled(a), _compiled(b)
    for field in ("body_mass", "body_inertia", "jnt_axis", "eq_data"):
        np.testing.assert_allclose(
            getattr(ma, field), getattr(mb, field), rtol=0, atol=1e-15
        )
    # Re-styling an existing revision mints no new physics revision.
    c = apply_grip_config(
        a, _cfg(style=GripStyle.INTERLOCK, lead_thumb_offset_deg=30.0)
    )
    assert physics_spec_sha256(c) == physics_spec_sha256(a)


@pytest.mark.parametrize("spacing", [0.0762, 0.05, 0.1, 0.0123456789])
def test_grip_interface_spacing_equals_the_configured_value(base, spacing) -> None:
    out = apply_grip_config(base, _cfg(hand_spacing_m=spacing))
    grip = GripInterface.from_spec(out)
    assert grip.right.position_m[1] - grip.left.position_m[1] == pytest.approx(
        spacing, abs=1e-9
    )


def test_depth_moves_the_lead_wrist_and_the_club_descriptor(base) -> None:
    out = apply_grip_config(base, _cfg(lead_below_butt_m=0.05))
    assert _wrist_y(out) == pytest.approx(-(base["club"]["length_m"] - 0.05), abs=1e-12)
    assert out["club"]["wrist_to_head_m"] == pytest.approx(-_wrist_y(out), abs=1e-12)


def test_hand_solids_follow_the_hands_without_double_counting(base) -> None:
    cfg = _cfg(lead_below_butt_m=0.05, hand_spacing_m=0.09)
    out = apply_grip_config(base, cfg)
    lead_y = _wrist_y(out)
    assert _solid(out, "/LHand")["placement"][1][3] == pytest.approx(lead_y, abs=1e-12)
    assert _solid(out, "/RHand")["placement"][1][3] == pytest.approx(
        lead_y + 0.09, abs=1e-12
    )
    assert out["closure"]["placement_b"][1][3] == pytest.approx(
        lead_y + 0.09, abs=1e-12
    )
    for name in ("/LHand", "/RHand", "/LHandStandoff", "/Rigid Shaft", "/Clubhead"):
        for key in ("mass_kg", "inertia_com_kg_m2", "com_m"):
            assert _solid(out, name)[key] == _solid(base, name)[key], (name, key)
    assert sum(m for m in _compiled(out).body_mass) == pytest.approx(
        sum(m for m in _compiled(base).body_mass), abs=1e-12
    )


def test_strong_preset_rotates_the_compiled_lead_frame_by_15_about_the_shaft(
    base,
) -> None:
    neutral = apply_grip_config(base, GripSpec.preset("neutral"))
    strong = apply_grip_config(base, GripSpec.preset("strong"))
    frames = []
    for doc in (neutral, strong):
        club, forearm = _site_rotations(doc, "Clubhead", "LF")
        frames.append(club.T @ forearm)  # the hand (forearm) frame in the club frame
    placement = next(f for f in base["frames"] if f["name"] == "Clubhead")["placement"]
    rel = Rotation.from_matrix(frames[1] @ frames[0].T).as_rotvec()
    # Butt to head is +y of the club body, expressed in the Clubhead frame.
    shaft = np.asarray(placement)[:3, :3].T @ np.array([0.0, 1.0, 0.0])
    assert np.degrees(rel @ shaft) == pytest.approx(15.0, abs=0.01)
    assert np.degrees(np.linalg.norm(rel - (rel @ shaft) * shaft)) < 0.01


def test_left_handed_physical_spec_is_refused_not_silently_right_handed(base) -> None:
    with pytest.raises(ValueError, match="left-handed"):
        apply_grip_config(base, GripSpec.preset("neutral", handedness="left"))
    left = copy.deepcopy(base)
    left["subject"]["handedness"] = "left"
    with pytest.raises(ValueError, match="left-handed"):
        apply_grip_config(left, GripSpec.preset("neutral"))


def test_double_precision_is_path_independent(base) -> None:
    odd = replace(_cfg(), lead=HandAngles(13.123456789012, 41.987654321, 3.3333333333))
    direct = apply_grip_config(base, odd)
    detour = apply_grip_config(apply_grip_config(base, GripSpec.preset("strong")), odd)
    for doc in (direct, detour):
        assert doc["subject"]["grip_config"]["lead"]["alpha_deg"] == 13.123456789012
    a = np.array(
        next(j for j in direct["joints"] if j["child"].endswith("Clubface Vector"))[
            "child_to_follower"
        ]
    )
    b = np.array(
        next(j for j in detour["joints"] if j["child"].endswith("Clubface Vector"))[
            "child_to_follower"
        ]
    )
    assert np.abs(a - b).max() <= 1e-14
    assert grip_spec_of(direct).lead.alpha_deg == 13.123456789012


def test_iron_spec_uses_its_own_club_length() -> None:
    iron = json.loads(IRON.read_text(encoding="utf-8"))
    out = apply_grip_config(iron, _cfg())
    assert _wrist_y(out) == pytest.approx(_wrist_y(iron), abs=1e-9)


def test_provenance_is_recorded_per_parameter_and_validated(base) -> None:
    out = apply_grip_config(base, _cfg())
    prov = out["subject"]["grip_provenance"]
    assert set(prov.values()) == {"user_declared"}
    assert {"lead.alpha_deg", "hand_spacing_m", "lead_below_butt_m"} <= set(prov)
    mixed = apply_grip_config(
        base,
        _cfg(),
        provenance={
            "lead.alpha_deg": "rom_set_identified",
            "hand_spacing_m": "default",
        },
    )
    assert mixed["subject"]["grip_provenance"]["lead.alpha_deg"] == "rom_set_identified"
    assert mixed["subject"]["grip_provenance"]["hand_spacing_m"] == "default"
    assert mixed["subject"]["grip_provenance"]["trail.psi_deg"] == "user_declared"
    assert GRIP_PROVENANCE_SOURCES == ("user_declared", "rom_set_identified", "default")
    with pytest.raises(ValueError, match="provenance"):
        apply_grip_config(base, _cfg(), provenance="guessed")
    with pytest.raises(ValueError, match="provenance"):
        apply_grip_config(base, _cfg(), provenance={"nonsense": "default"})


def test_fitted_grip_uses_provenance_not_nonzero_rotation(base) -> None:
    """Red before HRG-GR2: a declared grip has nonzero rotations and read as fitted."""
    declared = apply_grip_config(base, GripSpec.preset("strong"))
    assert any(
        abs(v) > 0 for a in declared["subject"]["grip_rotation_deg"].values() for v in a
    )
    assert fitted_grip(declared) is False
    fitted = apply_grip_config(
        base, GripSpec.preset("strong"), provenance="rom_set_identified"
    )
    assert fitted_grip(fitted) is True
    assert fitted_grip(base) is True  # legacy: no provenance, nonzero rotation
    zero = copy.deepcopy(base)
    zero["subject"]["grip_rotation_deg"] = {"L": [0, 0, 0], "R": [0, 0, 0]}
    assert fitted_grip(zero) is False


def test_visual_hands_default_from_the_spec_grip_and_block_physical_overrides(
    base,
) -> None:
    grip = GripSpec.preset("strong")
    pose = resolve_visual_pose(grip)
    assert pose == grip.pose()
    restyled = replace(pose, style=GripStyle.TEN_FINGER, lead_thumb_offset_deg=25.0)
    assert resolve_visual_pose(grip, restyled).style is GripStyle.TEN_FINGER
    twisted = replace(pose, lead_v_azimuth_deg=pose.lead_v_azimuth_deg + 5.0)
    with pytest.raises(ValueError, match="physical"):
        resolve_visual_pose(grip, twisted)
    allowed = resolve_visual_pose(grip, twisted, allow_physical_override=True)
    assert allowed.lead_v_azimuth_deg == twisted.lead_v_azimuth_deg
    assert grip_spec_of(apply_grip_config(base, grip)).pose() == pose


def _receipt(doc: dict[str, Any]) -> dict[str, Any]:
    lane = MagicMock()
    lane.bounds = {}
    lane.labels = ("m1",)
    lane.ground_cal.height_m = 0.0
    lane.ground_cal.lowest_marker_height_m = 0.03
    lane.stance = [("heel_r",)] * 4
    with patch(
        "src.shared.python.motion_matching.pipeline.receipt.posture_summary",
        return_value={"spine_bend_deg": {}},
    ):
        return build_ground_support_receipt(
            GroundSupportReceiptInputs(
                base_spec=doc,
                spec_path=DRIVER,
                scaled_path=Path("s.json"),
                hipcal_path=Path("h.json"),
                recalibrate_upper=False,
                anthropometric=None,
                qualification_note="t",
                spec_bytes=b"{}",
                hip_report={},
                candidate_bytes=b"{}",
                c3d_path=Path(__file__),
                capture_name="driver",
                lane=lane,
                address_report={},
                ik_report={},
                dynamics_report={},
                kin=MagicMock(),
                q_ref=np.zeros((300, 4)),
                elapsed_s=1.0,
                validate=False,
            )
        )


def test_receipt_records_grip_provenance_and_revision(base) -> None:
    assert "grip_provenance" in Receipt.model_fields
    assert "spec_revision" in Receipt.model_fields
    legacy = _receipt(base)
    assert legacy["grip_provenance"] is None and legacy["spec_revision"] is None
    out = apply_grip_config(
        base, GripSpec.preset("strong"), provenance="rom_set_identified"
    )
    receipt = _receipt(out)
    assert receipt["grip_provenance"] == out["subject"]["grip_provenance"]
    assert receipt["spec_revision"] == out["spec_revision"]
