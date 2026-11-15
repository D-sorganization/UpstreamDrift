"""HRG-GB2 procedural grip builder (#15124)."""

from __future__ import annotations

import ast
import importlib
import math
import pkgutil
from dataclasses import replace
from pathlib import Path

import pytest

from src.shared.python.core.handedness import DEFAULT_HANDEDNESS
from src.shared.python.grip_build.builder import (
    BONE_TOLERANCE_M,
    FlatHandCurlBuilder,
    build_lead_fingers,
)
from src.shared.python.grip_build.club_sdf import ClubSDF
from src.shared.python.grip_build.hand_chain import (
    HandChain,
    flat_start_club_below_pinky_base,
    flat_start_index_proximal_on_axis,
)
from src.shared.python.grip_build.receipt import BuildStatus
from src.shared.python.grip_contact.grip_geometry import get_grip_preset
from src.shared.python.grip_spec.spec import (
    HandSizeSpec,
    GripSpec,
    hand_length_sweep_sizes,
)


@pytest.mark.unit
def test_flat_start_index_proximal_on_club_axis() -> None:
    spec = GripSpec.neutral()
    geometry = get_grip_preset()
    chain = HandChain(
        hand_size=spec.hand_size,
        grip_radius_m=geometry.radius_at(0.05),
    )
    club = ClubSDF(geometry)
    axis = club.axis_segment()
    assert flat_start_index_proximal_on_axis(chain, axis)


@pytest.mark.unit
def test_flat_start_club_exits_below_pinky_base() -> None:
    spec = GripSpec.neutral()
    geometry = get_grip_preset()
    chain = HandChain(
        hand_size=spec.hand_size,
        grip_radius_m=geometry.radius_at(0.05),
    )
    assert flat_start_club_below_pinky_base(chain, spec.lead.club_cross_angle_deg)


@pytest.mark.unit
def test_each_finger_stops_at_contact_within_segment_tolerance() -> None:
    spec = GripSpec.neutral()
    result = build_lead_fingers(spec)
    for stage in result.receipt.stages:
        if stage.status == BuildStatus.OK:
            assert stage.min_clearance_m >= -stage.target_tolerance_m - 1e-6


@pytest.mark.unit
def test_joint_limits_never_exceeded() -> None:
    spec = GripSpec.neutral()
    result = build_lead_fingers(spec)
    for finger, joints in result.joint_angles_rad.items():
        assert joints["mcp"] <= math.radians(90.0) + 1e-9
        assert joints["pip"] <= math.radians(100.0) + 1e-9
        assert joints["dip"] <= math.radians(80.0) + 1e-9
        _ = finger


@pytest.mark.unit
def test_dip_pip_coupling_default_and_override() -> None:
    spec = GripSpec.neutral()
    chain = HandChain(hand_size=spec.hand_size, dip_over_pip={"index": 0.5})
    chain.joint_angles["index"].pip_rad = math.radians(60.0)
    chain.apply_dip_coupling("index")
    assert chain.joint_angles["index"].dip_rad == pytest.approx(
        chain.joint_angles["index"].pip_rad * 0.5
    )
    default = HandChain(hand_size=spec.hand_size)
    default.joint_angles["index"].pip_rad = math.radians(60.0)
    default.apply_dip_coupling("index")
    assert default.joint_angles["index"].dip_rad == pytest.approx(math.radians(40.0))


@pytest.mark.unit
def test_build_is_deterministic_bitwise_same_inputs() -> None:
    spec = GripSpec.neutral()
    a = build_lead_fingers(spec)
    b = build_lead_fingers(spec)
    assert a.receipt.canonical_json() == b.receipt.canonical_json()


@pytest.mark.unit
def test_curl_order_recorded_and_order_change_changes_receipt() -> None:
    spec = GripSpec.neutral()
    result = build_lead_fingers(spec)
    assert result.receipt.curl_order == spec.curl_order
    reordered = replace(
        GripSpec.neutral(),
        curl_order=("little", "ring", "middle", "index"),
    )
    other = build_lead_fingers(reordered)
    assert other.receipt.canonical_json() != result.receipt.canonical_json()


@pytest.mark.unit
def test_small_hand_reports_status_not_clamp() -> None:
    spec = GripSpec.neutral(hand_length_m=0.14)
    result = build_lead_fingers(spec)
    assert result.status in (
        BuildStatus.JOINT_LIMIT,
        BuildStatus.GAP,
        BuildStatus.INFEASIBLE,
        BuildStatus.OK,
    )
    if result.status != BuildStatus.OK:
        assert any(s.status != BuildStatus.OK for s in result.receipt.stages)


@pytest.mark.unit
def test_cross_angle_and_depth_sweep_monotone_wrap() -> None:
    """More palm depth should not increase total flexion (property smoke test)."""
    from src.shared.python.grip_spec.spec import HandGripSpec

    shallow = GripSpec.neutral(hand_length_m=0.2)
    deep = GripSpec(
        hand_size=HandSizeSpec(length_m=0.2),
        lead=HandGripSpec(
            club_cross_angle_deg=shallow.lead.club_cross_angle_deg,
            palm_finger_depth=0.9,
            thumb_azimuth_deg=shallow.lead.thumb_azimuth_deg,
        ),
        trail=shallow.trail,
    )
    shallow_res = build_lead_fingers(shallow)
    deep_res = build_lead_fingers(deep)

    def total_flex(res):
        return sum(
            joints["mcp"] + joints["pip"] + joints["dip"]
            for joints in res.joint_angles_rad.values()
        )

    if shallow_res.status == BuildStatus.OK and deep_res.status == BuildStatus.OK:
        assert total_flex(deep_res) <= total_flex(shallow_res) + 1e-6


@pytest.mark.unit
@pytest.mark.parametrize("hand_length_m", hand_length_sweep_sizes())
def test_three_hand_sizes_build(hand_length_m: float) -> None:
    spec = GripSpec.neutral(hand_length_m=hand_length_m)
    result = build_lead_fingers(spec)
    assert result.receipt.spec_sha256 == spec.spec_sha256()


@pytest.mark.unit
@pytest.mark.parametrize("handedness", ("right", "left"))
def test_mirrored_left_hand_case(handedness: str) -> None:
    spec = GripSpec.neutral(handedness=handedness)  # type: ignore[arg-type]
    result = build_lead_fingers(spec)
    assert result.receipt.handedness == handedness
    if handedness != DEFAULT_HANDEDNESS:
        rhs = build_lead_fingers(GripSpec.neutral())
        assert result.receipt.canonical_json() != rhs.receipt.canonical_json()


@pytest.mark.unit
def test_grip_build_has_no_engine_imports() -> None:
    root = Path("src/shared/python/grip_build")
    for mod in pkgutil.walk_packages([str(root)]):
        if mod.ispkg:
            continue
        name = f"src.shared.python.grip_build.{mod.name}"
        module = importlib.import_module(name)
        source_path = Path(module.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "engines" not in alias.name
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("src.engines")


@pytest.mark.unit
def test_no_numeric_force_literals_in_grip_build_modules() -> None:
    root = Path("src/shared/python/grip_build")
    forbidden = ("force_n", "mu_crit", "stiffness")
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in text


@pytest.mark.unit
def test_builder_registry_swappable() -> None:
    from src.shared.python.grip_build.registry import get_grip_builder

    builder = get_grip_builder()
    assert isinstance(builder, FlatHandCurlBuilder)


@pytest.mark.unit
def test_receipt_schema_versioned() -> None:
    spec = GripSpec.neutral()
    result = build_lead_fingers(spec)
    data = result.receipt.to_dict()
    assert data["schema_version"] == 1
    assert data["builder_version"].startswith("gb2-")
    assert data["bone_tolerance_m"] == BONE_TOLERANCE_M
