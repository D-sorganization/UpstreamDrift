"""Attach, detach, undo and URDF round-trip for the assembly session."""

from __future__ import annotations

import pytest

from src.tools.model_explorer.assembly_session import (
    AssemblyError,
    AssemblySession,
)
from src.tools.model_explorer.part_catalog import PartCatalog, PartSpec

pytestmark = pytest.mark.unit


@pytest.fixture()
def catalog() -> PartCatalog:
    return PartCatalog.bundled()


@pytest.fixture()
def session(catalog: PartCatalog) -> AssemblySession:
    return AssemblySession(catalog, "humanoid_torso")


def test_valid_drop_attaches_part_and_exposes_its_sockets(
    session: AssemblySession,
) -> None:
    decision = session.evaluate_drop("leg_left", "hip_left")
    assert decision.accepted and decision.reason == "compatible"
    placed = session.attach("leg_left", "hip_left")
    assert placed.prefix + "thigh_left" in session.model.links
    assert "hip_left" in session.occupied_ports()
    assert session.model.validate_composition().ok
    ankle = placed.prefix + "ankle"
    assert session.host_port(ankle) is not None
    assert session.evaluate_drop("shoe_left", ankle).accepted


def test_mated_limb_is_attached_at_the_socket_frame(session: AssemblySession) -> None:
    placed = session.attach("leg_left", "hip_left")
    joint = next(
        j
        for n, j in session.model.joints.items()
        if n in placed.joints
        if j.find("parent").get("link") == "pelvis"  # type: ignore[union-attr]
    )
    origin = joint.find("origin")
    assert origin is not None
    assert [float(v) for v in origin.get("xyz", "").split()] == [0.0, 0.09, 0.0]


@pytest.mark.parametrize(
    ("part_id", "port", "fragment"),
    [
        ("leg_right", "hip_left", "side mismatch"),
        ("head", "hip_left", "type mismatch"),
        ("club_iron", "neck", "type mismatch"),
        ("humanoid_torso", "hip_left", "no plug"),
        ("leg_left", "no_such_port", "unknown host port"),
        ("no_such_part", "hip_left", "unknown part"),
    ],
)
def test_invalid_drop_is_rejected_with_a_reason(
    session: AssemblySession, part_id: str, port: str, fragment: str
) -> None:
    before = session.to_urdf()
    decision = session.evaluate_drop(part_id, port)
    assert not decision.accepted and fragment in decision.reason
    with pytest.raises(AssemblyError, match=fragment):
        session.attach(part_id, port)
    assert session.to_urdf() == before
    assert not session.can_undo


def test_occupied_port_is_rejected(session: AssemblySession) -> None:
    session.attach("leg_left", "hip_left")
    decision = session.evaluate_drop("leg_left", "hip_left")
    assert not decision.accepted and "occupied" in decision.reason


def test_validation_errors_block_a_drop(catalog: PartCatalog) -> None:
    arm = catalog.get("arm_left")
    start = arm.urdf_xml.index('<link name="forearm_left">')
    end = arm.urdf_xml.index("</inertial>", start) + len("</inertial>")
    start_inertial = arm.urdf_xml.index("<inertial>", start)
    broken_xml = arm.urdf_xml[:start_inertial] + arm.urdf_xml[end:]
    catalog.register(PartSpec("arm_bad", "Bad Arm", "limb", "", broken_xml, arm.ports))
    session = AssemblySession(catalog, "humanoid_torso")
    decision = session.evaluate_drop("arm_bad", "shoulder_left")
    assert not decision.accepted
    assert "inertial" in decision.reason
    assert any(f.code == "missing_inertial" for f in decision.findings)
    with pytest.raises(AssemblyError):
        session.attach("arm_bad", "shoulder_left")


def test_undo_and_redo_restore_exact_state(session: AssemblySession) -> None:
    initial = session.to_urdf()
    session.attach("leg_left", "hip_left")
    attached = session.to_urdf()
    assert session.can_undo and not session.can_redo
    assert session.undo()
    assert session.to_urdf() == initial
    assert "hip_left" not in session.occupied_ports()
    assert session.redo()
    assert session.to_urdf() == attached
    assert not session.redo()


def test_new_change_clears_redo_and_undo_stops_at_start(
    session: AssemblySession,
) -> None:
    assert not session.undo()
    session.attach("leg_left", "hip_left")
    session.undo()
    session.attach("head", "neck")
    assert not session.can_redo


def test_detach_removes_part_and_dependents(session: AssemblySession) -> None:
    leg = session.attach("leg_left", "hip_left")
    shoe = session.attach("shoe_left", leg.prefix + "ankle")
    removed = session.detach(leg.instance_id)
    assert set(removed) == {leg.instance_id, shoe.instance_id}
    assert set(session.model.links) == {"pelvis", "chest"}
    assert set(session.model.joints) == {"pelvis_to_chest"}
    assert session.free_sockets()
    assert session.undo()
    assert shoe.prefix + "shoe_left" in session.model.links


def test_base_cannot_be_detached(session: AssemblySession) -> None:
    with pytest.raises(AssemblyError, match="base"):
        session.detach(session.placed_parts[0].instance_id)
    with pytest.raises(KeyError):
        session.detach("nope")


def test_urdf_round_trip_preserves_assembly(
    session: AssemblySession, catalog: PartCatalog
) -> None:
    leg = session.attach("leg_left", "hip_left")
    session.attach("shoe_left", leg.prefix + "ankle")
    xml = session.to_urdf()
    restored = AssemblySession.from_urdf(xml, catalog)
    assert restored.to_urdf() == xml
    assert restored.occupied_ports() == session.occupied_ports()
    assert [p.instance_id for p in restored.placed_parts] == [
        p.instance_id for p in session.placed_parts
    ]
    # the restored session keeps working
    restored.attach("head", "neck")
    assert restored.model.validate_composition().ok


def test_round_trip_rejects_urdf_without_a_record(catalog: PartCatalog) -> None:
    with pytest.raises(AssemblyError, match="no embedded"):
        AssemblySession.from_urdf('<robot name="r"><link name="a"/></robot>', catalog)
    with pytest.raises(ValueError):
        AssemblySession.from_urdf("", catalog)


def test_listeners_fire_on_change_undo_and_redo(session: AssemblySession) -> None:
    calls: list[int] = []
    session.subscribe(lambda: calls.append(1))
    session.attach("head", "neck")
    session.undo()
    session.redo()
    assert len(calls) == 3


def test_robot_arm_chain_and_club_on_hand(catalog: PartCatalog) -> None:
    session = AssemblySession(catalog, "pedestal")
    arm = session.attach("robot_arm", "top_mount")
    club = session.attach("club_driver", arm.prefix + "tool_grip")
    assert club.prefix + "grip" in session.model.links
    assert session.model.validate_composition().ok
