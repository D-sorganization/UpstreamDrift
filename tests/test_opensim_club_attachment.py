"""Pure-XML tests for optional OpenSim compliant club attachment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILDER_PATH = REPO_ROOT / "scripts" / "build_humanoid_osim.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_humanoid_osim", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_minimal_base_osim(tmp_path: Path) -> Path:
    base_path = tmp_path / "base.osim"
    base_path.write_text(
        """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
	<Model name="OpenSense_Subject">
		<BodySet name="bodyset">
			<objects>
				<Body name="ground_body" />
				<Body name="hand_r" />
			</objects>
		</BodySet>
		<JointSet name="jointset">
			<objects>
				<PinJoint name="hand_pin">
					<coordinates>
						<Coordinate name="wrist_flex_r" />
					</coordinates>
				</PinJoint>
			</objects>
		</JointSet>
		<ForceSet name="forceset">
			<objects />
		</ForceSet>
	</Model>
</OpenSimDocument>
""",
        encoding="utf-8",
    )
    return base_path


def _build_model(tmp_path: Path, *, club_attachment=None) -> ET.Element:
    builder = _load_builder()
    builder.BASE_OSIM = _write_minimal_base_osim(tmp_path)
    output_path = tmp_path / "golf_humanoid.osim"
    if club_attachment is None:
        builder.build(output_path=output_path)
    else:
        builder.build(output_path=output_path, club_attachment=club_attachment)
    root = ET.parse(output_path).getroot()
    model = root.find("Model")
    assert model is not None
    return model


def _joint_objects(model: ET.Element) -> ET.Element:
    jointset = model.find("JointSet")
    assert jointset is not None
    objects = jointset.find("objects")
    assert objects is not None
    return objects


def _force_objects(model: ET.Element) -> ET.Element:
    forceset = model.find("ForceSet")
    assert forceset is not None
    objects = forceset.find("objects")
    assert objects is not None
    return objects


@pytest.mark.unit
def test_default_builder_keeps_rigid_club_weld_and_no_bushing(tmp_path: Path) -> None:
    model = _build_model(tmp_path)

    welds = [
        joint
        for joint in _joint_objects(model)
        if joint.tag == "WeldJoint" and joint.get("name") == "hand_r_to_club"
    ]
    bushings = [
        force
        for force in _force_objects(model)
        if force.tag == "BushingForce" and force.get("name") == "hand_r_to_club_bushing"
    ]

    assert len(welds) == 1
    assert bushings == []


@pytest.mark.unit
def test_compliant_club_attachment_emits_bushing_without_rigid_weld(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    config = builder.CompliantClubAttachmentConfig(
        translational_stiffness=(1000.0, 1100.0, 1200.0),
        rotational_stiffness=(10.0, 11.0, 12.0),
        translational_damping=(50.0, 51.0, 52.0),
        rotational_damping=(1.0, 1.1, 1.2),
    )

    model = _build_model(tmp_path, club_attachment=config)

    welds = [
        joint
        for joint in _joint_objects(model)
        if joint.tag == "WeldJoint" and joint.get("name") == "hand_r_to_club"
    ]
    bushings = [
        force
        for force in _force_objects(model)
        if force.tag == "BushingForce" and force.get("name") == "hand_r_to_club_bushing"
    ]

    assert welds == []
    assert len(bushings) == 1
    bushing = bushings[0]
    assert bushing.findtext("socket_frame1") == "hand_r_grip_offset"
    assert bushing.findtext("socket_frame2") == "club_grip_offset"
    assert bushing.findtext("translational_stiffness") == "1000.0 1100.0 1200.0"
    assert bushing.findtext("rotational_stiffness") == "10.0 11.0 12.0"
    assert bushing.findtext("translational_damping") == "50.0 51.0 52.0"
    assert bushing.findtext("rotational_damping") == "1.0 1.1 1.2"
    assert bushing.findtext("translational_units") == "N_per_m"
    assert bushing.findtext("rotational_units") == "N_m_per_rad"


@pytest.mark.unit
def test_compliant_club_attachment_rejects_invalid_parameters() -> None:
    builder = _load_builder()

    invalid_cases = [
        (
            {"translational_stiffness": (-1.0, 0.0, 0.0)},
            "translational_stiffness values must be non-negative",
        ),
        (
            {"rotational_damping": (0.0, -0.1, 0.0)},
            "rotational_damping values must be non-negative",
        ),
        ({"parent_body": ""}, "parent_body must be a non-empty body name"),
        (
            {"child_body": "   "},
            "child_body must be a non-empty body name",
        ),
        (
            {"translational_units": "N/mm"},
            "Unsupported translational_units",
        ),
        (
            {"rotational_units": "deg"},
            "Unsupported rotational_units",
        ),
    ]

    for kwargs, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            builder.CompliantClubAttachmentConfig(**kwargs)


@pytest.mark.unit
def test_compliant_club_attachment_rejects_missing_model_body(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    builder.BASE_OSIM = _write_minimal_base_osim(tmp_path)
    config = builder.CompliantClubAttachmentConfig(parent_body="missing_hand")

    with pytest.raises(
        ValueError,
        match="Compliant club attachment references missing body names: missing_hand",
    ):
        builder.build(output_path=tmp_path / "invalid.osim", club_attachment=config)


@pytest.mark.unit
def test_compliant_club_attachment_serialization_is_deterministic(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    builder.BASE_OSIM = _write_minimal_base_osim(tmp_path)
    config = builder.CompliantClubAttachmentConfig()
    first = tmp_path / "first.osim"
    second = tmp_path / "second.osim"

    builder.build(output_path=first, club_attachment=config)
    builder.build(output_path=second, club_attachment=config)

    assert first.read_bytes() == second.read_bytes()
