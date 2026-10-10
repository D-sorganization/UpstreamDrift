"""Pure-XML tests for the builder's two-hand club attachment (OSV-7/OSV-9).

The builder attaches the shared club with the ``weld`` grip (lead WeldJoint,
trail WeldConstraint) by default or the ``bushing`` grip (free club, one
BushingForce per hand). Runs on a minimal stand-in base model so no OpenSim or
opensim-models checkout is needed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python import msk_club as mc

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
				<Body name="hand_l" />
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
		<ConstraintSet name="constraintset">
			<objects />
		</ConstraintSet>
		<ForceSet name="forceset">
			<objects />
		</ForceSet>
	</Model>
</OpenSimDocument>
""",
        encoding="utf-8",
    )
    return base_path


def _calibration(tmp_path: Path) -> Path:
    path = tmp_path / "calibration.json"
    mc.store_calibration(
        mc.GripCalibration(
            model="golf_humanoid",
            club="driver",
            hand_frames={"L": np.eye(4), "R": np.eye(4)},
            address_q={"wrist_flex_r": 0.25},
            club_in_ground=np.eye(4),
            report={},
        ),
        path,
    )
    return path


def _build_model(tmp_path: Path, grip_model: str = "weld") -> ET.Element:
    builder = _load_builder()
    out = builder.build(
        output_path=tmp_path / "golf_humanoid.osim",
        grip_model=grip_model,
        base_osim=_write_minimal_base_osim(tmp_path),
        calibration_path=_calibration(tmp_path),
    )
    model = ET.parse(out).getroot().find("Model")
    assert model is not None
    return model


def _tags(model: ET.Element, set_tag: str) -> list[tuple[str, str | None]]:
    objects = model.find(f"{set_tag}/objects")
    assert objects is not None
    return [(e.tag, e.get("name")) for e in objects]


@pytest.mark.unit
def test_default_builder_holds_the_club_with_both_hands(tmp_path: Path) -> None:
    model = _build_model(tmp_path)
    assert ("WeldJoint", "hand_l_to_club") in _tags(model, "JointSet")
    assert ("WeldConstraint", "hand_r_to_club") in _tags(model, "ConstraintSet")
    assert not [t for t in _tags(model, "ForceSet") if t[0] == "BushingForce"]
    default = model.find(".//Coordinate[@name='wrist_flex_r']/default_value")
    assert default is not None and float(default.text) == 0.25  # address pose


@pytest.mark.unit
def test_bushing_grip_frees_the_club_with_one_bushing_per_hand(tmp_path: Path) -> None:
    model = _build_model(tmp_path, "bushing")
    assert ("FreeJoint", "ground_to_club") in _tags(model, "JointSet")
    assert not [t for t in _tags(model, "JointSet") if t[0] == "WeldJoint"]
    forces = _tags(model, "ForceSet")
    assert ("BushingForce", "grip_bushing_left") in forces
    assert ("BushingForce", "grip_bushing_right") in forces
    # the free club coordinates stay passive: actuators only for the skeleton
    actuators = [n for t, n in forces if t == "CoordinateActuator"]
    assert actuators == ["tau_wrist_flex_r"]


@pytest.mark.unit
def test_builder_rejects_unknown_grip_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="grip_model"):
        _build_model(tmp_path, "contact")


@pytest.mark.unit
def test_builder_requires_a_grip_calibration(tmp_path: Path) -> None:
    builder = _load_builder()
    with pytest.raises(KeyError, match="no grip calibration"):
        builder.build(
            output_path=tmp_path / "other_model.osim",
            base_osim=_write_minimal_base_osim(tmp_path),
            calibration_path=_calibration(tmp_path),
        )


@pytest.mark.unit
def test_two_hand_build_is_deterministic(tmp_path: Path) -> None:
    first = (tmp_path / "a").resolve()
    second = (tmp_path / "b").resolve()
    first.mkdir()
    second.mkdir()
    _build_model(first)
    _build_model(second)
    assert (first / "golf_humanoid.osim").read_bytes() == (
        second / "golf_humanoid.osim"
    ).read_bytes()
