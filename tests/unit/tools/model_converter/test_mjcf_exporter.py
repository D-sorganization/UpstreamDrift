"""Unit tests for MuJoCo MJCF model exporter (Issue #9965)."""

from __future__ import annotations

import sys
import defusedxml.ElementTree as ET
from pathlib import Path
import pytest

pytestmark = pytest.mark.unit


REPO_ROOT = Path(__file__).parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.model_converter.mjcf_exporter import export_mjcf
from tools.model_converter.schema_validator import validate_canonical_model

CANONICAL_YAML = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "pinocchio"
    / "models"
    / "spec"
    / "golfer_canonical.yaml"
)


@pytest.fixture(scope="module")
def canonical_model():
    return validate_canonical_model(CANONICAL_YAML)


@pytest.fixture(scope="module")
def mjcf_xml(canonical_model) -> str:
    return export_mjcf(canonical_model)


@pytest.fixture(scope="module")
def mjcf_root(mjcf_xml: str) -> ET.Element:
    return ET.fromstring(mjcf_xml)


class TestMjcfExporter:
    """Test exported MJCF XML structure and MuJoCo compilation."""

    def test_xml_is_well_formed(self, mjcf_root: ET.Element) -> None:
        assert mjcf_root.tag == "mujoco"
        assert mjcf_root.attrib.get("model") == "golfer"

    def test_compiles_in_mujoco(self, mjcf_xml: str) -> None:
        try:
            import mujoco
        except ImportError:
            pytest.skip("mujoco not installed")

        model = mujoco.MjModel.from_xml_string(mjcf_xml)
        assert model.nq > 0
        assert model.nv > 0
        assert model.nbody > 0
        # Check floating base (6 nv) + internal joints
        assert model.nv == 49
        assert model.nq == 50

    def test_pelvis_freejoint(self, mjcf_root: ET.Element) -> None:
        pelvis = mjcf_root.find(".//body[@name='pelvis']")
        assert pelvis is not None
        freejoint = pelvis.find("freejoint")
        assert freejoint is not None
        assert freejoint.attrib.get("name") == "pelvis_free"

    def test_dual_grip_equality_weld(self, mjcf_root: ET.Element) -> None:
        equality = mjcf_root.find("equality")
        assert equality is not None
        weld = equality.find("weld")
        assert weld is not None
        assert weld.attrib.get("body1") == "hand_right"
        assert weld.attrib.get("body2") == "club_shaft"

    def test_actuators_present(self, mjcf_root: ET.Element) -> None:
        actuator = mjcf_root.find("actuator")
        assert actuator is not None
        motors = actuator.findall("motor")
        assert len(motors) >= 20

    def test_export_to_file(self, canonical_model, tmp_path: Path) -> None:
        out_file = tmp_path / "test_golfer.xml"
        xml_str = export_mjcf(canonical_model, out_path=out_file)
        assert out_file.exists()
        assert out_file.read_text(encoding="utf-8") == xml_str
