"""XML-level tests for MyoSuite golfer scene composition (MS-51, #10344).

These tests inspect committed MS-51 MJCF text only — no ``myosuite`` or MuJoCo
import is required, and CI does not need the pinned ``myo_sim`` gitlink.
Native load/step coverage lives under ``tests/myosuite/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import defusedxml.ElementTree as ET  # noqa: S314  # Security: defusedxml prevents XML attacks

import pytest

from src.engines.physics_engines.myosuite.python.golfer_scene import (
    COORDINATE_MAP_PATH,
    MYO_SIM_PIN_SHA,
    ClubKind,
    resolve_golfer_scene,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
BODY_DIR = REPO_ROOT / "shared/models/myosuite/golf/body"
DRIVER_XML = BODY_DIR / "golfer_myobody_driver.xml"
RECEIPT_JSON = BODY_DIR / "golfer_myobody_receipt.json"


@pytest.fixture()
def body_dir() -> Path:
    if not DRIVER_XML.is_file():
        pytest.skip("MS-51 committed golfer scenes missing from checkout")
    return BODY_DIR


class TestGolferSceneXml:
    def test_driver_scene_has_club_body_and_dual_welds(self, body_dir: Path) -> None:
        xml_path = body_dir / "golfer_myobody_driver.xml"
        root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
        bodies = {el.get("name") for el in root.iter("body")}
        assert "golf_club" in bodies
        welds = list(root.iter("weld"))
        assert len(welds) >= 2
        weld_names = {w.get("name") for w in welds}
        assert "grip_weld_r" in weld_names
        assert "grip_weld_l" in weld_names

    def test_four_foot_contact_geoms_present(self, body_dir: Path) -> None:
        xml_path = body_dir / "golfer_myobody_driver.xml"
        root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
        contact_names = {
            el.get("name")
            for el in root.iter("geom")
            if (el.get("name") or "").startswith("contact_")
        }
        expected = {
            "contact_heel_r",
            "contact_forefoot_r",
            "contact_heel_l",
            "contact_forefoot_l",
        }
        assert expected <= contact_names

    def test_grip_sites_present_for_dual_weld(self, body_dir: Path) -> None:
        xml_path = body_dir / "golfer_myobody_driver.xml"
        root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
        sites = {el.get("name") for el in root.iter("site")}
        for name in ("grip_site_club_r", "grip_site_club_l"):
            assert name in sites
        assets = body_dir / "assets"
        arm_text = "\n".join(
            p.read_text(encoding="utf-8")
            for p in assets.glob("myoarm_simple*_chain_grip.xml")
        )
        assert "grip_site_hand_r" in arm_text
        assert "grip_site_hand_l" in arm_text
        welds = {el.get("name") for el in root.iter("weld")}
        assert "grip_weld_r" in welds and "grip_weld_l" in welds

    def test_iron_scene_uses_iron7_club_metadata(self, body_dir: Path) -> None:
        receipt = json.loads(RECEIPT_JSON.read_text(encoding="utf-8"))
        assert receipt["clubs"]["iron"]["name"] == "iron7"
        assert receipt["clubs"]["driver"]["name"] == "driver"
        assert receipt["myo_sim_pin"] == MYO_SIM_PIN_SHA
        assert (body_dir / "golfer_myobody_iron.xml").is_file()


class TestCoordinateMap:
    def test_every_mapped_source_resolves_to_myosuite_joint(self) -> None:
        doc = json.loads(COORDINATE_MAP_PATH.read_text(encoding="utf-8"))
        targets = set(doc["target_coordinates"])
        for entry in doc["mappings"]:
            assert entry["target"] in targets
            assert entry["source"] in doc["source_coordinates"]
        assert "omitted_source" in doc
        assert isinstance(doc["omitted_source"], list)
        mapped = {m["source"] for m in doc["mappings"]}
        for name in doc["source_coordinates"]:
            assert name in mapped or name in doc["omitted_source"]

    def test_map_documents_partial_coverage_honestly(self) -> None:
        doc = json.loads(COORDINATE_MAP_PATH.read_text(encoding="utf-8"))
        n_source = len(doc["source_coordinates"])
        n_mapped = len(doc["mappings"])
        assert n_source == 44
        assert n_mapped <= 44
        assert "diagnostic_only" in doc.get("qualification", {})
        assert doc["qualification"]["diagnostic_only"] is True


class TestResolveGolferScene:
    def test_resolve_prefers_generated_driver_scene(self, body_dir: Path) -> None:
        scene = resolve_golfer_scene(
            club=ClubKind.DRIVER,
            models_dir=body_dir,
            repo_root=REPO_ROOT,
        )
        assert scene.xml_path.name == "golfer_myobody_driver.xml"
        assert not scene.is_placeholder
        assert scene.parity_budget_qualified is False
