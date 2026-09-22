"""XML-level tests for MyoSuite golfer scene composition (MS-51, #10344).

These tests inspect generated MJCF text only — no ``myosuite`` or MuJoCo
import is required. Native load/step coverage lives under ``tests/myosuite/``.
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
    generate_golfer_scene,
    resolve_golfer_scene,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]


def _ensure_scenes(tmp_path: Path | None = None) -> Path:
    """Generate scenes under the engine models tree (or a temp root)."""
    out_root = tmp_path if tmp_path is not None else None
    paths = generate_golfer_scene(repo_root=REPO_ROOT, output_root=out_root)
    assert paths.driver.is_file()
    assert paths.iron.is_file()
    return paths.driver.parent


class TestGolferSceneXml:
    def test_driver_scene_has_club_body_and_dual_welds(self, tmp_path: Path) -> None:
        generate_golfer_scene(repo_root=REPO_ROOT, output_root=tmp_path)
        xml_path = tmp_path / "golfer_myobody_driver.xml"
        root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
        bodies = {el.get("name") for el in root.iter("body")}
        assert "golf_club" in bodies
        welds = list(root.iter("weld"))
        assert len(welds) >= 2
        weld_names = {w.get("name") for w in welds}
        assert "grip_weld_r" in weld_names
        assert "grip_weld_l" in weld_names

    def test_four_foot_contact_geoms_present(self, tmp_path: Path) -> None:
        generate_golfer_scene(repo_root=REPO_ROOT, output_root=tmp_path)
        xml_path = tmp_path / "golfer_myobody_driver.xml"
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

    def test_grip_sites_present_for_dual_weld(self, tmp_path: Path) -> None:
        generate_golfer_scene(repo_root=REPO_ROOT, output_root=tmp_path)
        xml_path = tmp_path / "golfer_myobody_driver.xml"
        root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
        sites = {el.get("name") for el in root.iter("site")}
        for name in ("grip_site_club_r", "grip_site_club_l"):
            assert name in sites
        # Hand grip sites are injected into generated arm-chain includes.
        assets = tmp_path / "assets"
        arm_text = "\n".join(
            p.read_text(encoding="utf-8")
            for p in assets.glob("myoarm_simple*_chain_grip.xml")
        )
        assert "grip_site_hand_r" in arm_text
        assert "grip_site_hand_l" in arm_text
        welds = {el.get("name") for el in root.iter("weld")}
        assert "grip_weld_r" in welds and "grip_weld_l" in welds

    def test_iron_scene_uses_iron7_club_metadata(self, tmp_path: Path) -> None:
        paths = generate_golfer_scene(repo_root=REPO_ROOT, output_root=tmp_path)
        receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
        assert receipt["clubs"]["iron"]["name"] == "iron7"
        assert receipt["clubs"]["driver"]["name"] == "driver"
        assert receipt["myo_sim_pin"] == MYO_SIM_PIN_SHA


class TestCoordinateMap:
    def test_every_mapped_source_resolves_to_myosuite_joint(self) -> None:
        doc = json.loads(COORDINATE_MAP_PATH.read_text(encoding="utf-8"))
        targets = set(doc["target_coordinates"])
        for entry in doc["mappings"]:
            assert entry["target"] in targets
            assert entry["source"] in doc["source_coordinates"]
        # Diagnostic intermediate map: omissions must be explicit.
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
    def test_resolve_prefers_generated_driver_scene(self, tmp_path: Path) -> None:
        generate_golfer_scene(repo_root=REPO_ROOT, output_root=tmp_path)
        scene = resolve_golfer_scene(
            club=ClubKind.DRIVER,
            models_dir=tmp_path,
            repo_root=REPO_ROOT,
        )
        assert scene.xml_path.name == "golfer_myobody_driver.xml"
        assert not scene.is_placeholder
        assert scene.parity_budget_qualified is False
