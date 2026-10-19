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

    @pytest.mark.parametrize(
        ("kind", "xml_name", "spec_name"),
        [
            (ClubKind.DRIVER, "golfer_myobody_driver.xml", "driver"),
            (ClubKind.IRON, "golfer_myobody_iron.xml", "iron7"),
        ],
    )
    def test_club_shaft_spans_full_length_and_grip_sites_lie_within_grip_length(
        self,
        body_dir: Path,
        kind: ClubKind,
        xml_name: str,
        spec_name: str,
    ) -> None:
        from src.engines.physics_engines.myosuite.python.golfer_scene import (
            _DEFAULT_CONTACT,
            _worldbody_addons,
        )
        from src.shared.python.model_appearance.grip_pose import DEFAULT_GRIP_POSE, Hand
        from src.shared.python.motion_matching.club_models import CLUBS

        club = CLUBS[spec_name]
        xml_path = body_dir / xml_name
        committed_root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
        generated_root = ET.fromstring(
            f"<worldbody>{_worldbody_addons(club, _DEFAULT_CONTACT)}</worldbody>"
        )

        for label, root in (
            ("generated", generated_root),
            ("committed", committed_root),
        ):
            shaft = next(
                el for el in root.iter("geom") if el.get("name") == "club_shaft"
            )
            fromto = [float(v) for v in (shaft.get("fromto") or "").split()]
            assert len(fromto) == 6, f"{label} {xml_name}: invalid fromto"
            butt_y = min(fromto[1], fromto[4])
            head_y = max(fromto[1], fromto[4])
            assert butt_y == pytest.approx(-club.length_m), (
                f"{label} {xml_name}: shaft butt_y={butt_y} != {-club.length_m}"
            )
            assert head_y == pytest.approx(0.0), (
                f"{label} {xml_name}: shaft head_y={head_y} != 0.0"
            )

            grip_y_min = -club.length_m
            grip_y_max = -(club.length_m - club.grip_length_m)
            sites = {
                el.get("name"): [float(v) for v in (el.get("pos") or "").split()]
                for el in root.iter("site")
                if (el.get("name") or "").startswith("grip_site_club_")
            }
            assert set(sites) == {"grip_site_club_l", "grip_site_club_r"}
            for site_name, pos in sites.items():
                assert pos[0] == pytest.approx(0.0)
                assert pos[2] == pytest.approx(0.0)
                assert butt_y <= pos[1] <= head_y, (
                    f"{label} {xml_name}: {site_name} y={pos[1]} outside shaft [{butt_y}, {head_y}]"
                )
                assert grip_y_min <= pos[1] <= grip_y_max, (
                    f"{label} {xml_name}: {site_name} y={pos[1]} outside grip [{grip_y_min}, {grip_y_max}]"
                )

            expected_l = -(
                club.length_m - DEFAULT_GRIP_POSE.distance_below_butt_m(Hand.LEAD)
            )
            expected_r = -(
                club.length_m - DEFAULT_GRIP_POSE.distance_below_butt_m(Hand.TRAIL)
            )
            assert sites["grip_site_club_l"][1] == pytest.approx(expected_l)
            assert sites["grip_site_club_r"][1] == pytest.approx(expected_r)

    @pytest.mark.parametrize(
        "xml_name",
        ["golfer_myobody_driver.xml", "golfer_myobody_iron.xml"],
    )
    def test_club_weld_residuals_at_qpos0_are_zero(
        self,
        body_dir: Path,
        xml_name: str,
    ) -> None:
        mujoco = pytest.importorskip("mujoco")
        import numpy as np

        myo_body = (
            REPO_ROOT / "shared/models/myosuite/myo_sim/body/myobody_simpleupper.xml"
        )
        if not myo_body.is_file():
            pytest.skip("myo_sim not initialized; run scripts/setup_myosuite_models.sh")

        xml_path = body_dir / xml_name
        model = mujoco.MjModel.from_xml_path(str(xml_path.resolve()))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        for weld_name, hand_site, club_site in (
            ("grip_weld_l", "grip_site_hand_l", "grip_site_club_l"),
            ("grip_weld_r", "grip_site_hand_r", "grip_site_club_r"),
        ):
            sid_h = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, hand_site))
            sid_c = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, club_site))
            assert sid_h >= 0 and sid_c >= 0
            pos_h = np.asarray(data.site_xpos[sid_h], dtype=float)
            pos_c = np.asarray(data.site_xpos[sid_c], dtype=float)
            dist_m = float(np.linalg.norm(pos_h - pos_c))

            rot_h = np.asarray(data.site_xmat[sid_h], dtype=float).reshape(3, 3)
            rot_c = np.asarray(data.site_xmat[sid_c], dtype=float).reshape(3, 3)
            cos_theta = float(
                np.clip((np.trace(rot_h.T @ rot_c) - 1.0) / 2.0, -1.0, 1.0)
            )
            angle_deg = float(np.degrees(np.arccos(cos_theta)))

            assert dist_m <= 1e-6, (
                f"{xml_name} {weld_name}: qpos0 site distance residual "
                f"{dist_m:.6f} m > 1e-6 m"
            )
            assert angle_deg <= 1e-3, (
                f"{xml_name} {weld_name}: qpos0 rotation angle residual "
                f"{angle_deg:.4f} deg > 1e-3 deg"
            )


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
