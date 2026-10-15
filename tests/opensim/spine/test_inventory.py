"""Spine inventory (SPN-8a, #12753): level map, neutral offsets, ISB axis check."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET  # noqa: S405  # nosemgrep  # test-only parse of generated XML
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.spine.fixture_model import (
    FixtureSpec,
    fixture_level_names,
    fixture_neutral_transforms,
    write_fixture_osim,
)
from src.engines.physics_engines.opensim.python.spine.inventory import (
    DEFAULT_LEVEL_MAP_PATH,
    LevelMapEntry,
    SpineInventory,
    inventory_spine,
    load_level_maps,
    neutral_offset_errors,
)
from src.shared.python.spine.levels import SpinalLevel
from tests.helpers.opensim_support import require_opensim

CONFIG = Path(__file__).resolve().parents[3] / "src/config/spine_model_level_maps.json"


# --------------------------------------------------------------------- pure --


@pytest.mark.unit
class TestConfigSchema:
    def test_default_path_is_the_config(self) -> None:
        assert DEFAULT_LEVEL_MAP_PATH == CONFIG

    def test_expected_model_ids(self) -> None:
        maps = load_level_maps()
        assert set(maps) == {
            "fixture",
            "bruno2015_fullbody_os4_v2",
            "senteler2016_lumbspine_3dof",
            "senteler2016_lumbspine_30dof",
        }

    def test_fixture_entry_matches_fixture_names(self) -> None:
        entry = load_level_maps()["fixture"]
        assert entry.verified
        assert (entry.pelvis_body, entry.terminal_body) == ("pelvis", "thorax_proxy")
        for names in fixture_level_names(FixtureSpec()):
            lm = entry.levels[names.level]
            assert lm.joint == names.joint
            assert lm.coordinates == names.coordinates
            assert lm.axis_sign == (1, 1, 1)

    def test_licensed_entries_unverified_without_invented_names(self) -> None:
        for model_id, entry in load_level_maps().items():
            if model_id == "fixture":
                continue
            assert entry.verified is False
            assert entry.levels == {}

    def test_raw_json_flags(self) -> None:
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        for body in raw["models"].values():
            assert isinstance(body["verified"], bool)
            if not body["verified"]:
                assert "UNVERIFIED" in body["names_status"]

    def test_bad_level_name_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "m.json"
        p.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "models": {
                        "x": {
                            "verified": True,
                            "pelvis_body": "a",
                            "terminal_body": "b",
                            "levels": {
                                "Q1_Q2": {
                                    "joint": "j",
                                    "coordinates": ["a", "b", "c"],
                                    "axis_sign": [1, 1, 1],
                                }
                            },
                        }
                    },
                }
            )
        )
        with pytest.raises(ValueError, match="Q1_Q2"):
            load_level_maps(p)

    def test_bad_axis_sign_rejected(self) -> None:
        with pytest.raises(ValueError, match="axis_sign"):
            LevelMapEntry(joint="j", coordinates=("a", "b", "c"), axis_sign=(1, 2, 1))

    def test_duplicate_coordinates_rejected(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            LevelMapEntry(joint="j", coordinates=("a", "a", "c"), axis_sign=(1, 1, 1))


@pytest.mark.unit
def test_neutral_offset_errors_pure() -> None:
    a = {SpinalLevel.L5_S1: np.eye(4)}
    b = {SpinalLevel.L5_S1: np.eye(4)}
    b[SpinalLevel.L5_S1][0, 3] = 1e-3
    assert neutral_offset_errors(a, a)[SpinalLevel.L5_S1] == 0.0
    assert neutral_offset_errors(a, b)[SpinalLevel.L5_S1] == pytest.approx(1e-3)
    with pytest.raises(ValueError, match="levels"):
        neutral_offset_errors(a, {})


# ------------------------------------------------------------------- native --


@pytest.fixture(scope="module")
def fixture_path(tmp_path_factory) -> Path:
    return write_fixture_osim(
        FixtureSpec(), tmp_path_factory.mktemp("inv") / "fixture.osim"
    )


def _swapped_axis_path(src: Path, dst: Path) -> Path:
    tree = ET.parse(src)  # noqa: S314  # nosemgrep  # generated, trusted
    for joint in tree.getroot().iter("CustomJoint"):
        if joint.get("name") != "L4_L5_jnt":
            continue
        for axis in joint.iter("TransformAxis"):
            if axis.get("name") == "rotation1":
                axis.find(
                    "axis"
                ).text = "0 1 1"  # 45 deg from ISB Z; OpenSim rejects collinear axes
    tree.write(dst)
    return dst


@pytest.mark.requires_opensim
class TestNativeInventory:
    def test_fixture_maps_all_levels(self, fixture_path: Path) -> None:
        require_opensim()
        inv = inventory_spine(fixture_path, "fixture")
        assert isinstance(inv, SpineInventory)
        assert inv.levels == FixtureSpec().levels
        assert len(inv.levels) == 5
        for names in fixture_level_names(FixtureSpec()):
            assert inv.joint_names[names.level] == names.joint
            assert inv.coordinate_names[names.level] == names.coordinates

    def test_neutral_matches_fixture(self, fixture_path: Path) -> None:
        require_opensim()
        inv = inventory_spine(fixture_path, "fixture")
        ref = fixture_neutral_transforms(FixtureSpec())
        errs = neutral_offset_errors(inv.neutral, ref)
        assert max(errs.values()) < 1e-12
        for mat in inv.neutral.values():
            assert mat.shape == (4, 4)
            assert not mat.flags.writeable

    def test_axis_check_passes(self, fixture_path: Path) -> None:
        require_opensim()
        inv = inventory_spine(fixture_path, "fixture", strict=True)
        assert inv.axis_failures == ()
        assert inv.axis_ok
        for deg in inv.axis_alignment_deg.values():
            assert deg.shape == (3,)
            assert np.all(deg <= 1.0)

    def test_swapped_axis_listed(self, fixture_path: Path, tmp_path: Path) -> None:
        require_opensim()
        bad = _swapped_axis_path(fixture_path, tmp_path / "bad.osim")
        inv = inventory_spine(bad, "fixture")
        assert not inv.axis_ok
        assert len(inv.axis_failures) == 1
        assert "L4_L5" in inv.axis_failures[0]
        assert inv.axis_alignment_deg[SpinalLevel.L4_L5][0] == pytest.approx(45.0)

    def test_swapped_axis_strict_raises(
        self, fixture_path: Path, tmp_path: Path
    ) -> None:
        require_opensim()
        bad = _swapped_axis_path(fixture_path, tmp_path / "bad.osim")
        with pytest.raises(ValueError, match="L4_L5"):
            inventory_spine(bad, "fixture", strict=True)

    def test_flipped_sign_in_map_is_an_explicit_transform(
        self, fixture_path: Path
    ) -> None:
        require_opensim()
        base = load_level_maps()["fixture"]
        lm = base.levels[SpinalLevel.L4_L5]
        flipped = LevelMapEntry(lm.joint, lm.coordinates, (-1, 1, 1))
        levels = {**base.levels, SpinalLevel.L4_L5: flipped}
        entry = type(base)(
            verified=True,
            pelvis_body=base.pelvis_body,
            terminal_body=base.terminal_body,
            levels=levels,
        )
        inv = inventory_spine(fixture_path, "fixture", level_map=entry)
        assert not inv.axis_ok  # model axis +Z, map says it is -ISB Z

    def test_unmapped_joint_raises_naming_it(self, fixture_path: Path) -> None:
        require_opensim()
        base = load_level_maps()["fixture"]
        lm = base.levels[SpinalLevel.L4_L5]
        levels = {
            **base.levels,
            SpinalLevel.L4_L5: LevelMapEntry("nope_jnt", lm.coordinates, lm.axis_sign),
        }
        entry = type(base)(True, base.pelvis_body, base.terminal_body, levels)
        with pytest.raises(ValueError, match="nope_jnt"):
            inventory_spine(fixture_path, "fixture", level_map=entry)

    def test_unmapped_coordinate_raises_naming_it(self, fixture_path: Path) -> None:
        require_opensim()
        base = load_level_maps()["fixture"]
        lm = base.levels[SpinalLevel.L4_L5]
        levels = {
            **base.levels,
            SpinalLevel.L4_L5: LevelMapEntry(
                lm.joint,
                (lm.coordinates[0], "ghost_coord", lm.coordinates[2]),
                lm.axis_sign,
            ),
        }
        entry = type(base)(True, base.pelvis_body, base.terminal_body, levels)
        with pytest.raises(ValueError, match="ghost_coord"):
            inventory_spine(fixture_path, "fixture", level_map=entry)

    def test_unverified_model_refused(self, fixture_path: Path) -> None:
        require_opensim()
        with pytest.raises(ValueError, match="UNVERIFIED|unverified"):
            inventory_spine(fixture_path, "bruno2015_fullbody_os4_v2")

    def test_unknown_model_id(self, fixture_path: Path) -> None:
        require_opensim()
        with pytest.raises(ValueError, match="model_id"):
            inventory_spine(fixture_path, "nope")

    def test_missing_file(self, tmp_path: Path) -> None:
        require_opensim()
        with pytest.raises(ValueError, match="not found"):
            inventory_spine(tmp_path / "none.osim", "fixture")

    def test_non_contiguous_levels_raise(self, fixture_path: Path) -> None:
        require_opensim()
        base = load_level_maps()["fixture"]
        levels = {k: v for k, v in base.levels.items() if k is not SpinalLevel.L3_L4}
        entry = type(base)(True, base.pelvis_body, base.terminal_body, levels)
        with pytest.raises(ValueError, match="contiguous"):
            inventory_spine(fixture_path, "fixture", level_map=entry)
