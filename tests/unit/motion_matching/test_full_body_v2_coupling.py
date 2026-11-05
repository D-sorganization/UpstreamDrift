"""SHOW-A3 (#14552): v2 thoracic segment contracts and the OFF-by-default coupling."""

from __future__ import annotations

import json
from pathlib import Path

import defusedxml.ElementTree as DET
import pytest

from src.shared.python.motion_matching import full_body_v2 as v2
from src.shared.python.motion_matching.full_body_migration import RegionalPrior
from src.shared.python.motion_matching.full_body_v2_coupling import (
    trunk_coupling_ratios,
    with_trunk_coupling,
)
from src.shared.python.motion_matching.full_body_v2_geometry import (
    derive_landmark_v2_spec,
    spec_mass_properties,
)
from src.shared.python.spine.levels import SpinalLevel, SpinalRegion, parse_level

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
DRIVER = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
AXES = ("flexion", "lateral_bending", "axial_rotation")


@pytest.fixture(scope="module")
def v1() -> dict:
    return json.loads(DRIVER.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def v2_doc(v1) -> dict:
    return derive_landmark_v2_spec(v1, provenance="SHOW-A3 contract fixture")


def _short(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def test_joint_tree_is_pelvis_lumbar_thorax_with_upper_body_on_thorax(v2_doc) -> None:
    parent = {_short(j["child"]): _short(j["parent"]) for j in v2_doc["joints"]}
    assert parent["trunk_lumbar"] == "LowerTorso"
    assert parent["COMRod"] == "trunk_lumbar"
    assert "UpperTorsoBase" not in parent
    assert parent["Head"] == "COMRod"
    assert parent["HubtoLS"] == "COMRod" and parent["HubtoRS"] == "COMRod"


def test_coordinate_order_replaces_trunk_in_place_and_stays_contiguous(
    v1, v2_doc
) -> None:
    order = v2_doc["coordinate_order"]
    assert len(order) == len(v1["coordinate_order"]) + 3
    start = order.index("lumbar_flexion")
    assert tuple(order[start : start + 6]) == v2.V2_TRUNK.coordinates
    assert not {"TorsoInput", "SpineInputX", "SpineInputY"} & set(order)
    rest_v1 = [c for c in v1["coordinate_order"] if c not in v2.V1_TRUNK.coordinates]
    assert [c for c in order if c not in v2.V2_TRUNK.coordinates] == rest_v1


def test_trunk_mass_and_com_unchanged_within_one_percent(v1, v2_doc) -> None:
    before, after = spec_mass_properties(v1), spec_mass_properties(v2_doc)
    assert after.mass_kg == pytest.approx(before.mass_kg, rel=0.01)
    assert after.com_m == pytest.approx(before.com_m, abs=0.01 * 0.9)


def test_thoracic_rom_is_cited_in_spine_level_vocabulary(v2_doc) -> None:
    assert parse_level("T12_L1") is SpinalLevel.T12_L1
    assert SpinalLevel.T12_L1.region is SpinalRegion.THORACIC
    assert SpinalLevel.L5_S1.region is SpinalRegion.LUMBAR
    levels = [j.level for j in v2.V2_TRUNK.joints]
    assert [parse_level(x) for x in levels] == [SpinalLevel.L5_S1, SpinalLevel.T12_L1]
    soft = v2_doc["soft_limits_deg"]
    assert soft["thoracic_axial_rotation"] == [-50.0, 50.0]
    assert "thoracic_axial_rotation" not in v2_doc["coordinate_ranges_deg"]


def test_ratios_are_the_bruno_regional_split_thoracic_over_lumbar() -> None:
    ratios = trunk_coupling_ratios(RegionalPrior.from_coupling_law())
    assert ratios["flexion"] == pytest.approx(49 / 51)
    assert ratios["lateral_bending"] == pytest.approx(73 / 27)
    assert ratios["axial_rotation"] == pytest.approx(83 / 17)


def _compile(doc: dict):
    """(mujoco, xml, compiled model) of a spec through the MuJoCo exporter."""
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    xml, _ = export_full_body_mjcf(json.dumps(doc).encode())
    return mujoco, xml, mujoco.MjModel.from_xml_string(xml)


def test_coupling_is_off_by_default_and_changes_no_constraint_count(v2_doc) -> None:
    doc = with_trunk_coupling(v2_doc)  # enabled defaults to False
    assert doc["trunk_coupling"]["enabled"] is False
    _, xml, model = _compile(doc)
    assert xml == _compile(v2_doc)[1]
    assert model.neq == 1  # the grip weld only


def test_enabled_coupling_emits_one_joint_equality_per_axis(v2_doc) -> None:
    mujoco, _, model = _compile(with_trunk_coupling(v2_doc, enabled=True))
    assert model.neq == 1 + 3
    ratios = trunk_coupling_ratios(RegionalPrior.from_coupling_law())
    joint_eq = [
        i for i in range(model.neq) if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT
    ]
    assert len(joint_eq) == 3
    for i, axis in zip(joint_eq, AXES, strict=True):
        assert model.joint(model.eq_obj1id[i]).name == f"thoracic_{axis}"
        assert model.joint(model.eq_obj2id[i]).name == f"lumbar_{axis}"
        assert model.eq_data[i][:2] == pytest.approx([0.0, ratios[axis]])


def test_enabled_coupling_drives_thoracic_to_ratio_of_lumbar(v2_doc) -> None:
    mujoco, _, model = _compile(with_trunk_coupling(v2_doc, enabled=True))
    data = mujoco.MjData(model)
    data.qpos[model.joint("lumbar_axial_rotation").qposadr[0]] = 0.1
    for _ in range(400):
        mujoco.mj_step(model, data)
    got = data.qpos[model.joint("thoracic_axial_rotation").qposadr[0]]
    lum = data.qpos[model.joint("lumbar_axial_rotation").qposadr[0]]
    assert got == pytest.approx((83 / 17) * lum, rel=0.2)


def test_contracts_reject_v1_and_unknown_axis(v1, v2_doc) -> None:
    with pytest.raises(ValueError):
        with_trunk_coupling(v1)
    with pytest.raises(ValueError):
        with_trunk_coupling(v2_doc, law_id="no_such_law")
    with pytest.raises(TypeError):
        with_trunk_coupling(v2_doc, enabled="yes")  # type: ignore[arg-type]


def test_coupling_creates_the_equality_block_when_absent(v2_doc) -> None:
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        _add_trunk_coupling,
    )

    root = DET.fromstring("<mujoco/>")
    _add_trunk_coupling(root, with_trunk_coupling(v2_doc, enabled=True))
    assert len(root.findall("equality/joint")) == 3
