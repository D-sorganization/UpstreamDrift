"""Part library browsing API and bundled parts (CMB-8, #11659)."""

from __future__ import annotations

import pytest

from src.shared.python.model_generation.editor.attachment_ports import PortType
from src.tools.model_explorer.part_catalog import (
    PartCatalog,
    can_mate,
    find_mating_plug,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def catalog() -> PartCatalog:
    return PartCatalog.bundled()


def test_catalog_covers_the_required_part_families(catalog: PartCatalog) -> None:
    categories = {c for c, _ in catalog.categories()}
    assert {"club", "limb", "head", "shoe", "robot_arm", "torso"} <= categories


def test_driver_comes_from_the_repository_urdf(catalog: PartCatalog) -> None:
    driver = catalog.get("club_driver")
    assert driver.source.endswith("driver.urdf")
    assert driver.root_link() == "base_link"
    assert driver.plug_ports()[0].port_type is PortType.GRIP


def test_every_bundled_part_loads_with_a_single_root(catalog: PartCatalog) -> None:
    for part_id in catalog.all_ids():
        part = catalog.get(part_id)
        assert part.root_link() in part.load_model().links
        assert part.mass_kg() >= 0.0


def test_list_parts_filters_by_category_and_query(catalog: PartCatalog) -> None:
    clubs = catalog.list_parts(category="club")
    assert {p.part_id for p in clubs} == {"club_driver", "club_iron"}
    assert [p.part_id for p in catalog.list_parts(query="iron")] == ["club_iron"]
    assert catalog.list_parts(query="no such thing") == ()


def test_compatible_with_filters_by_port_rules(catalog: PartCatalog) -> None:
    torso = catalog.get("humanoid_torso")
    left_hip = torso.port("hip_left")
    assert left_hip is not None
    ids = {p.part_id for p in catalog.list_parts(compatible_with=left_hip)}
    assert ids == {"leg_left"}


def test_untyped_host_port_never_mates(catalog: PartCatalog) -> None:
    from src.tools.model_explorer.attachment_manifest import AttachmentPoint

    untyped = AttachmentPoint(name="x", link_name="l", role="legacy")
    plug, verdict = find_mating_plug(untyped, catalog.get("head"))
    assert plug is None and "untyped" in verdict.reason
    assert not can_mate(untyped, catalog.get("head")).ok


def test_payload_limit_blocks_a_heavy_part(catalog: PartCatalog) -> None:
    ankle = catalog.get("leg_left").port("ankle")
    assert ankle is not None
    assert can_mate(ankle, catalog.get("shoe_left")).ok  # 0.6 kg < 2.5 kg


def test_registry_preconditions(catalog: PartCatalog) -> None:
    with pytest.raises(ValueError):
        catalog.register(catalog.get("head"))
    with pytest.raises(KeyError):
        catalog.get("missing")
    with pytest.raises(ValueError):
        catalog.list_parts(query=None)  # type: ignore[arg-type]
