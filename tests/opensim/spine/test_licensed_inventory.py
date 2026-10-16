"""Asset-gated inventory of the licensed spine models (SPN-8c, #12755).

These tests need the owner's SimTK downloads and never run in CI: without the
asset env var (see ``src/config/spine_model_assets.json``) they skip with a
reason starting ``UNQUALIFIED: <asset_id>``.  A skip is never acceptance.

The per-model level maps for the licensed assets are still empty and
UNVERIFIED (names pending #12749), so even with an asset present these tests
skip until the owner fills ``src/config/spine_model_level_maps.json`` from an
observed inventory.  No SimTK value is stored in the repository; everything
below is read at runtime.
"""

from __future__ import annotations

import pytest

from src.engines.physics_engines.opensim.python.spine.assets import (
    SpineAssetStatus,
    admit_spine_asset,
    resolve_spine_asset,
)
from src.engines.physics_engines.opensim.python.spine.inventory import (
    inventory_spine,
    load_level_maps,
)
from src.engines.physics_engines.opensim.python.spine.passive_elements import (
    extract_passive_elements,
)
from tests.helpers.opensim_support import require_opensim

ASSET_IDS = (
    "bruno2015_fullbody_os4_v2",
    "senteler2016_lumbspine_3dof",
    "senteler2016_lumbspine_30dof",
)
pytestmark = [pytest.mark.requires_opensim, pytest.mark.integration]


def _admitted_path(asset_id: str):
    """Model path of an asset, or skip with an ``UNQUALIFIED`` reason."""
    request = resolve_spine_asset(asset_id)
    if request is None:
        pytest.skip(f"UNQUALIFIED: {asset_id} asset env var not set")
    receipt = admit_spine_asset(request)
    if receipt.status is not SpineAssetStatus.QUALIFIED:
        pytest.skip(f"UNQUALIFIED: {asset_id} admission is {receipt.status.value}")
    return request.source_path


def _verified_map_or_skip(asset_id: str):
    entry = load_level_maps()[asset_id]
    if not entry.verified or not entry.levels:
        pytest.skip(f"UNQUALIFIED: {asset_id} level map unverified (names pending)")
    return entry


@pytest.mark.parametrize("asset_id", ASSET_IDS)
def test_axis_alignment_within_one_degree(asset_id: str) -> None:
    path = _admitted_path(asset_id)
    entry = _verified_map_or_skip(asset_id)
    require_opensim()
    inv = inventory_spine(path, asset_id, level_map=entry, strict=True)
    assert inv.axis_ok, inv.axis_failures


def test_senteler_30dof_passive_elements_are_read_at_runtime() -> None:
    asset_id = "senteler2016_lumbspine_30dof"
    path = _admitted_path(asset_id)
    entry = _verified_map_or_skip(asset_id)
    require_opensim()
    inv = inventory_spine(path, asset_id, level_map=entry)
    found = extract_passive_elements(path, inv)
    assert found, "no bushing matched any mapped level"
    for spec in found.values():
        assert spec.bushing_type
