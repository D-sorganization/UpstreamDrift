"""ADR-0046 Stage 2 (G2, #9349): both launch-monitor tiles name their shared engine.

ADR-0046's decision keeps both surfaces — the UD 9-tab workbench
(``launch_monitor_analytics``) and the Impact Explorer's launch-monitor tab
(``rate_of_closure``) — and requires that the launcher "keeps both tiles with
descriptions that state the relationship ('the same analytics engine')". With
every ``port-up``/``merge`` module retired onto the canonical layer (#9348),
that sentence is the last Stage 2 deliverable this repository owns, and this
file is what keeps it honest on both launchers: the desktop launcher renders
``models.yaml`` descriptions, the web launcher renders
``launcher_manifest.json`` descriptions, so both are asserted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "src" / "config"
_SHARED_ENGINE_PHRASE = "the same analytics engine"
_TILE_IDS = ("launch_monitor_analytics", "rate_of_closure")


def _manifest_descriptions() -> dict[str, str]:
    raw = json.loads(
        (_CONFIG_DIR / "launcher_manifest.json").read_text(encoding="utf-8")
    )
    return {tile["id"]: tile["description"] for tile in raw["tiles"]}


def _registry_descriptions() -> dict[str, str]:
    raw = yaml.safe_load((_CONFIG_DIR / "models.yaml").read_text(encoding="utf-8"))
    return {model["id"]: model["description"] for model in raw["models"]}


@pytest.mark.parametrize("tile_id", _TILE_IDS)
@pytest.mark.parametrize(
    "descriptions",
    [_manifest_descriptions, _registry_descriptions],
    ids=["launcher_manifest.json", "models.yaml"],
)
def test_launch_monitor_tiles_state_the_shared_engine_relationship(
    tile_id: str, descriptions
) -> None:
    """Each tile, on each launcher, says the two share one analytics engine."""
    description = descriptions()[tile_id]
    assert _SHARED_ENGINE_PHRASE in description, (
        f"{tile_id!r} must state the ADR-0046 relationship "
        f"({_SHARED_ENGINE_PHRASE!r}); got: {description!r}"
    )
