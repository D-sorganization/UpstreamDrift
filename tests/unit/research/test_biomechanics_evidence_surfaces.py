"""Tests for generated reviewer and paper biomechanics evidence surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.biomechanics_evidence_surfaces import (
    BRIDGE_REL,
    PAPER_FRAGMENT_REL,
    REVIEWER_SURFACE_REL,
    SOURCE_REGISTER_REL,
    render_paper_fragment,
    render_reviewer_surface,
)

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.unit


def _load(relative: Path) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_generated_biomechanics_surfaces_match_machine_authorities() -> None:
    bridge = _load(BRIDGE_REL)
    sources = _load(SOURCE_REGISTER_REL)

    reviewer = render_reviewer_surface(bridge, sources)
    paper = render_paper_fragment(bridge, sources)

    assert (ROOT / REVIEWER_SURFACE_REL).read_text(encoding="utf-8") == reviewer
    assert (ROOT / PAPER_FRAGMENT_REL).read_text(encoding="utf-8") == paper


def test_reviewer_surface_round_trips_every_registered_record() -> None:
    bridge = _load(BRIDGE_REL)
    sources = _load(SOURCE_REGISTER_REL)
    rendered = render_reviewer_surface(bridge, sources)

    for collection, identifier in (
        (bridge["modalities"], "modality_id"),
        (bridge["mechanisms"], "mechanism_id"),
        (bridge["transportability"], "dimension_id"),
        (sources["sources"], "source_id"),
        (sources["coverage"], "domain_id"),
    ):
        for record in collection:
            assert f"`{record[identifier]}`" in rendered
