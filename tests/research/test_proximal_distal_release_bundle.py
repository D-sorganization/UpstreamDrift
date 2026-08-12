from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.release_bundle import (
    build_release_manifest,
    validate_release_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
pytestmark = pytest.mark.unit


def test_release_manifest_has_model_ladder_presets_and_neutral_boundaries() -> None:
    manifest = build_release_manifest(ROOT)
    assert list(manifest["presets"]) == [
        "double_pendulum",
        "forward_two_hand",
        "moving_base_flexible_club",
        "shaft_beam_reference",
        "spatial_common_state",
        "spatial_forward_contact",
        "uncertainty_control",
        "experimental_readiness",
    ]
    assert manifest["claims"]["human_experimental"] == "untested"
    assert (
        manifest["claims"]["distributed_shaft_modal_reduction"]
        == "supported_on_synthetic_structural_case"
    )
    assert (
        manifest["claims"]["passive_negative_couple_spatial_forward"]
        == "supported_at_declared_reduced_contact_tier"
    )
    assert (
        manifest["archive"]["persistent_identifier_status"]
        == "pending_external_archive"
    )
    assert manifest["resource_framing"] == "neutral_open_research_resource"


def test_committed_release_manifest_matches_builder_and_validates() -> None:
    committed = json.loads((ARTICLE / "release_manifest.json").read_text())
    assert committed == build_release_manifest(ROOT)
    report = validate_release_manifest(ROOT, committed)
    assert report["valid"]
    assert report["mismatches"] == []


def test_validator_fails_closed_on_tampered_file(tmp_path: Path) -> None:
    manifest = build_release_manifest(ROOT)
    copied = tmp_path / "copy.json"
    copied.write_text("{}\n", encoding="utf-8")
    first = next(iter(manifest["artifacts"]))
    manifest["artifacts"] = {str(copied): manifest["artifacts"][first]}
    with pytest.raises(ValueError, match="release manifest validation failed"):
        validate_release_manifest(ROOT, manifest)


def test_checksum_file_is_sorted_and_covers_every_artifact() -> None:
    manifest = build_release_manifest(ROOT)
    lines = (ARTICLE / "CHECKSUMS.sha256").read_text().splitlines()
    assert lines == sorted(lines, key=lambda item: item.split("  ", 1)[1])
    assert len(lines) == len(manifest["artifacts"])
    assert {line.split("  ", 1)[1] for line in lines} == set(manifest["artifacts"])
