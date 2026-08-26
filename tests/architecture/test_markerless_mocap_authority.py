"""Executable ownership and qualification contract for markerless mocap."""

from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ADR = ROOT / "docs" / "adr" / "0041-markerless-mocap-consumer-authority.md"
ACCEPTANCE = ROOT / "docs" / "motion_capture" / "markerless_mocap_acceptance.md"
HANDOFF = ROOT / "AGENT_HANDOFF.md"
SPEC = ROOT / "SPEC.md"

pytestmark = pytest.mark.gate


def _text(path: Path) -> str:
    assert path.is_file(), f"missing governed document: {path.relative_to(ROOT)}"
    return path.read_text(encoding="utf-8")


def test_adr_assigns_one_owner_to_each_program_responsibility() -> None:
    """The product repository must consume Tools instead of copying it."""
    text = _text(ADR).casefold()
    required = (
        "tools",
        "upstreamdrift",
        "affinedrift",
        "tools_private",
        "camera and capture contracts",
        "session orchestration",
        "evidence publication",
        "not part of the public runtime",
        "c3d",
    )
    for phrase in required:
        assert phrase in text

    copied_contract = ROOT / "src" / "shared" / "python" / "sidekick" / "lab" / "mocap"
    assert not copied_contract.exists(), (
        "Tools-owned mocap source was copied downstream"
    )


def test_adr_records_license_privacy_and_existing_work_boundaries() -> None:
    """M0 must block incompatible licensing and duplicated historical scope."""
    text = _text(ADR).casefold()
    required = (
        "agpl",
        "subprocess or ipc",
        "raw video",
        "consent",
        "#4558",
        "#8865",
        "tools #4571",
        "simulation viewport",
        "single-camera",
        "model-conditioned",
    )
    for phrase in required:
        assert phrase in text


def test_acceptance_program_fails_closed_across_evidence_levels() -> None:
    """Software, camera, and physical qualification must remain separate."""
    text = _text(ACCEPTANCE)
    for heading in (
        "## Qualification Levels",
        "## Release Outcomes",
        "## Camera-Agnostic Acceptance",
        "## Physical Lab Hold Points",
        "## UI and Accessibility Acceptance",
        "## C3D Compatibility Acceptance",
    ):
        assert heading in text
    folded = text.casefold()
    for phrase in (
        "supported",
        "degraded",
        "blocked",
        "unavailable",
        "synthetic evidence",
        "does not qualify",
        "units",
        "theme",
        "hover",
        "keyboard",
        "uncertainty",
    ):
        assert phrase in folded


def test_spec_and_handoff_point_to_the_current_program() -> None:
    """Agent context must expose the epic, limits, and next dependency."""
    spec = _text(SPEC)
    handoff = _text(HANDOFF)
    assert "Markerless Mocap Program (#9063)" in spec
    assert "Markerless Mocap Program (#9063)" in handoff
    assert "Tools #4706" in handoff
    assert "UpstreamDrift #9069" in handoff
    assert "no physical-lab qualification" in handoff
    assert len(handoff.splitlines()) <= 150
