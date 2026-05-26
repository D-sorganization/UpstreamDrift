"""Regression guards for URDF subsystem governance docs.

Issue #6094 found ADR-0020 stuck in ``Proposed`` even though #4521 was
closed with Option B (Layer) recorded as the repo decision. These tests keep
the ADR, boundary doc, and package-level documentation aligned.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(*parts: str) -> str:
    path = _REPO_ROOT.joinpath(*parts)
    assert path.exists(), f"{path} must exist"
    return path.read_text(encoding="utf-8")


class TestUrdfAdr0020:
    """ADR-0020 must reflect the accepted URDF layering decision."""

    def test_status_is_accepted(self) -> None:
        text = _read("docs", "adr", "0020-canonical-urdf-subsystem.md")
        assert "Status:** Accepted" in text or "Status: Accepted" in text

    def test_records_option_b_layering(self) -> None:
        text = _read("docs", "adr", "0020-canonical-urdf-subsystem.md")
        assert "Option B" in text
        assert "Layer" in text
        assert "#4521" in text

    def test_has_decision_in_practice_section(self) -> None:
        text = _read("docs", "adr", "0020-canonical-urdf-subsystem.md")
        assert "## Decision in practice" in text
        assert "model_generation.humanoid" in text


class TestUrdfBoundaryDoc:
    """The boundary doc must describe the accepted layering, not a split stack."""

    def test_boundary_doc_exists(self) -> None:
        assert (
            _REPO_ROOT / "docs" / "architecture" / "URDF_SUBSYSTEM_BOUNDARY.md"
        ).exists()

    def test_boundary_doc_marks_model_generation_as_canonical_toolkit(self) -> None:
        text = _read("docs", "architecture", "URDF_SUBSYSTEM_BOUNDARY.md")
        assert "canonical low-level URDF / mesh /" in text
        assert "inertia toolkit" in text
        assert "humanoid_character_builder" in text
        assert "anthropometric domain layer" in text

    def test_boundary_doc_mentions_compatibility_facade(self) -> None:
        text = _read("docs", "architecture", "URDF_SUBSYSTEM_BOUNDARY.md")
        assert "model_generation.humanoid" in text
        assert "compatibility facade" in text


class TestUrdfPackageDocs:
    """Package docstrings must match the accepted layering decision."""

    def test_humanoid_character_builder_docstring_describes_domain_layer(self) -> None:
        text = _read(
            "src", "shared", "python", "humanoid_character_builder", "__init__.py"
        )
        assert "anthropometric domain layer" in text
        assert "model_generation" in text

    def test_shared_python_index_describes_canonical_toolkit(self) -> None:
        text = _read("src", "shared", "python", "__init__.py")
        assert "canonical generic URDF/MJCF toolkit" in text


class TestAdrIndex:
    """The ADR index must reflect ADR-0020's accepted status."""

    def test_adr_index_marks_0020_accepted(self) -> None:
        text = _read("docs", "adr", "README.md")
        assert "[0020](0020-canonical-urdf-subsystem.md)" in text
        assert "Canonical URDF subsystem" in text
        assert "Accepted | 2026-05-08" in text
