"""Tests verifying synchronization of Same-Input Parity documentation across references."""

from __future__ import annotations

from pathlib import Path
from typing import Final
import pytest

pytestmark = pytest.mark.unit

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[4]
TEX_PATH: Final[Path] = (
    REPO_ROOT / "docs/research/same_input_parity/same_input_parity.tex"
)
QMD_PATH: Final[Path] = (
    REPO_ROOT / "manuals/upstreamdrift/chapters/16-same-input-parity.qmd"
)


def _read_reference_contents() -> tuple[str, str]:
    """Read both reference files and return raw text contents.

    Returns:
        tuple[str, str]: Raw LaTeX and Quarto reference contents.
    """
    assert TEX_PATH.is_file(), f"Expected LaTeX reference at {TEX_PATH}"
    assert QMD_PATH.is_file(), f"Expected Quarto chapter at {QMD_PATH}"

    tex_text: str = TEX_PATH.read_text(encoding="utf-8")
    qmd_text: str = QMD_PATH.read_text(encoding="utf-8")
    return tex_text, qmd_text


def _normalize_latex(content: str) -> str:
    """Normalize common LaTeX escape sequences for uniform token assertion."""
    return content.replace(r"\_", "_").replace(r"\#", "#")


def test_reference_files_exist_and_document_acceptance_levels() -> None:
    """Both references must exist and document acceptance levels L0 through L3."""
    tex_text, qmd_text = _read_reference_contents()
    levels: list[str] = ["L0", "L1", "L2", "L3"]

    for level in levels:
        assert level in tex_text, f"Missing acceptance level {level} in {TEX_PATH}"
        assert level in qmd_text, f"Missing acceptance level {level} in {QMD_PATH}"


def test_reference_files_document_engine_equivalency_contract() -> None:
    """Both references must document the 3-Tier Engine Equivalency Contract (#11709)."""
    tex_text, qmd_text = _read_reference_contents()
    normalized_tex: str = _normalize_latex(tex_text)

    contract_tokens: list[str] = [
        "tier_1_rigid_body",
        "tier_2_soft_contact",
        "tier_3_musculoskeletal",
        "are_engines_tier_compatible",
        "#11709",
    ]

    for token in contract_tokens:
        assert token in normalized_tex, (
            f"Missing equivalency contract token '{token}' in {TEX_PATH}"
        )
        assert token in qmd_text, (
            f"Missing equivalency contract token '{token}' in {QMD_PATH}"
        )


def test_reference_files_document_subject_mass_hand_scaling_policy() -> None:
    """Both references must document the subject-mass hand-scaling policy (#12300)."""
    tex_text, qmd_text = _read_reference_contents()
    normalized_tex: str = _normalize_latex(tex_text)

    hand_scaling_tokens: list[str] = [
        "#12300",
        "match_subject_mass",
        "LHand",
        "RHand",
    ]

    for token in hand_scaling_tokens:
        assert token in normalized_tex, (
            f"Missing hand-scaling policy token '{token}' in {TEX_PATH}"
        )
        assert token in qmd_text, (
            f"Missing hand-scaling policy token '{token}' in {QMD_PATH}"
        )


def test_reference_files_preserve_registry_and_schema_references() -> None:
    """Both references must preserve the registry blocker and schema version identifiers."""
    tex_text, qmd_text = _read_reference_contents()
    normalized_tex: str = _normalize_latex(tex_text)

    schema_and_registry_tokens: list[str] = [
        "UP-D1-same-input-parity-inventory",
        "same-input-bundle/v1",
    ]

    for token in schema_and_registry_tokens:
        assert token in normalized_tex, (
            f"Missing blocker/schema token '{token}' in {TEX_PATH}"
        )
        assert token in qmd_text, (
            f"Missing blocker/schema token '{token}' in {QMD_PATH}"
        )
