"""Structural documentation checks; these do not validate equations."""

from pathlib import Path
import re
import pytest

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs/research/simscape_matching_reference"
TEX = DOC / "simscape_matching_reference.tex"


@pytest.mark.parametrize("name", ["simscape_matching_reference.tex", "README.md"])
def test_reference_sources_are_present(name: str) -> None:
    assert (DOC / name).read_text(encoding="utf-8").strip()


def test_tex_is_standalone() -> None:
    text = TEX.read_text(encoding="utf-8")
    assert r"\documentclass" in text and r"\end{document}" in text
    assert not re.search(r"\\(?:input|include)\s*\{", text)


def test_governance_preserves_canonical_manual() -> None:
    text = TEX.read_text(encoding="utf-8")
    assert "manuals/upstreamdrift" in text
    assert "separate research and experiment record" in text.lower()
    policy = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "Modeling Reference Documentation" in policy
    assert "simscape_matching_reference.tex" in policy


def test_no_private_absolute_paths_in_public_reference() -> None:
    text = TEX.read_text(encoding="utf-8")
    assert not re.search(r"[A-Z]:[\\/]Users[\\/]", text)
    assert "capture-A" in text and "capture-O" in text
