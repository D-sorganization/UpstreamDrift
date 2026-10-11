"""Structural documentation checks; these do not validate equations."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs/research/model_aware_matching"
TEX = DOC / "model_aware_matching.tex"


@pytest.mark.parametrize("name", ["model_aware_matching.tex", "README.md"])
def test_reference_sources_are_present(name: str) -> None:
    assert (DOC / name).read_text(encoding="utf-8").strip()


def test_tex_is_standalone() -> None:
    text = TEX.read_text(encoding="utf-8")
    assert r"\documentclass" in text and r"\end{document}" in text
    assert not re.search(r"\\(?:input|include)\s*\{", text)


def test_governance_and_cross_links() -> None:
    text = TEX.read_text(encoding="utf-8")
    assert "manuals/upstreamdrift" in text
    assert "separate research and experiment record" in text.lower()
    assert "simscape\\_matching\\_reference" in text
    assert "#11421" in text or "\\#11421" in text
    gs3dx = (
        ROOT
        / "docs/research/simscape_matching_reference/simscape_matching_reference.tex"
    )
    assert "model\\_aware\\_matching" in gs3dx.read_text(encoding="utf-8")


def test_reference_names_the_implementation_and_gates() -> None:
    text = TEX.read_text(encoding="utf-8")
    assert "estimation/mosaic" in text
    for gate in ("Kinematic whiteness", "Open-loop replay", "Observability"):
        assert gate in text
    assert (ROOT / "src/shared/python/estimation/mosaic/outer_solve.py").exists()
