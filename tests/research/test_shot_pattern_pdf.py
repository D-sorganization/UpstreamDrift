"""Research PDF renderer preserves document structure and publication safety."""

from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

pytest.importorskip("reportlab")
pypdf = pytest.importorskip("pypdf")

from scripts.research.shot_pattern_analysis import build_pdf

pytestmark = pytest.mark.unit


def test_math_normalization_preserves_supported_equation_semantics():
    text = r"\mathbf n_0+\boldsymbol\omega+\frac12"
    assert build_pdf.normalize_math_for_mathtext(text) == (
        r"\mathbf{n}_0+\boldsymbol{\omega}+\frac{1}{2}"
    )


def test_rendered_pdf_has_headings_link_math_wrapped_table_and_image(tmp_path):
    from PIL import Image

    image = tmp_path / "diagram.png"
    Image.new("RGB", (1600, 900), "navy").save(image)
    source = tmp_path / "study.md"
    source.write_text(
        "# Shot Pattern Study\n\n"
        "## Methods and Evidence\n\n"
        "A [source reference](https://example.org/research) explains "
        r"$\mathbf n_0+\frac12$"
        " in the model.\n\n"
        r"$$\frac{df}{d\theta}=\boldsymbol\omega+\frac12$$"
        "\n\n"
        "| Club | Assumption and Boundary |\n| --- | --- |\n"
        f"| Driver | {'Detailed documented assumption with source evidence. ' * 9} |\n\n"
        "![Overhead Diagram](diagram.png)\n\n"
        "1. First Controlled Case\n2. Second Controlled Case\n\n"
        "## Findings\n\n"
        + ("This is a documented conditional finding with limitations.\n\n" * 45),
        encoding="utf-8",
    )
    output = tmp_path / "study.pdf"
    result = build_pdf.render_markdown_to_pdf(source, output)
    assert result == output
    reader = pypdf.PdfReader(output)
    assert len(reader.pages) >= 2
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "Shot Pattern Study" in text
    assert "Methods and Evidence" in text
    assert "Assumption and Boundary" in text
    assert "Findings" in text
    assert "1. First Controlled Case" in text
    assert "2. Second Controlled Case" in text
    assert "Page 1" in text
    assert any(
        annotation.get_object().get("/A", {}).get("/URI")
        == "https://example.org/research"
        for page in reader.pages
        for annotation in page.get("/Annots", [])
    )
    assert any(page.get("/Resources", {}).get("/XObject") for page in reader.pages)
    assert reader.outline


def test_build_failure_preserves_existing_output(tmp_path, monkeypatch):
    source = tmp_path / "study.md"
    source.write_text("# Study\n")
    output = tmp_path / "study.pdf"
    output.write_bytes(b"existing publication")

    def broken(*args, **kwargs):
        raise RuntimeError("pandoc failed")

    monkeypatch.setattr(build_pdf, "_pandoc_ast", broken)
    with pytest.raises(RuntimeError, match="pandoc failed"):
        build_pdf.render_markdown_to_pdf(source, output)
    assert output.read_bytes() == b"existing publication"


def test_source_and_pandoc_are_required(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_pdf.render_markdown_to_pdf(tmp_path / "absent.md", tmp_path / "out.pdf")
    source = tmp_path / "study.md"
    source.write_text("# Study\n")
    with pytest.raises(FileNotFoundError, match="pandoc"):
        build_pdf.render_markdown_to_pdf(
            source, tmp_path / "out.pdf", pandoc_binary=Path("/missing/pandoc")
        )


def test_pdf_runtime_is_declared_as_optional_extra():
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    dependencies = project["project"]["optional-dependencies"]["shot-pattern-report"]
    assert any(item.startswith("reportlab>=4") for item in dependencies)
    assert any(item.startswith("pypdf>=6") for item in dependencies)


def test_identical_markdown_produces_identical_pdf_bytes(tmp_path):
    source = tmp_path / "study.md"
    source.write_text(
        "# Reproducible Study\n\nA measured $x$ value.\n", encoding="utf-8"
    )
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    build_pdf.render_markdown_to_pdf(source, first)
    build_pdf.render_markdown_to_pdf(source, second)
    assert first.read_bytes() == second.read_bytes()
