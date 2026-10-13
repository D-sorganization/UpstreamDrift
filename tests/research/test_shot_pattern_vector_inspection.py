"""Unit and specification tests for vector inspection module."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from defusedxml.common import EntitiesForbidden

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pypdf
except (ImportError, OSError) as exc:
    pytest.skip(
        f"Vector review dependencies unavailable: {exc}", allow_module_level=True
    )

from scripts.research.shot_pattern_analysis.vector_inspection import (
    inspect_pdf,
    inspect_svg,
    inspect_vector,
)


pytestmark = pytest.mark.unit


def _compute_sha256(content: bytes) -> str:
    """Return SHA256 hex digest for given byte sequence."""
    return hashlib.sha256(content).hexdigest()


def _write_file_and_hash(path: Path, data: bytes) -> str:
    """Write binary data to path and return computed SHA256 digest."""
    path.write_bytes(data)
    return _compute_sha256(data)


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> tuple[Path, str]:
    """Generate a real compact Matplotlib PDF figure fixture under tmp_path."""
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0.0, 1.0, 2.0], [0.0, 2.0, 1.0], label="trajectory")
    ax.set_title("Test Shot Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    pdf_path = tmp_path / "fixture_plot.pdf"
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
        metadata={"Creator": "UpstreamDrift test fixture"},
    )
    plt.close(fig)
    digest = _compute_sha256(pdf_path.read_bytes())
    return pdf_path, digest


@pytest.fixture
def sample_svg_path(tmp_path: Path) -> tuple[Path, str]:
    """Generate a real compact Matplotlib SVG figure fixture under tmp_path."""
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0.0, 1.0, 2.0], [1.0, 3.0, 2.0], label="overlay")
    ax.set_title("Overlay Pattern")
    svg_path = tmp_path / "fixture_plot.svg"
    fig.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
        metadata={"Creator": "UpstreamDrift test fixture"},
    )
    plt.close(fig)
    digest = _compute_sha256(svg_path.read_bytes())
    return svg_path, digest


@pytest.fixture
def sample_pdf_with_image_path(tmp_path: Path) -> tuple[Path, str]:
    """Generate a real Matplotlib PDF containing an embedded raster image."""
    fig, ax = plt.subplots(figsize=(2, 2))
    raster = np.arange(16, dtype=np.uint8).reshape((4, 4))
    ax.imshow(raster, cmap="gray")
    pdf_path = tmp_path / "fixture_image.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    digest = _compute_sha256(pdf_path.read_bytes())
    return pdf_path, digest


@pytest.fixture
def sample_svg_with_image_path(tmp_path: Path) -> tuple[Path, str]:
    """Generate a real Matplotlib SVG containing an embedded raster data image."""
    fig, ax = plt.subplots(figsize=(2, 2))
    raster = np.arange(16, dtype=np.uint8).reshape((4, 4))
    ax.imshow(raster, cmap="gray")
    svg_path = tmp_path / "fixture_image.svg"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    digest = _compute_sha256(svg_path.read_bytes())
    return svg_path, digest


def test_pdf_matplotlib_fixture_positive(
    sample_pdf_path: tuple[Path, str],
) -> None:
    """PDF inspection succeeds on valid Matplotlib single-page output."""
    pdf_path, digest = sample_pdf_path
    diagnostics = inspect_pdf(pdf_path, digest)

    assert diagnostics["format"] == "pdf"
    assert diagnostics["file_sha256"] == digest
    assert diagnostics["expected_sha256"] == digest
    assert diagnostics["visual_approval"] is False
    assert "visual acceptance" in diagnostics["claim"]

    dim = diagnostics["dimensions"]
    assert dim["page_count"] == 1
    assert dim["media_box"]["width"] > 0.0
    assert dim["media_box"]["height"] > 0.0
    assert math.isfinite(dim["media_box"]["width"])
    assert math.isfinite(dim["media_box"]["height"])

    assert "Shot" in diagnostics["text"]
    assert "Trajectory" in diagnostics["text"]
    assert isinstance(diagnostics["font_resources"], list)
    assert len(diagnostics["font_resources"]) >= 1
    # Check that Type3 fonts are catalogued rather than called missing
    subtypes = [f.get("subtype") for f in diagnostics["font_resources"]]
    assert any(st in ("/Type3", "/TrueType", "/Type1") for st in subtypes)
    assert diagnostics["embedded_image_count"] == 0

    assert "python" in diagnostics["runtime_versions"]
    assert "pypdf" in diagnostics["runtime_versions"]
    assert isinstance(diagnostics["source_sha256"], str)
    assert len(diagnostics["source_sha256"]) == 64

    # Verify JSON serializability
    serialized = json.dumps(diagnostics)
    assert isinstance(serialized, str)


def test_pdf_with_image_positive(
    sample_pdf_with_image_path: tuple[Path, str],
) -> None:
    """PDF inspection properly inventories embedded raster images."""
    pdf_path, digest = sample_pdf_with_image_path
    diagnostics = inspect_pdf(pdf_path, digest)
    assert diagnostics["embedded_image_count"] >= 1


def test_pdf_rejects_hash_mismatch(sample_pdf_path: tuple[Path, str]) -> None:
    """PDF inspection rejects pre-read SHA mismatch."""
    pdf_path, _ = sample_pdf_path
    wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000"
    with pytest.raises(ValueError, match="Pre-read SHA256 mismatch"):
        inspect_pdf(pdf_path, wrong_hash)


def test_pdf_rejects_missing_file(tmp_path: Path) -> None:
    """PDF inspection raises FileNotFoundError when path does not exist."""
    missing_path = tmp_path / "nonexistent.pdf"
    with pytest.raises(FileNotFoundError):
        inspect_pdf(missing_path, "0" * 64)


def test_pdf_rejects_malformed_bytes(tmp_path: Path) -> None:
    """PDF inspection rejects malformed bytes with descriptive ValueError."""
    corrupt_path = tmp_path / "corrupt.pdf"
    digest = _write_file_and_hash(corrupt_path, b"%PDF-1.4 malformed header and body")
    with pytest.raises(ValueError, match="Malformed PDF"):
        inspect_pdf(corrupt_path, digest)


def test_pdf_rejects_multipage(tmp_path: Path) -> None:
    """PDF inspection rejects multipage PDF documents."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=100)

    buf = io.BytesIO()
    writer.write(buf)
    data = buf.getvalue()
    multi_path = tmp_path / "multipage.pdf"
    digest = _write_file_and_hash(multi_path, data)

    with pytest.raises(ValueError, match="Expected exactly 1 page"):
        inspect_pdf(multi_path, digest)


def test_pdf_rejects_empty_page_stream(tmp_path: Path) -> None:
    """PDF inspection rejects page with empty content stream."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    writer.write(buf)
    empty_path = tmp_path / "empty_page.pdf"
    digest = _write_file_and_hash(empty_path, buf.getvalue())

    with pytest.raises(ValueError, match="PDF page stream is empty"):
        inspect_pdf(empty_path, digest)


def test_pdf_rejects_invalid_mediabox(tmp_path: Path) -> None:
    """PDF inspection rejects nonpositive or nonfinite box dimensions."""
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=100, height=100)
    stream = pypdf.generic.DecodedStreamObject()
    stream.set_data(b"q Q")
    page[pypdf.generic.NameObject("/Contents")] = stream
    page.mediabox = pypdf.generic.RectangleObject([0, 0, 0, 100])  # width = 0

    buf = io.BytesIO()
    writer.write(buf)
    bad_box_path = tmp_path / "bad_box.pdf"
    digest = _write_file_and_hash(bad_box_path, buf.getvalue())

    with pytest.raises(ValueError, match="Non-positive or non-finite dimensions"):
        inspect_pdf(bad_box_path, digest)


def test_pdf_rejects_encrypted_file(tmp_path: Path) -> None:
    """PDF inspection rejects encrypted files."""
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=100, height=100)
    stream = pypdf.generic.DecodedStreamObject()
    stream.set_data(b"q Q")
    page[pypdf.generic.NameObject("/Contents")] = stream
    writer.encrypt("secret_password")

    buf = io.BytesIO()
    writer.write(buf)
    enc_path = tmp_path / "encrypted.pdf"
    digest = _write_file_and_hash(enc_path, buf.getvalue())

    with pytest.raises(ValueError, match="Encrypted PDF rejected"):
        inspect_pdf(enc_path, digest)


def test_svg_matplotlib_fixture_positive(
    sample_svg_path: tuple[Path, str],
) -> None:
    """SVG inspection succeeds on real Matplotlib SVG output."""
    svg_path, digest = sample_svg_path
    diagnostics = inspect_svg(svg_path, digest)

    assert diagnostics["format"] == "svg"
    assert diagnostics["file_sha256"] == digest
    assert diagnostics["expected_sha256"] == digest
    assert diagnostics["visual_approval"] is False
    assert "visual acceptance" in diagnostics["claim"]

    vp = diagnostics["viewport"]
    assert vp["width"] > 0.0
    assert vp["height"] > 0.0
    assert vp["width_unit"] in ("pt", "px", "in", "mm", "cm", "")
    assert vp["height_unit"] in ("pt", "px", "in", "mm", "cm", "")

    vb = diagnostics["view_box"]
    assert vb is not None
    assert vb["width"] > 0.0
    assert vb["height"] > 0.0
    assert math.isfinite(vb["min_x"])
    assert math.isfinite(vb["min_y"])

    # Comments and IDs extracted
    assert isinstance(diagnostics["comments"], list)
    assert isinstance(diagnostics["ids"], list)
    assert len(diagnostics["ids"]) > 0

    # Matplotlib uses glyph paths in defs and <use xlink:href='#...'>
    assert diagnostics["embedded_image_count"] == 0

    # JSON serializability
    serialized = json.dumps(diagnostics)
    assert isinstance(serialized, str)


def test_svg_with_data_image_positive(
    sample_svg_with_image_path: tuple[Path, str],
) -> None:
    """SVG inspection inventories embedded data:image elements."""
    svg_path, digest = sample_svg_with_image_path
    diagnostics = inspect_svg(svg_path, digest)
    assert diagnostics["embedded_image_count"] >= 1


def test_svg_metadata_namespaces_not_treated_as_external_dependencies(
    tmp_path: Path,
) -> None:
    """RDF metadata namespace URLs are not flagged as external resources."""
    svg_content = b"""<?xml version="1.0" encoding="utf-8" standalone="no"?>
<svg xmlns:xlink="http://www.w3.org/1999/xlink" width="100pt" height="100pt" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <metadata>
    <rdf:RDF xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:cc="http://creativecommons.org/ns#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
      <cc:Work rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <defs>
    <path id="glyph1" d="M0 0 L10 10"/>
  </defs>
  <!-- figure comment -->
  <use xlink:href="#glyph1" x="5" y="5"/>
  <text x="10" y="20">Sample text</text>
</svg>"""
    svg_path = tmp_path / "metadata_test.svg"
    digest = _write_file_and_hash(svg_path, svg_content)

    diagnostics = inspect_svg(svg_path, digest)
    assert "Sample text" in diagnostics["text"]
    assert "figure comment" in diagnostics["comments"]
    assert "glyph1" in diagnostics["ids"]


def test_svg_rejects_hash_mismatch(sample_svg_path: tuple[Path, str]) -> None:
    """SVG inspection rejects pre-read SHA mismatch."""
    svg_path, _ = sample_svg_path
    with pytest.raises(ValueError, match="Pre-read SHA256 mismatch"):
        inspect_svg(svg_path, "0" * 64)


def test_svg_rejects_malformed_xml(tmp_path: Path) -> None:
    """SVG inspection rejects malformed XML."""
    svg_path = tmp_path / "malformed.svg"
    digest = _write_file_and_hash(svg_path, b"<svg><unclosed>")
    with pytest.raises(ValueError, match="Malformed SVG XML"):
        inspect_svg(svg_path, digest)


def test_svg_rejects_non_svg_root(tmp_path: Path) -> None:
    """SVG inspection rejects XML where root element is not <svg>."""
    svg_path = tmp_path / "not_svg.svg"
    digest = _write_file_and_hash(
        svg_path, b"<html><body width='10' height='10'></body></html>"
    )
    with pytest.raises(ValueError, match="Root element is not <svg>"):
        inspect_svg(svg_path, digest)


def test_svg_rejects_dtd_entity_declaration(tmp_path: Path) -> None:
    """SVG inspection explicitly rejects entity declarations."""
    svg_content = b"""<?xml version="1.0"?>
<!DOCTYPE svg [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <text x="10" y="10">&xxe;</text>
</svg>"""
    svg_path = tmp_path / "entity_test.svg"
    digest = _write_file_and_hash(svg_path, svg_content)
    with pytest.raises(
        ValueError, match="XML (entity declaration|DTD internal subset) rejected"
    ):
        inspect_svg(svg_path, digest)


def test_svg_rejects_custom_system_dtd(tmp_path: Path) -> None:
    """SVG inspection rejects custom external system DTD declarations."""
    svg_content = b"""<?xml version="1.0"?>
<!DOCTYPE svg SYSTEM "http://evil.com/evil.dtd">
<svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
</svg>"""
    svg_path = tmp_path / "system_dtd.svg"
    digest = _write_file_and_hash(svg_path, svg_content)
    with pytest.raises(ValueError, match="DTD"):
        inspect_svg(svg_path, digest)


def test_svg_rejects_invalid_viewbox(tmp_path: Path) -> None:
    """SVG inspection rejects nonpositive or nonfinite viewBox."""
    svg_content = b"""<svg width="100" height="100" viewBox="0 0 -50 100" xmlns="http://www.w3.org/2000/svg"></svg>"""
    svg_path = tmp_path / "bad_viewbox.svg"
    digest = _write_file_and_hash(svg_path, svg_content)
    with pytest.raises(ValueError, match="viewBox width and height must be positive"):
        inspect_svg(svg_path, digest)


def test_svg_rejects_unknown_local_reference(tmp_path: Path) -> None:
    """SVG inspection rejects references to non-existent local IDs."""
    svg_content = b"""<svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
  <use xlink:href="#missing_glyph" x="0" y="0"/>
</svg>"""
    svg_path = tmp_path / "unknown_local.svg"
    digest = _write_file_and_hash(svg_path, svg_content)
    with pytest.raises(ValueError, match="Unknown local reference"):
        inspect_svg(svg_path, digest)


def test_svg_rejects_unknown_local_url_reference_in_style(tmp_path: Path) -> None:
    """SVG inspection rejects url(#...) pointing to missing ID in presentation attribute."""
    svg_content = b"""<svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="10" cy="10" r="5" clip-path="url(#missing_clip)"/>
</svg>"""
    svg_path = tmp_path / "unknown_clip.svg"
    digest = _write_file_and_hash(svg_path, svg_content)
    with pytest.raises(ValueError, match="Unknown local reference"):
        inspect_svg(svg_path, digest)


def test_svg_rejects_external_image_resource(tmp_path: Path) -> None:
    """SVG inspection rejects external network/file image links."""
    svg_content = b"""<svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <image href="http://example.com/raster.png" width="100" height="100"/>
</svg>"""
    svg_path = tmp_path / "external_image.svg"
    digest = _write_file_and_hash(svg_path, svg_content)
    with pytest.raises(ValueError, match="External resource reference rejected"):
        inspect_svg(svg_path, digest)


def test_svg_rejects_external_style_import(tmp_path: Path) -> None:
    """SVG inspection rejects external style import."""
    svg_content = b"""<svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <style>@import url("https://fonts.googleapis.com/css?family=Open+Sans");</style>
</svg>"""
    svg_path = tmp_path / "external_style.svg"
    digest = _write_file_and_hash(svg_path, svg_content)
    with pytest.raises(ValueError, match="External style or font resource rejected"):
        inspect_svg(svg_path, digest)


def test_inspect_vector_dispatcher_pdf(sample_pdf_path: tuple[Path, str]) -> None:
    """inspect_vector properly dispatches PDF paths."""
    pdf_path, digest = sample_pdf_path
    res = inspect_vector(pdf_path, digest)
    assert res["format"] == "pdf"


def test_inspect_vector_dispatcher_svg(sample_svg_path: tuple[Path, str]) -> None:
    """inspect_vector properly dispatches SVG paths."""
    svg_path, digest = sample_svg_path
    res = inspect_vector(svg_path, digest)
    assert res["format"] == "svg"


def test_inspect_vector_rejects_unsupported_suffix(tmp_path: Path) -> None:
    """inspect_vector rejects non-PDF and non-SVG extensions."""
    png_path = tmp_path / "image.png"
    digest = _write_file_and_hash(png_path, b"\x89PNG\r\n\x1a\n")
    with pytest.raises(ValueError, match="Unsupported vector format"):
        inspect_vector(png_path, digest)


def test_post_read_mutation_rejection(
    monkeypatch: Any, sample_pdf_path: tuple[Path, str]
) -> None:
    """Rehash after read rejects file that mutates during read."""
    pdf_path, digest = sample_pdf_path

    # Simulate mutating the file on disk during reading
    from scripts.research.shot_pattern_analysis import vector_inspection

    original_read = vector_inspection._read_pdf_page

    def mutating_read(path: Path) -> Any:
        page, reader, warnings = original_read(path)
        # Mutate the file on disk before post-read hash
        path.write_bytes(path.read_bytes() + b"corruption")
        return page, reader, warnings

    monkeypatch.setattr(vector_inspection, "_read_pdf_page", mutating_read)

    with pytest.raises(ValueError, match="Post-read SHA256 mutation detected"):
        inspect_pdf(pdf_path, digest)


@pytest.mark.parametrize(
    "xml",
    [
        '<svg xmlns="urn:not-svg" width="1" height="1"/>',
        '<?xml-stylesheet href="https://example.invalid/a.css"?><svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"><image href="data:text/html;base64,SGk="/></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"><path id="a"/><path id="a"/></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="1"/>',
    ],
)
def test_independent_svg_contract_rejections(tmp_path: Path, xml: str) -> None:
    path = tmp_path / "case.svg"
    digest = _write_file_and_hash(path, xml.encode())
    with pytest.raises(ValueError):
        inspect_svg(path, digest)


def test_local_style_and_text_content(tmp_path: Path) -> None:
    xml = """<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1">
    <metadata><title>Not Visible Metadata</title></metadata>
    <defs><clipPath id="a"/></defs><style>.x { clip-path: url(#a); }</style>
    <!-- Separate Caption --><text>Visible<tspan> Tail</tspan> After</text></svg>"""
    path = tmp_path / "local.svg"
    result = inspect_svg(path, _write_file_and_hash(path, xml.encode()))
    assert result["text"] == ["Visible Tail After"]
    assert result["comments"] == ["Separate Caption"]
    assert result["visual_approval"] is False


def test_pdf_late_warning_capture_and_handler_restoration(
    sample_pdf_path: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import logging
    import warnings

    original = pypdf.PageObject.extract_text
    logger = logging.getLogger("pypdf")
    before = list(logger.handlers)

    def noisy(self: pypdf.PageObject, *args: Any, **kwargs: Any) -> str:
        warnings.warn("late extraction warning", UserWarning, stacklevel=1)
        logger.warning("late resource warning")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pypdf.PageObject, "extract_text", noisy)
    result = inspect_pdf(*sample_pdf_path)
    assert "late extraction warning" in result["warnings"]
    assert "late resource warning" in result["warnings"]
    assert logger.handlers == before


def test_svg_mutation_during_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.research.shot_pattern_analysis import vector_inspection as module

    path = tmp_path / "mutating.svg"
    digest = _write_file_and_hash(
        path, b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>'
    )
    original = module._validate_svg_resources_and_references

    def mutate(root: Any) -> Any:
        result = original(root)
        path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(module, "_validate_svg_resources_and_references", mutate)
    with pytest.raises(ValueError, match="mutation"):
        inspect_svg(path, digest)


def test_pdf_warning_handler_restored_after_extraction_error(
    sample_pdf_path: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import logging

    logger = logging.getLogger("pypdf")
    before = list(logger.handlers)

    def fail(*args: Any, **kwargs: Any) -> str:
        raise ValueError("extraction failure")

    monkeypatch.setattr(pypdf.PageObject, "extract_text", fail)
    with pytest.raises(ValueError, match="extraction failure"):
        inspect_pdf(*sample_pdf_path)
    assert logger.handlers == before


@pytest.mark.parametrize(
    "declaration",
    [
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">',
        "",
    ],
)
def test_inert_matplotlib_doctype_and_viewbox(tmp_path: Path, declaration: str) -> None:
    path = tmp_path / "doctype.svg"
    xml = (
        declaration
        + '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 20"><defs><path id="glyph"/></defs><use href="#glyph"/></svg>'
    )
    result = inspect_svg(path, _write_file_and_hash(path, xml.encode()))
    assert result["view_box"]["width"] == 10
    assert result["ids"] == ["glyph"]


def test_svg_tree_parser_rejects_entity_expansion() -> None:
    """The second XML parse must reject entities even without the Expat preflight."""
    from scripts.research.shot_pattern_analysis import vector_inspection as module

    xml = (
        b'<!DOCTYPE svg [<!ENTITY payload "EXPANDED">]>'
        b'<svg xmlns="http://www.w3.org/2000/svg">&payload;</svg>'
    )
    with pytest.raises(EntitiesForbidden):
        module.ET.fromstring(xml)


def test_svg_inspector_uses_only_hardened_xml_parsers() -> None:
    """Untrusted SVG bytes must never reach a raw stdlib XML parser."""
    from scripts.research.shot_pattern_analysis import vector_inspection as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    imports = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported_modules = [
        alias.name
        for node in imports
        if isinstance(node, ast.Import)
        for alias in node.names
    ] + [node.module or "" for node in imports if isinstance(node, ast.ImportFrom)]
    assert all(
        name != "xml" and not name.startswith("xml.") for name in imported_modules
    )


@pytest.mark.parametrize("subset", ["[]", "[ <!-- inert --> ]"])
def test_svg_rejects_even_inert_internal_dtd_subset(
    tmp_path: Path, subset: str
) -> None:
    """The permitted Matplotlib DOCTYPE must not gain an internal subset."""
    xml = (
        f"<!DOCTYPE svg {subset}>"
        '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>'
    )
    path = tmp_path / "internal_subset.svg"
    digest = _write_file_and_hash(path, xml.encode())
    with pytest.raises(ValueError, match="DTD internal subset rejected"):
        inspect_svg(path, digest)
