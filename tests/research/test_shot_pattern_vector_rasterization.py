"""Tests for vector rasterization helper."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import pypdf  # noqa: F401
except (ImportError, OSError) as exc:
    pytest.skip(
        f"Vector review dependencies unavailable: {exc}", allow_module_level=True
    )

from scripts.research.shot_pattern_analysis.vector_rasterization import render_vector

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.name != "posix", reason="Fault-injection fixtures require POSIX shell"
    ),
]

NATIVE_TOOLS = json.loads(os.environ.get("P2_VECTOR_TOOLS", "{}"))


def _compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest for a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _create_sample_pdf(path: Path) -> str:
    """Create a minimal 2x3 inch Matplotlib PDF and return its SHA-256."""
    fig, ax = plt.subplots(figsize=(2.0, 3.0))
    ax.plot([0.0, 1.0], [0.0, 1.0], label="line")
    ax.set_title("Test PDF")
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return _compute_sha256(path)


def _create_sample_svg(path: Path) -> str:
    """Create a minimal 2x3 inch Matplotlib SVG and return its SHA-256."""
    fig, ax = plt.subplots(figsize=(2.0, 3.0))
    ax.plot([0.0, 1.0], [0.0, 1.0], label="line")
    ax.set_title("Test SVG")
    fig.savefig(path, format="svg")
    plt.close(fig)
    return _compute_sha256(path)


def _make_executable(script_path: Path, content: str) -> None:
    """Write shell script and make it executable."""
    script_path.write_text(content, encoding="utf-8")
    st = os.stat(script_path)
    os.chmod(script_path, st.st_mode | stat.S_IEXEC)


def test_negative_expected_sha256_mismatch(tmp_path: Path) -> None:
    """Expected SHA-256 mismatch must fail before any render invocation."""
    pdf_path = tmp_path / "valid.pdf"
    _create_sample_pdf(pdf_path)
    out_dir = tmp_path / "out_mismatch"
    bad_sha = "0" * 64

    with pytest.raises(ValueError, match="mismatch"):
        render_vector(pdf_path, bad_sha, out_dir, tool_paths=NATIVE_TOOLS)

    assert not out_dir.exists(), (
        "Output directory should not be created on hash mismatch"
    )


def test_negative_unsupported_extension(tmp_path: Path) -> None:
    """Unsupported file extension must be rejected."""
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("not a vector file", encoding="utf-8")
    sha = _compute_sha256(txt_path)
    out_dir = tmp_path / "out_unsupported"

    with pytest.raises(ValueError, match="Unsupported vector format"):
        render_vector(txt_path, sha, out_dir, tool_paths=NATIVE_TOOLS)

    assert not out_dir.exists()


def test_negative_rejected_output_reuse(tmp_path: Path) -> None:
    """Existing output directory must be rejected to prevent overwrites."""
    pdf_path = tmp_path / "valid.pdf"
    sha = _create_sample_pdf(pdf_path)
    existing_dir = tmp_path / "already_exists"
    existing_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        render_vector(pdf_path, sha, existing_dir, tool_paths=NATIVE_TOOLS)


def test_negative_external_renderer_nonzero(tmp_path: Path) -> None:
    """Nonzero renderer exit code must return failure diagnostics and preserve output dir."""
    pdf_path = tmp_path / "sample.pdf"
    sha = _create_sample_pdf(pdf_path)
    out_dir = tmp_path / "out_nonzero"

    stub_script = tmp_path / "failing_pdftoppm.sh"
    _make_executable(
        stub_script,
        "#!/bin/sh\necho 'pdftoppm simulated fatal error' >&2\nexit 42\n",
    )

    record = render_vector(
        pdf_path,
        sha,
        out_dir,
        tool_paths={"pdftoppm": stub_script},
    )

    assert record["success"] is False
    assert record["visual_approval"] is False
    assert record["exit_code"] == 42
    assert "simulated fatal error" in record["stderr"]
    assert record["output_png_path"] is None
    assert record["output_dimensions"] is None
    assert out_dir.is_dir(), "Output directory must be preserved for post-mortem review"


def test_negative_successful_exit_without_png(tmp_path: Path) -> None:
    """Renderer exiting 0 without creating the target PNG must be flagged as failure."""
    pdf_path = tmp_path / "sample.pdf"
    sha = _create_sample_pdf(pdf_path)
    out_dir = tmp_path / "out_no_png"

    stub_script = tmp_path / "no_png_pdftoppm.sh"
    _make_executable(stub_script, "#!/bin/sh\nexit 0\n")

    record = render_vector(
        pdf_path,
        sha,
        out_dir,
        tool_paths={"pdftoppm": stub_script},
    )

    assert record["success"] is False
    assert record["visual_approval"] is False
    assert record["exit_code"] == 0
    assert record["output_png_path"] is None
    assert "Expected output PNG not found" in record["error"]
    assert out_dir.is_dir()


def test_negative_successful_exit_with_undecodable_png(tmp_path: Path) -> None:
    """Renderer writing non-PNG corrupt bytes must be flagged as failure."""
    pdf_path = tmp_path / "sample.pdf"
    sha = _create_sample_pdf(pdf_path)
    out_dir = tmp_path / "out_corrupt_png"

    stub_script = tmp_path / "corrupt_pdftoppm.sh"
    _make_executable(
        stub_script,
        '#!/bin/sh\nif [ -n "$8" ]; then echo "NOT_A_VALID_PNG" > "$8.png"; fi\nexit 0\n',
    )

    record = render_vector(
        pdf_path,
        sha,
        out_dir,
        tool_paths={"pdftoppm": stub_script},
    )

    assert record["success"] is False
    assert record["visual_approval"] is False
    assert record["exit_code"] == 0
    assert "undecodable" in record["error"].lower()
    assert out_dir.is_dir()


def test_negative_postrender_input_mutation(tmp_path: Path) -> None:
    """Mutation of the input file during subprocess execution must trigger an error."""
    pdf_path = tmp_path / "mutating.pdf"
    sha = _create_sample_pdf(pdf_path)
    out_dir = tmp_path / "out_mutated"

    stub_script = tmp_path / "mutating_pdftoppm.sh"
    _make_executable(
        stub_script,
        '#!/bin/sh\nif [ -n "$7" ]; then echo "CORRUPTION" >> "$7"; fi\nexit 0\n',
    )

    with pytest.raises(ValueError, match="Post-render SHA256 mutation detected"):
        render_vector(
            pdf_path,
            sha,
            out_dir,
            tool_paths={"pdftoppm": stub_script},
        )


@pytest.mark.skipif(
    not NATIVE_TOOLS,
    reason="Set P2_VECTOR_TOOLS JSON to opt in to native render checks",
)
def test_negative_svg_missing_or_relative_viewport(tmp_path: Path) -> None:
    """SVGs with relative dimensions or missing viewports must be rejected."""
    svg_rel = tmp_path / "relative.svg"
    svg_rel.write_text(
        '<svg width="100%" height="100%" viewBox="0 0 100 100" '
        'xmlns="http://www.w3.org/2000/svg"><rect width="100" height="100" fill="blue"/></svg>',
        encoding="utf-8",
    )
    sha_rel = _compute_sha256(svg_rel)
    out_dir1 = tmp_path / "out_rel"

    with pytest.raises(ValueError):
        render_vector(svg_rel, sha_rel, out_dir1, tool_paths=NATIVE_TOOLS)

    svg_missing = tmp_path / "missing_dim.svg"
    svg_missing.write_text(
        '<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="100" height="100" fill="red"/></svg>',
        encoding="utf-8",
    )
    sha_missing = _compute_sha256(svg_missing)
    out_dir2 = tmp_path / "out_missing"

    with pytest.raises(ValueError, match="viewport"):
        render_vector(svg_missing, sha_missing, out_dir2, tool_paths=NATIVE_TOOLS)


@pytest.mark.skipif(
    not NATIVE_TOOLS,
    reason="Set P2_VECTOR_TOOLS JSON to opt in to native render checks",
)
def test_positive_compact_matplotlib_pdf(tmp_path: Path) -> None:
    """Render real compact Matplotlib PDF fixture to 144 DPI PNG."""
    pdf_path = tmp_path / "figure.pdf"
    sha = _create_sample_pdf(pdf_path)
    out_dir = tmp_path / "pdf_raster_dir"

    record = render_vector(pdf_path, sha, out_dir, tool_paths=NATIVE_TOOLS)

    assert record["success"] is True
    assert record["visual_approval"] is False
    assert record["format"] == "pdf"
    assert record["exit_code"] == 0
    assert record["configured_density"] == 144
    assert record["error"] is None

    # Verify dimensions: 2x3 in at 144 dpi = 288x432
    assert record["actual_pixel_shape"] == {"width": 288, "height": 432}
    assert record["output_dimensions"] == {"width": 288, "height": 432}

    output_png = Path(record["output_png_path"])
    assert output_png.is_file()
    assert _compute_sha256(output_png) == record["output_png_sha256"]

    with Image.open(output_png) as img:
        assert img.format == "PNG"
        assert img.size == (288, 432)

    # Must be JSON-serializable
    serialized = json.dumps(record)
    loaded = json.loads(serialized)
    assert loaded["success"] is True
    assert loaded["visual_approval"] is False


@pytest.mark.skipif(
    not NATIVE_TOOLS,
    reason="Set P2_VECTOR_TOOLS JSON to opt in to native render checks",
)
def test_positive_compact_matplotlib_svg(tmp_path: Path) -> None:
    """Render real compact Matplotlib SVG fixture to 144 DPI PNG."""
    svg_path = tmp_path / "figure.svg"
    sha = _create_sample_svg(svg_path)
    out_dir = tmp_path / "svg_raster_dir"

    record = render_vector(svg_path, sha, out_dir, tool_paths=NATIVE_TOOLS)

    assert record["success"] is True
    assert record["visual_approval"] is False
    assert record["format"] == "svg"
    assert record["exit_code"] == 0
    assert record["configured_density"] == 144
    assert record["error"] is None

    # Matplotlib 2x3 in SVG uses 144pt x 216pt => 288x432 px at 144 dpi
    assert record["actual_pixel_shape"] == {"width": 288, "height": 432}
    assert record["output_dimensions"] == {"width": 288, "height": 432}
    assert record["original_viewport"]["width"] == 144.0
    assert record["original_viewport"]["width_unit"] == "pt"
    assert record["original_viewport"]["height"] == 216.0
    assert record["original_viewport"]["height_unit"] == "pt"

    output_png = Path(record["output_png_path"])
    assert output_png.is_file()
    assert _compute_sha256(output_png) == record["output_png_sha256"]

    with Image.open(output_png) as img:
        assert img.format == "PNG"
        assert img.size == (288, 432)

    assert "sharp_versions" in record["renderer_versions"]
    assert json.dumps(record) is not None


@pytest.mark.skipif(
    not NATIVE_TOOLS,
    reason="Set P2_VECTOR_TOOLS JSON to opt in to native render checks",
)
def test_positive_known_72pt_x_144pt_svg_physical_aspect(tmp_path: Path) -> None:
    """Known 72pt x 144pt SVG physical-aspect proof: outputs 144x288 px."""
    svg_path = tmp_path / "known_aspect.svg"
    svg_path.write_text(
        '<svg width="72pt" height="144pt" viewBox="0 0 72 144" '
        'xmlns="http://www.w3.org/2000/svg"><rect width="72" height="144" fill="blue"/></svg>',
        encoding="utf-8",
    )
    sha = _compute_sha256(svg_path)
    out_dir = tmp_path / "svg_known_aspect_dir"

    record = render_vector(svg_path, sha, out_dir, tool_paths=NATIVE_TOOLS)

    assert record["success"] is True
    assert record["visual_approval"] is False
    assert record["actual_pixel_shape"] == {"width": 144, "height": 288}
    assert record["output_dimensions"] == {"width": 144, "height": 288}

    # Verify physical aspect ratio 72/144 = 0.5 preserved
    assert (
        record["actual_pixel_shape"]["width"] / record["actual_pixel_shape"]["height"]
        == 0.5
    )

    output_png = Path(record["output_png_path"])
    with Image.open(output_png) as img:
        assert img.size == (144, 288)


@pytest.mark.skipif(
    not NATIVE_TOOLS,
    reason="Set P2_VECTOR_TOOLS JSON to opt in to native render checks",
)
def test_positive_svg_css_px_units(tmp_path: Path) -> None:
    """SVG with CSS px units scaled at 1.5x (144 / 96)."""
    svg_path = tmp_path / "px_units.svg"
    svg_path.write_text(
        '<svg width="100px" height="200px" viewBox="0 0 100 200" '
        'xmlns="http://www.w3.org/2000/svg"><rect width="100" height="200" fill="green"/></svg>',
        encoding="utf-8",
    )
    sha = _compute_sha256(svg_path)
    out_dir = tmp_path / "svg_px_dir"

    record = render_vector(svg_path, sha, out_dir, tool_paths=NATIVE_TOOLS)

    assert record["success"] is True
    assert record["visual_approval"] is False
    assert record["actual_pixel_shape"] == {"width": 150, "height": 300}
    assert record["output_dimensions"] == {"width": 150, "height": 300}


@pytest.mark.skipif(
    not NATIVE_TOOLS,
    reason="Set P2_VECTOR_TOOLS JSON to opt in to native render checks",
)
def test_visual_approval_always_false(tmp_path: Path) -> None:
    """visual_approval must remain strictly False across all calls."""
    pdf_path = tmp_path / "dummy.pdf"
    sha = _create_sample_pdf(pdf_path)
    out_dir = tmp_path / "out_va"

    record = render_vector(pdf_path, sha, out_dir, tool_paths=NATIVE_TOOLS)
    assert record["visual_approval"] is False
    assert "visual_approval is false" in record["claim"].lower()


def test_explicit_tool_configuration_required(tmp_path: Path) -> None:
    pdf = tmp_path / "input.pdf"
    digest = _create_sample_pdf(pdf)
    with pytest.raises(ValueError, match="explicit tool_paths"):
        render_vector(pdf, digest, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_mutation_preserves_subprocess_evidence(tmp_path: Path) -> None:
    pdf = tmp_path / "input.pdf"
    digest = _create_sample_pdf(pdf)
    stub = tmp_path / "mutate.sh"
    _make_executable(
        stub,
        '#!/bin/sh\necho captured-stdout\necho captured-stderr >&2\necho changed >> "$7"\nexit 7\n',
    )
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        render_vector(pdf, digest, out, tool_paths={"pdftoppm": stub})
    receipt = json.loads((out / "raster_diagnostics.json").read_text())
    assert receipt["success"] is False
    assert receipt["exit_code"] == 7
    assert receipt["stdout"].strip() == "captured-stdout"
    assert receipt["stderr"].strip() == "captured-stderr"
    assert receipt["input_sha256"] == digest
    assert receipt["post_input_sha256"] != digest
    assert receipt["command"][0] == str(stub)


def test_missing_tool_rejected_without_output_creation(tmp_path: Path) -> None:
    pdf = tmp_path / "input.pdf"
    digest = _create_sample_pdf(pdf)
    with pytest.raises(ValueError, match="tool"):
        render_vector(
            pdf, digest, tmp_path / "out", tool_paths={"pdftoppm": tmp_path / "absent"}
        )
    assert not (tmp_path / "out").exists()


def test_decoded_png_identity_cannot_change_before_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager
    from collections.abc import Iterator
    from typing import Any
    from scripts.research.shot_pattern_analysis import vector_rasterization as module

    pdf = tmp_path / "input.pdf"
    digest = _create_sample_pdf(pdf)
    template = tmp_path / "template.png"
    Image.new("RGB", (16, 8), "white").save(template)
    stub = tmp_path / "copy.sh"
    _make_executable(
        stub,
        f'#!/bin/sh\nif [ "$1" = "-v" ]; then exit 0; fi\ncp "{template}" "$8.png"\n',
    )
    original_open = Image.open

    @contextmanager
    def mutate_after_decode(
        path: Any, *args: Any, **kwargs: Any
    ) -> Iterator[Image.Image]:
        with original_open(path, *args, **kwargs) as im:
            yield im
        Image.new("RGB", (1, 1), "black").save(path)

    monkeypatch.setattr(module.Image, "open", mutate_after_decode)
    result = render_vector(pdf, digest, tmp_path / "out", tool_paths={"pdftoppm": stub})
    assert result["success"] is False
    assert "changed" in result["error"].lower()


def test_native_launch_error_retains_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.research.shot_pattern_analysis import vector_rasterization as module

    pdf = tmp_path / "input.pdf"
    digest = _create_sample_pdf(pdf)
    stub = tmp_path / "exists.sh"
    _make_executable(stub, "#!/bin/sh\nexit 0\n")

    def fail(*args: object, **kwargs: object) -> object:
        raise OSError("controlled launch failure")

    monkeypatch.setattr(module.subprocess, "run", fail)
    result = render_vector(pdf, digest, tmp_path / "out", tool_paths={"pdftoppm": stub})
    assert result["success"] is False
    assert result["input_sha256"] == result["post_input_sha256"] == digest
    assert "controlled launch failure" in result["stderr"]
    assert json.loads((tmp_path / "out/raster_diagnostics.json").read_text()) == result
