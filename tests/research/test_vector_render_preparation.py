"""Native vector command preparation must preserve explicit tools and dimensions."""

from pathlib import Path

import pytest

try:
    import pypdf  # noqa: F401
except (ImportError, OSError) as exc:
    pytest.skip(
        f"Vector review dependencies unavailable: {exc}", allow_module_level=True
    )

from scripts.research.shot_pattern_analysis import vector_rasterization as module

pytestmark = pytest.mark.unit


def test_pdf_command_preparation_preserves_density_and_does_not_launch(tmp_path):
    tool = tmp_path / "pdftoppm"
    tool.write_text("#!/bin/sh\nexit 0\n")
    tool.chmod(0o755)
    source = tmp_path / "source.pdf"
    destination = tmp_path / "out"
    cmd, tools, viewport, dimensions = module._prepare_render_command(
        source,
        destination,
        {"format": "pdf", "dimensions": {"media_box": [0, 0, 72, 144]}},
        {"pdftoppm": tool},
    )
    assert cmd == [
        str(tool),
        "-f",
        "1",
        "-singlefile",
        "-r",
        "144",
        "-png",
        str(source),
        str(destination / "source"),
    ]
    assert tools == {"pdftoppm": str(tool)}
    assert viewport == [0, 0, 72, 144]
    assert dimensions is None
    assert not destination.exists()


def test_preparation_requires_explicit_tool_paths():
    with pytest.raises(ValueError, match="explicit tool_paths"):
        module._prepare_render_command(
            Path("x.pdf"), Path("out"), {"format": "pdf"}, None
        )
