"""Heavy integration tests for ezdxf CAD export capability (fixes #1986).

Tests that ezdxf can create, populate, and round-trip DXF documents.
All tests skip gracefully when ezdxf is not installed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def ezdxf():
    """Import ezdxf or skip the module."""
    ezdxf_mod = pytest.importorskip("ezdxf")
    return ezdxf_mod


class TestEzdxfDocumentCreation:
    """Contract: ezdxf can create and manipulate DXF documents."""

    def test_create_new_document(self, ezdxf) -> None:
        """ezdxf.new() returns a valid Drawing object."""
        doc = ezdxf.new("R2010")
        assert doc is not None
        assert doc.dxfversion == "AC1024"

    def test_add_modelspace_entities(self, ezdxf) -> None:
        """Polylines and lines can be added to modelspace."""
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()

        # Add a line
        msp.add_line((0, 0), (10, 10))

        # Add a lwpolyline
        points = [(0, 0), (5, 0), (5, 5), (0, 5), (0, 0)]
        msp.add_lwpolyline(points)

        entities = list(msp)
        assert len(entities) >= 2

    def test_add_circle_and_arc(self, ezdxf) -> None:
        """Circles and arcs can be added to modelspace."""
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()

        msp.add_circle((0, 0), radius=5.0)
        msp.add_arc(center=(0, 0), radius=3.0, start_angle=0, end_angle=90)

        entities = list(msp)
        assert len(entities) == 2

    def test_add_text_entity(self, ezdxf) -> None:
        """Text entities can be added to modelspace."""
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        msp.add_text("Golf Swing Analysis", dxfattribs={"height": 0.5})
        entities = list(msp)
        assert len(entities) == 1


class TestEzdxfRoundtrip:
    """Contract: DXF documents survive a write→read cycle intact."""

    def test_save_and_reload_document(self, ezdxf) -> None:
        """Write a DXF file and read it back with the same entity count."""
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        msp.add_line((0, 0), (100, 0))
        msp.add_line((100, 0), (100, 100))
        msp.add_line((100, 100), (0, 100))
        msp.add_line((0, 100), (0, 0))

        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = Path(tmpdir) / "test_output.dxf"
            doc.saveas(str(dxf_path))

            assert dxf_path.exists(), "DXF file was not created"
            assert dxf_path.stat().st_size > 0, "DXF file is empty"

            # Reload and verify
            loaded_doc = ezdxf.readfile(str(dxf_path))
            loaded_msp = loaded_doc.modelspace()
            loaded_entities = list(loaded_msp)
            assert len(loaded_entities) == 4, (
                f"Expected 4 lines after reload, got {len(loaded_entities)}"
            )

    def test_layers_survive_roundtrip(self, ezdxf) -> None:
        """Layer assignments survive a write→read cycle."""
        doc = ezdxf.new("R2010")
        doc.layers.add("TRAJECTORY", color=2)  # yellow

        msp = doc.modelspace()
        msp.add_line((0, 0), (1, 0), dxfattribs={"layer": "TRAJECTORY"})

        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = Path(tmpdir) / "layers.dxf"
            doc.saveas(str(dxf_path))

            loaded = ezdxf.readfile(str(dxf_path))
            assert "TRAJECTORY" in [layer.dxf.name for layer in loaded.layers]
            loaded_msp = loaded.modelspace()
            line = next(iter(loaded_msp))
            assert line.dxf.layer == "TRAJECTORY"


pytestmark = pytest.mark.live_simulation
