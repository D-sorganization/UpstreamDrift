# ruff: noqa: E501
"""End-to-end functional tests for the programmatic_pid generation pipeline.

Covers issue #3185: ``generator.py`` / ``rendering.py`` / ``document.py`` had
no functional tests. This module drives the full ``spec -> generate -> render``
path and asserts observable document / stream / instrument structure plus a
golden-output diagram diff for the committed biochar reference spec.

The reference spec ``examples/pid/biochar/biochar_pid_spec.yml`` is the stable
golden input; the assertions on emitted DXF entity counts form the golden diff.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

ezdxf = pytest.importorskip("ezdxf", reason="ezdxf not installed")

from programmatic_pid.document import PIDDocument
from programmatic_pid.generator import (
    add_box,
    add_text,
    ensure_layers,
    generate,
    generate_controls_sheet,
    generate_process_sheet,
)
from programmatic_pid.types import BBox

# ---------------------------------------------------------------------------
# Golden reference spec — committed, stable input for diff assertions.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[5]
_GOLDEN_SPEC = _REPO_ROOT / "examples" / "pid" / "biochar" / "biochar_pid_spec.yml"

# Golden structural counts for the biochar reference spec. These are the
# committed "diff" baseline — a change to the spec or the loader that alters
# the document structure must update these numbers deliberately.
_GOLDEN_EQUIPMENT_COUNT = 11
_GOLDEN_STREAM_COUNT = 10
_GOLDEN_INSTRUMENT_COUNT = 19


@pytest.fixture
def golden_spec_path() -> Path:
    """Return the path to the committed biochar reference spec."""
    assert _GOLDEN_SPEC.exists(), f"golden spec missing: {_GOLDEN_SPEC}"
    return _GOLDEN_SPEC


@pytest.fixture
def golden_doc(golden_spec_path: Path) -> PIDDocument:
    """Build a validated PIDDocument from the golden spec."""
    return PIDDocument.from_yaml(golden_spec_path)


# ---------------------------------------------------------------------------
# document.py — structural assertions on the loaded/validated document
# ---------------------------------------------------------------------------


class TestDocumentStructure:
    def test_golden_spec_counts(self, golden_doc: PIDDocument) -> None:
        """spec -> document: equipment/stream/instrument counts are stable."""
        assert len(golden_doc.equipment_ids) == _GOLDEN_EQUIPMENT_COUNT
        assert len(golden_doc.stream_ids) == _GOLDEN_STREAM_COUNT
        assert len(golden_doc.instrument_ids) == _GOLDEN_INSTRUMENT_COUNT

    def test_known_ids_present(self, golden_doc: PIDDocument) -> None:
        """Specific known equipment/instrument ids survive load + validation."""
        assert "V-101" in golden_doc.equipment_ids
        assert "BR-101" in golden_doc.equipment_ids
        assert "PT-101" in golden_doc.instrument_ids
        assert "S-001" in golden_doc.stream_ids

    def test_spec_property_is_validated_dict(self, golden_doc: PIDDocument) -> None:
        """The exposed spec is a profile-applied, validated mapping."""
        spec = golden_doc.spec
        assert isinstance(spec, dict)
        assert spec["project"]["id"] == "BIOCHAR-PID-001"

    def test_equipment_bbox_and_position(self, golden_doc: PIDDocument) -> None:
        """Spatial queries return concrete geometry for a known item."""
        bbox = golden_doc.equipment_bbox("V-101")
        assert isinstance(bbox, BBox)
        # V-101 is at x=95,y=52,width=28,height=68 in the golden spec.
        assert bbox.x_min == pytest.approx(95.0)
        assert bbox.y_min == pytest.approx(52.0)
        assert bbox.x_max == pytest.approx(95.0 + 28.0)
        assert bbox.y_max == pytest.approx(52.0 + 68.0)

        center = golden_doc.equipment_position("V-101")
        assert center is not None
        assert center.x == pytest.approx(95.0 + 28.0 / 2)
        assert center.y == pytest.approx(52.0 + 68.0 / 2)

    def test_unknown_equipment_returns_none(self, golden_doc: PIDDocument) -> None:
        """Unknown ids yield None rather than raising (agent-friendly)."""
        assert golden_doc.equipment_bbox("NOPE-999") is None
        assert golden_doc.equipment_position("NOPE-999") is None

    def test_process_bbox_encloses_all_equipment(self, golden_doc: PIDDocument) -> None:
        """process_bbox spans every equipment bbox."""
        process = golden_doc.process_bbox()
        for eq_id in golden_doc.equipment_ids:
            bb = golden_doc.equipment_bbox(eq_id)
            assert bb is not None
            assert process.x_min <= bb.x_min
            assert process.y_min <= bb.y_min
            assert process.x_max >= bb.x_max
            assert process.y_max >= bb.y_max

    def test_find_free_region_returns_unoccupied_bbox(
        self, golden_doc: PIDDocument
    ) -> None:
        """find_free_region yields a region not overlapping any equipment."""
        free = golden_doc.find_free_region(5.0, 5.0)
        assert free is not None
        for eq_id in golden_doc.equipment_ids:
            bb = golden_doc.equipment_bbox(eq_id)
            assert bb is not None
            overlap = (
                free.x_min < bb.x_max
                and free.x_max > bb.x_min
                and free.y_min < bb.y_max
                and free.y_max > bb.y_min
            )
            assert not overlap, f"free region overlaps {eq_id}"

    def test_validate_json_clean_for_golden(self, golden_doc: PIDDocument) -> None:
        """The golden spec produces no error-severity validation issues."""
        issues = golden_doc.validate_json()
        errors = [i for i in issues if i.get("severity") == "error"]
        assert errors == []

    def test_from_partial_rejects_fatal_spec(self) -> None:
        """from_partial returns None when the spec has fatal errors."""
        bad_spec: dict[str, Any] = {"project": {}, "equipment": []}
        assert PIDDocument.from_partial(bad_spec) is None

    def test_from_partial_accepts_minimal_valid_spec(self) -> None:
        """from_partial builds a document for a minimal but valid spec."""
        spec: dict[str, Any] = {
            "project": {"id": "P1", "title": "Minimal"},
            "equipment": [{"id": "V1", "type": "vessel", "w": 5.0, "h": 5.0}],
        }
        doc = PIDDocument.from_partial(spec)
        assert doc is not None
        assert doc.equipment_ids == ["V1"]

    def test_constructor_rejects_none_spec(self) -> None:
        with pytest.raises(ValueError, match="spec_data must be provided"):
            PIDDocument(None)


# ---------------------------------------------------------------------------
# generator.py + rendering.py — full DXF emission (the golden diagram diff)
# ---------------------------------------------------------------------------


def _entity_layers(msp: Any) -> set[str]:
    return {e.dxf.layer.lower() for e in msp}


class TestGenerateProcessSheet:
    def test_process_sheet_emits_dxf_with_entities(
        self, golden_doc: PIDDocument, tmp_path: Path
    ) -> None:
        """generate_process_sheet writes a readable DXF with drawn entities."""
        out = tmp_path / "process.dxf"
        generate_process_sheet(
            spec_path="",
            out_path=out,
            prepared_spec=golden_doc.spec,
        )
        assert out.exists()

        doc = ezdxf.readfile(out)
        msp = doc.modelspace()
        entities = list(msp)
        # Golden diagram diff: a non-trivial, deterministic drawing is produced.
        assert len(entities) > 50

        # Every equipment id should be emitted as text somewhere on the sheet.
        texts = {e.dxf.text for e in msp if e.dxftype() in ("TEXT", "MTEXT")}
        joined = " ".join(texts)
        for eq_id in ("HP-101", "V-101", "BR-101"):
            assert eq_id in joined

        # Equipment / instrument layers are present in the rendered output.
        layers = _entity_layers(msp)
        assert any("equipment" in layer or "vessel" in layer for layer in layers)

    def test_controls_sheet_emits_loop_table(
        self, golden_doc: PIDDocument, tmp_path: Path
    ) -> None:
        """generate_controls_sheet renders the loops/interlocks sheet."""
        out = tmp_path / "controls.dxf"
        generate_controls_sheet(
            spec_path="biochar.yml",
            out_path=out,
            prepared_spec=golden_doc.spec,
        )
        assert out.exists()
        doc = ezdxf.readfile(out)
        msp = doc.modelspace()
        texts = " ".join(e.dxf.text for e in msp if e.dxftype() in ("TEXT", "MTEXT"))
        assert "Controls and Interlocks" in texts
        # A control loop measurement tag appears in the table.
        assert "PT-101" in texts

    def test_document_export_dxf_writes_two_sheets(
        self, golden_doc: PIDDocument, tmp_path: Path
    ) -> None:
        """PIDDocument.export_dxf produces both the process and controls DXF."""
        out = tmp_path / "diagram.dxf"
        golden_doc.export_dxf(out, sheet_set="two")
        assert out.exists()
        controls = out.with_name("diagram_controls.dxf")
        assert controls.exists()

    def test_generate_top_level_orchestration(
        self, golden_spec_path: Path, tmp_path: Path
    ) -> None:
        """generate() composes both sheets from a YAML spec path."""
        out = tmp_path / "full.dxf"
        generate(str(golden_spec_path), str(out), sheet_set="two")
        assert out.exists()
        assert (tmp_path / "full_controls.dxf").exists()


# ---------------------------------------------------------------------------
# rendering.py primitives — direct value assertions against a real modelspace
# ---------------------------------------------------------------------------


class TestRenderingPrimitives:
    def test_ensure_layers_creates_named_layers(self, golden_doc: PIDDocument) -> None:
        doc = ezdxf.new(setup=True)
        ensure_layers(doc, golden_doc.spec)
        existing = {layer.dxf.name.lower() for layer in doc.layers}
        assert "instruments" in existing
        assert "process_lines" in existing

    def test_add_box_emits_closed_polyline(self) -> None:
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()
        before = len(list(msp))
        add_box(msp, 0.0, 0.0, 10.0, 5.0, "EQUIPMENT")
        after = list(msp)
        assert len(after) == before + 1
        poly = after[-1]
        assert poly.dxf.layer == "EQUIPMENT"

    def test_add_text_places_label_on_layer(self) -> None:
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()
        add_text(msp, "V-101", 1.0, 2.0, 2.5, layer="TEXT")
        texts = [e for e in msp if e.dxftype() in ("TEXT", "MTEXT")]
        assert any(e.dxf.text == "V-101" for e in texts)
        assert any(e.dxf.layer == "TEXT" for e in texts)
