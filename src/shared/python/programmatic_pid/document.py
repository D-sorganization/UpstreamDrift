"""PIDDocument — the primary public API for programmatic-pid.

Provides a clean, object-oriented interface for external programs and agents
to create, inspect, and export P&ID drawings.

Usage::

    from programmatic_pid import PIDDocument

    doc = PIDDocument.from_yaml("spec.yml", profile="compact")
    doc.export_dxf(Path("output.dxf"))
    doc.export_svg(Path("output.svg"))
    doc.export_pdf(Path("output.pdf"))  # if ezdxf[draw] supports it

    # Spatial queries (agent-friendly)
    bbox = doc.equipment_bbox("V-101")
    free = doc.find_free_region(20, 15)
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from programmatic_pid.equipment import equipment_center, equipment_dims
from programmatic_pid.geometry import find_free_region, to_float
from programmatic_pid.profiles import apply_profile
from programmatic_pid.spec_loader import SpecAccessor, load_spec
from programmatic_pid.types import BBox, Point, SpecDict
from programmatic_pid.validation import (
    collect_issues,
    validate_spec,
    validate_spec_json,
)

logger = logging.getLogger(__name__)


class PIDDocument:
    """High-level facade for generating P&ID drawings from a spec.

    Invariant: self._spec is always validated before any drawing operation.

    Precondition for construction:
        - spec_data is a dict (typically loaded from YAML).
        - If profile is given, it must be a valid profile name.

    Postconditions:
        - The spec has passed validation.
        - Layout regions have been computed.
    """

    def __init__(self, spec_data: SpecDict, profile: str | None = None) -> None:
        if spec_data is None:
            raise ValueError("spec_data must be provided")
        self._raw = deepcopy(spec_data)
        self._spec = apply_profile(spec_data, profile)
        validate_spec(self._spec)
        self._accessor = SpecAccessor(self._spec)
        self._equipment_by_id: dict[str, dict[str, Any]] = {
            eq["id"]: eq for eq in self._accessor.equipment if eq.get("id")
        }

    @classmethod
    def from_yaml(cls, path: str | Path, profile: str | None = None) -> PIDDocument:
        """Load from a YAML file path."""
        if path is None:
            raise ValueError("path must be provided")
        spec = load_spec(path)
        return cls(spec, profile=profile)

    @classmethod
    def from_partial(
        cls, spec_data: SpecDict, profile: str | None = None
    ) -> PIDDocument | None:
        """Try to build from a potentially incomplete spec.

        Returns None if the spec has fatal errors; otherwise returns
        a PIDDocument (with warnings logged).
        """
        if spec_data is None:
            raise ValueError("spec_data must be provided")
        issues = collect_issues(spec_data)
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            for e in errors:
                logger.warning("Validation error at %s: %s", e.path, e.message)
            return None
        return cls(spec_data, profile=profile)

    # ----- Properties -----

    @property
    def spec(self) -> SpecDict:
        """The validated, profile-applied spec."""
        return self._spec

    @property
    def accessor(self) -> SpecAccessor:
        return self._accessor

    @property
    def equipment_ids(self) -> list[str]:
        return list(self._equipment_by_id.keys())

    @property
    def instrument_ids(self) -> list[str]:
        return [
            ins.get("id", "") for ins in self._accessor.instruments if ins.get("id")
        ]

    @property
    def stream_ids(self) -> list[str]:
        return [s.get("id", "") for s in self._accessor.streams if s.get("id")]

    # ----- Spatial Queries (Agent-Friendly) -----

    def equipment_bbox(self, eq_id: str) -> BBox | None:
        """Return the bounding box of a specific equipment item.

        Returns None if the equipment ID is not found.
        """
        if eq_id is None:
            raise ValueError("eq_id must be provided")
        eq = self._equipment_by_id.get(eq_id)
        if eq is None:
            return None
        x = to_float(eq.get("x", 0.0))
        y = to_float(eq.get("y", 0.0))
        w, h = equipment_dims(eq)
        return BBox(x, y, x + w, y + h)

    def equipment_position(self, eq_id: str) -> Point | None:
        """Return the center point of a specific equipment item."""
        if eq_id is None:
            raise ValueError("eq_id must be provided")
        eq = self._equipment_by_id.get(eq_id)
        if eq is None:
            return None
        cx, cy = equipment_center(eq)
        return Point(cx, cy)

    def process_bbox(self) -> BBox:
        """Return the bounding box enclosing all equipment."""
        equipment = self._accessor.equipment
        if not equipment:
            return BBox(0.0, 0.0, 240.0, 160.0)
        x_min = min(to_float(eq.get("x", 0.0)) for eq in equipment)
        y_min = min(to_float(eq.get("y", 0.0)) for eq in equipment)
        x_max = max(
            to_float(eq.get("x", 0.0)) + equipment_dims(eq)[0] for eq in equipment
        )
        y_max = max(
            to_float(eq.get("y", 0.0)) + equipment_dims(eq)[1] for eq in equipment
        )
        return BBox(x_min, y_min, x_max, y_max)

    def find_free_region(self, width: float, height: float) -> BBox | None:
        """Find an unoccupied region for placing new equipment.

        Searches outward from the process area center.
        """
        if width is None:
            raise ValueError("width must be provided")
        occupied = []
        for eq in self._accessor.equipment:
            bb = self.equipment_bbox(eq.get("id", ""))
            if bb:
                occupied.append(bb)
        origin = self.process_bbox().center
        return find_free_region(occupied, width, height, search_origin=origin)

    # ----- Validation -----

    def validate_json(self) -> list[dict[str, str]]:
        """Return structured validation results as JSON-serializable dicts."""
        return validate_spec_json(self._spec)  # type: ignore[no-any-return]

    # ----- Export -----

    def export_dxf(self, path: Path, sheet_set: str = "two") -> None:
        """Generate and save DXF output.

        Delegates to the existing generate() function for backward compat.
        """
        # Import here to avoid circular imports during transition
        if path is None:
            raise ValueError("path must be provided")
        from programmatic_pid.generator import (
            generate_controls_sheet,
            generate_process_sheet,
        )

        path = Path(path)
        generate_process_sheet(
            spec_path="",  # not used when prepared_spec is given
            out_path=str(path),
            svg_path=None,
            profile=None,
            prepared_spec=self._spec,
        )
        if sheet_set == "two":
            from programmatic_pid.generator import derive_related_path

            controls_path = derive_related_path(path, "controls")
            generate_controls_sheet(
                spec_path="",
                out_path=str(controls_path),
                svg_path=None,
                profile=None,
                prepared_spec=self._spec,
            )

    def export_svg(self, path: Path) -> None:
        """Generate DXF then convert to SVG."""
        if path is None:
            raise ValueError("path must be provided")
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tmp:
            tmp_dxf = Path(tmp.name)
        try:
            from programmatic_pid.generator import generate_process_sheet

            generate_process_sheet(
                spec_path="",
                out_path=str(tmp_dxf),
                svg_path=str(path),
                profile=None,
                prepared_spec=self._spec,
            )
        finally:
            tmp_dxf.unlink(missing_ok=True)

    def export_pdf(self, path: Path) -> None:
        """Export as PDF via SVG intermediate.

        Requires svglib + reportlab or ezdxf's drawing backend.
        Falls back gracefully with a clear error message.
        """
        if path is None:
            raise ValueError("path must be provided")
        import os
        import tempfile

        _fd, _svg_tmp = tempfile.mkstemp(suffix=".svg")
        os.close(_fd)
        svg_path = Path(_svg_tmp)
        try:
            self.export_svg(svg_path)
            if not svg_path.exists():
                raise RuntimeError("SVG generation failed; cannot produce PDF")

            try:
                from reportlab.graphics import renderPDF
                from svglib.svglib import svg2rlg

                drawing = svg2rlg(str(svg_path))
                if drawing:
                    renderPDF.drawToFile(drawing, str(path))
                    logger.info("PDF exported to %s", path)
                else:
                    raise RuntimeError("svglib could not parse SVG")
            except ImportError as err:
                raise RuntimeError(
                    "PDF export requires 'svglib' and 'reportlab'. "
                    "Install with: pip install svglib reportlab"
                ) from err
        finally:
            svg_path.unlink(missing_ok=True)
