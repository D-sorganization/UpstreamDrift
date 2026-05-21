"""programmatic-pid: Generate P&ID drawings from YAML specifications.

Public API::

    from programmatic_pid import PIDDocument, validate_spec, SpecValidationError

    doc = PIDDocument.from_yaml("spec.yml", profile="compact")
    doc.export_dxf(Path("output.dxf"))

    # Agent-friendly structured validation
    issues = doc.validate_json()

    # Spatial queries
    bbox = doc.equipment_bbox("V-101")
"""

# Primary API
# CLI / orchestration
from programmatic_pid.cli import (
    generate,
    generate_controls_sheet,
    generate_process_sheet,
)
from programmatic_pid.controls import add_control_loops
from programmatic_pid.document import PIDDocument

# Equipment
from programmatic_pid.equipment import (
    EQUIPMENT_RENDERERS,
    equipment_anchor,
    equipment_center,
    equipment_dims,
    register_equipment,
)

# Geometry
from programmatic_pid.geometry import to_float
from programmatic_pid.instruments import add_instrument

# Layout
from programmatic_pid.layout import (
    LabelPlacer,
    compute_layout_regions,
    spread_instrument_positions,
)

# Profiles
from programmatic_pid.profiles import PROFILE_PRESETS, apply_profile

# Rendering
from programmatic_pid.rendering import (
    add_arrow,
    add_box,
    add_text,
    ensure_layer,
    ensure_layers,
)

# Spec loading
from programmatic_pid.spec_loader import SpecAccessor, load_spec, prepare_spec

# Streams, instruments, controls
from programmatic_pid.streams import add_stream
from programmatic_pid.types import BBox, Point, SpecValidationError, ValidationIssue

# Validation
from programmatic_pid.validation import (
    collect_issues,
    validate_spec,
    validate_spec_json,
)

__version__ = "1.0.0"

__all__ = [
    # Primary API
    "PIDDocument",
    # Validation
    "validate_spec",
    "validate_spec_json",
    "collect_issues",
    "SpecValidationError",
    "ValidationIssue",
    # Spec loading
    "load_spec",
    "prepare_spec",
    "SpecAccessor",
    # Profiles
    "apply_profile",
    "PROFILE_PRESETS",
    # Equipment
    "register_equipment",
    "EQUIPMENT_RENDERERS",
    "equipment_dims",
    "equipment_center",
    "equipment_anchor",
    # Geometry
    "to_float",
    "BBox",
    "Point",
    # Layout
    "LabelPlacer",
    "compute_layout_regions",
    "spread_instrument_positions",
    # Rendering
    "ensure_layer",
    "ensure_layers",
    "add_text",
    "add_box",
    "add_arrow",
    # Drawing
    "add_stream",
    "add_instrument",
    "add_control_loops",
    # Orchestration
    "generate",
    "generate_process_sheet",
    "generate_controls_sheet",
]
