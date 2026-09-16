"""Markdown documentation generator for ground-support receipts (HO-2 #10156)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from .receipt_components import AddressReceipt, GroundReceipt, IkReceipt
from .receipt_dynamics import DynamicsReceipt

if TYPE_CHECKING:
    from .receipt_schema import Receipt


def _render_section_table(model_cls: type[BaseModel], default_stage: str) -> str:
    """Render a markdown table for fields in a Pydantic model with aligned columns matching Prettier."""
    rows: list[tuple[str, str, str, str]] = []
    for name, field in model_cls.model_fields.items():
        extra = field.json_schema_extra
        unit_val = (
            extra.get("unit", "dimensionless")
            if isinstance(extra, dict)
            else "dimensionless"
        )
        stage_val = (
            extra.get("stage", default_stage)
            if isinstance(extra, dict)
            else default_stage
        )
        unit = str(unit_val) if unit_val is not None else "dimensionless"
        stage = str(stage_val) if stage_val is not None else default_stage
        meaning = field.description or name
        rows.append((f"`{name}`", unit, meaning, stage))

    headers = ("Field", "Unit", "Meaning", "Stage")
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(4)]
    dividers = [f":{'-' * (w - 1)}" for w in widths]

    lines = [
        f"| {headers[0]:<{widths[0]}} | {headers[1]:<{widths[1]}} | {headers[2]:<{widths[2]}} | {headers[3]:<{widths[3]}} |",
        f"| {dividers[0]} | {dividers[1]} | {dividers[2]} | {dividers[3]} |",
    ]
    for r in rows:
        lines.append(
            f"| {r[0]:<{widths[0]}} | {r[1]:<{widths[1]}} | {r[2]:<{widths[2]}} | {r[3]:<{widths[3]}} |"
        )
    return "\n".join(lines)


def render_receipts_markdown() -> str:
    """Generate the comprehensive RECEIPTS.md documentation string."""
    from .receipt_schema import Receipt

    sections = [
        (
            "# Ground-Support Pipeline Receipts Reference\n\n"
            "This document defines the formal schema, units, meanings, and originating pipeline "
            "stages for all fields stored in `receipt.json` produced by the ground-support pipeline "
            "(`src/shared/python/motion_matching/pipeline/`).\n\n"
            "Downstream tools (launcher tile, feature parity, MJX trajectory optimization) read "
            "these fields by name according to this contract.\n\n"
            "Regenerate with:\n\n"
            "```bash\n"
            "python -m src.shared.python.motion_matching.pipeline.receipt_schema --markdown\n"
            "```"
        ),
        (
            "## Top-Level Metadata\n\n"
            "Summary provenance, hashes, file references, and execution duration written by "
            "`pipeline.receipt.build_ground_support_receipt`.\n\n"
            + _render_section_table(Receipt, "metadata")
        ),
        (
            "## Ground Stage (`ground`)\n\n"
            "Ground contact plane calibration, toe contact sphere placements, and per-sphere "
            "stance detection fractions written by `pipeline.lane.Lane`.\n\n"
            + _render_section_table(GroundReceipt, "ground")
        ),
        (
            "## Address Stage (`address`)\n\n"
            "Address pose calibration, static neutral trial placement, CoM support check, and "
            "optional closure weld optimization written by `pipeline.address`.\n\n"
            + _render_section_table(AddressReceipt, "address")
        ),
        (
            "## Inverse Kinematics Stage (`ik`)\n\n"
            "Full-trajectory marker matching, alternating calibration, limb scaling, and range-of-motion "
            "flags written by `pipeline.reference`.\n\n"
            + _render_section_table(IkReceipt, "ik")
        ),
        (
            "## Dynamics Stage (`dynamics`)\n\n"
            "Computed-torque tracking simulation, zero-moment point diagnostics, contact parameters, "
            "and optional optimization filters written by `pipeline.dynamics`.\n\n"
            + _render_section_table(DynamicsReceipt, "dynamics")
        ),
    ]
    return "\n\n".join(sections) + "\n"
