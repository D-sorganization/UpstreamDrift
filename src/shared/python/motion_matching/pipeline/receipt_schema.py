"""Pydantic schema, validator, and documentation generator for ground-support receipts (HO-2 #10156)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .receipt_components import (
    AcceptanceGateReport,
    AcceptanceReceipt,
    AddressReceipt,
    AttachmentOffset,
    CalibratedAddressReport,
    CalibrationReport,
    CentreOfMassReport,
    ClavicleLinkDeg,
    ClosureFitReport,
    ClubReceipt,
    ConstrainedIkReceipt,
    GroundReceipt,
    HipCalibrationReceipt,
    IkReceipt,
    PostureSummary,
    ReferenceReport,
    RomFlag,
    SeedOffsetsReport,
    SegmentScalingEntry,
    SegmentScalingReport,
    SpineBendDeg,
    StaticTrialReport,
    ToeSphereReceipt,
)
from .receipt_docs import render_receipts_markdown
from .receipt_dynamics import (
    BackswingReceipt,
    ContactParametersReceipt,
    ControllerReceipt,
    DynamicsReceipt,
    ReferenceZmpReceipt,
    ShootingFitReport,
    WeightFractionReceipt,
    ZmpFilterPass,
    ZmpFilterReport,
)

__all__ = [
    "AddressReceipt",
    "AttachmentOffset",
    "BackswingReceipt",
    "CalibratedAddressReport",
    "CandidateReplayReceipt",
    "CalibrationReport",
    "CentreOfMassReport",
    "ClavicleLinkDeg",
    "ClosureFitReport",
    "ClubReceipt",
    "ConstrainedIkReceipt",
    "ContactParametersReceipt",
    "ControllerReceipt",
    "DynamicsReceipt",
    "GroundReceipt",
    "HipCalibrationReceipt",
    "IkReceipt",
    "PostureSummary",
    "Receipt",
    "ReferenceReport",
    "ReferenceZmpReceipt",
    "RomFlag",
    "SeedOffsetsReport",
    "SegmentScalingEntry",
    "SegmentScalingReport",
    "ShootingFitReport",
    "SpineBendDeg",
    "StaticTrialReport",
    "ToeSphereReceipt",
    "WeightFractionReceipt",
    "ZmpFilterPass",
    "ZmpFilterReport",
    "render_receipts_markdown",
    "validate_receipt",
]


# -----------------------------------------------------------------------------
# Top-Level Receipt Model
# -----------------------------------------------------------------------------


class CandidateReplayReceipt(BaseModel):
    """Same-input native replay evidence, distinct from an IK/fit receipt (#10336)."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = "matched-swing-replay/1"
    engine: str
    engine_version: str
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attachments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration: dict[str, Any]
    parity: dict[str, Any]
    integration: dict[str, Any]
    shared_metrics: dict[str, float]
    g1_metrics: dict[str, float]
    acceptance: AcceptanceReceipt
    artifacts: dict[str, Any]
    elapsed_s: float
    qualification: str


class Receipt(BaseModel):
    """Complete standardized execution receipt for the full-body ground-support pipeline."""

    model_config = ConfigDict(extra="ignore")

    backend: str = Field(
        "mujoco",
        description="Kinematic backend engine used for tracking (mujoco or pink)",
        json_schema_extra={"unit": "string", "stage": "metadata"},
    )
    engine: str = Field(
        "mujoco",
        description="Full-body dynamics and plant engine (mujoco, drake, pinocchio)",
        json_schema_extra={"unit": "string", "stage": "metadata"},
    )
    ik_backend: str | None = Field(
        None,
        description="Marker inverse kinematics solver backend (scipy, mujoco-minimize)",
        json_schema_extra={"unit": "string", "stage": "metadata"},
    )
    tracking_backend: str | None = Field(
        None,
        description="Forward dynamics tracking solver backend (computed-torque, mj-inverse)",
        json_schema_extra={"unit": "string", "stage": "metadata"},
    )
    base_spec_sha256: str = Field(
        ...,
        description="SHA256 hash of the initial input model specification document",
        json_schema_extra={"unit": "hash", "stage": "metadata"},
    )
    base_spec_file: str | None = Field(
        None,
        description="Filename of the input model specification document",
        json_schema_extra={"unit": "filename", "stage": "metadata"},
    )
    spec_file: str = Field(
        ...,
        description="Filename of the final scaled and calibrated spec document",
        json_schema_extra={"unit": "filename", "stage": "metadata"},
    )
    hipcal_spec_file: str = Field(
        ...,
        description="Filename of the intermediate hip-calibrated spec document",
        json_schema_extra={"unit": "filename", "stage": "metadata"},
    )
    recalibrate_upper: bool = Field(
        ...,
        description="Whether upper-body marker placements were recalibrated",
        json_schema_extra={"unit": "bool", "stage": "metadata"},
    )
    anthropometric: list[float] | None = Field(
        None,
        description="Subject stature (m) and mass (kg) if anthropometric geometry",
        json_schema_extra={"unit": "m, kg", "stage": "metadata"},
    )
    de_leva_table_sha256: str | None = Field(
        None,
        description="SHA256 hash of the de Leva anthropometric table if anthropometric geometry",
        json_schema_extra={"unit": "hash", "stage": "metadata"},
    )
    posture_top_of_backswing: PostureSummary | None = Field(
        None,
        description="Upper body posture metrics at top of backswing (~0.83 s)",
        json_schema_extra={"unit": "compound", "stage": "metadata"},
    )
    spec_sha256: str = Field(
        ...,
        description="SHA256 hash of the final scaled model specification",
        json_schema_extra={"unit": "hash", "stage": "metadata"},
    )
    hip_calibration: HipCalibrationReceipt | None = Field(
        None,
        description="Functional hip calibration metrics and alignment",
        json_schema_extra={"unit": "compound", "stage": "metadata"},
    )
    candidate_sha256: str = Field(
        ...,
        description="SHA256 hash of candidate artifact byte stream",
        json_schema_extra={"unit": "hash", "stage": "metadata"},
    )
    capture: str | None = Field(
        None,
        description="Optical motion capture trial name (driver or iron)",
        json_schema_extra={"unit": "text", "stage": "metadata"},
    )
    capture_sha256: str = Field(
        ...,
        description="SHA256 hash of raw C3D optical motion capture file",
        json_schema_extra={"unit": "hash", "stage": "metadata"},
    )
    club: ClubReceipt | None = Field(
        None,
        description="Club inertial properties, dimensions, and grip offset",
        json_schema_extra={"unit": "compound", "stage": "metadata"},
    )
    grip_rotation_deg: dict[str, list[float]] | None = Field(
        None,
        description="Fitted constant grip roll rotations for lead and trail hands",
        json_schema_extra={"unit": "deg", "stage": "metadata"},
    )
    wrists_bounded: bool | None = Field(
        None,
        description="Whether wrist and forearm joints were bounded to human ranges",
        json_schema_extra={"unit": "bool", "stage": "metadata"},
    )
    labels: list[str] = Field(
        ...,
        description="List of tracked optical marker labels in kinematic model",
        json_schema_extra={"unit": "names", "stage": "metadata"},
    )
    ground: GroundReceipt = Field(
        ...,
        description="Ground height, toe spheres, and stance detection results",
        json_schema_extra={"unit": "compound", "stage": "ground"},
    )
    address: AddressReceipt = Field(
        ...,
        description="Address pose fit, static trial, and initial stance posture",
        json_schema_extra={"unit": "compound", "stage": "address"},
    )
    ik: IkReceipt = Field(
        ...,
        description="Full-trajectory kinematic matching and marker calibration",
        json_schema_extra={"unit": "compound", "stage": "ik"},
    )
    dynamics: DynamicsReceipt = Field(
        ...,
        description="Forward dynamics tracking, computed torque, and ZMP diagnostics",
        json_schema_extra={"unit": "compound", "stage": "dynamics"},
    )
    elapsed_s: float = Field(
        ...,
        description="Total wall-clock runtime of ground-support execution",
        json_schema_extra={"unit": "s", "stage": "metadata"},
    )
    qualification: str = Field(
        ...,
        description="Qualification note and status claim for the run",
        json_schema_extra={"unit": "text", "stage": "metadata"},
    )
    acceptance: AcceptanceReceipt | None = Field(
        None,
        description="Physical and kinematic acceptance evaluation verdict (MS-01)",
        json_schema_extra={"unit": "compound", "stage": "metadata"},
    )


# -----------------------------------------------------------------------------
# Validation Function (DbC)
# -----------------------------------------------------------------------------


def _is_legacy_without_zmp(document: dict[str, Any]) -> bool:
    """Determine if document is an early exploratory run prior to reference_zmp."""
    if not isinstance(document, dict):
        return False
    # Section 13 closure fit run
    if document.get("address", {}).get("closure_fit") is not None:
        return True
    # Early pre-club runs from AN-1 / GS-5
    return not bool(document.get("club"))


def validate_receipt(document: dict[str, Any]) -> Receipt:
    """Validate a ground-support execution receipt dictionary against schema contract.

    Args:
        document: Parsed JSON receipt dictionary from ground-support run.

    Returns:
        Validated `Receipt` Pydantic instance.

    Raises:
        ValueError: If validation fails, with the exact dot-separated field path
            and description of the contract violation.
    """
    if not isinstance(document, dict):
        raise ValueError(
            f"Expected dictionary for receipt document, got {type(document).__name__}"
        )

    # Required sections check
    for req in ("ground", "address", "ik", "dynamics"):
        if req not in document:
            raise ValueError(f"{req}: Field required")

    # Special handling for reference_zmp in legacy runs vs standard runs
    if not _is_legacy_without_zmp(document):
        dynamics = document.get("dynamics", {})
        if not isinstance(dynamics, dict) or "reference_zmp" not in dynamics:
            raise ValueError("dynamics.reference_zmp: Field required")

    try:
        return Receipt.model_validate(document)
    except ValidationError as err:
        errors = err.errors()
        if errors:
            first_err = errors[0]
            loc_parts = [str(p) for p in first_err.get("loc", ())]
            loc_path = ".".join(loc_parts) if loc_parts else "root"
            msg = first_err.get("msg", "Validation error")
            raise ValueError(f"{loc_path}: {msg}") from err
        raise ValueError(str(err)) from err


def main() -> None:
    """CLI entry point for receipt schema inspection, validation, and documentation rendering."""
    parser = argparse.ArgumentParser(
        description="Ground-support receipt schema and documentation."
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Generate Markdown documentation for docs/development/full_body_models/RECEIPTS.md",
    )
    parser.add_argument(
        "receipts",
        nargs="*",
        type=Path,
        help="Receipt JSON file paths to validate",
    )
    args = parser.parse_args()

    if args.markdown:
        out_path = Path("docs/development/full_body_models/RECEIPTS.md")
        content = render_receipts_markdown()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        sys.stdout.write(f"Wrote {len(content)} bytes to {out_path}\n")
    elif args.receipts:
        for receipt_path in args.receipts:
            doc = json.loads(receipt_path.read_text(encoding="utf-8"))
            validate_receipt(doc)
            sys.stdout.write(f"Valid: {receipt_path}\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
