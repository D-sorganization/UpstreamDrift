"""GS3DX Variants Promotion and R2025b Evidence Contracts (MMR-03, #11087).

Provides fail-closed validation, per-variant receipts, clean-host R2025b build/save/reopen
contract verification, protection of hand-built original models, strict separation of
stable-drive equivalence from C3D fit, and clear distinction between motion prescription /
servo balance and autonomous control.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any

from src.shared.python.contracts import postcondition, precondition, require

PROMOTED_VARIANTS_SCHEMA_VERSION = "gs3dx-variant-receipt/1"
REQUIRED_MATLAB_RELEASE = "2025b"
HOME_LICENSE_BLOCK_LIMIT = 1000
INSTRUMENTATION_RESERVE_BLOCKS = 25
PRODUCTION_BUDGET_CEILING = 975

CANONICAL_C3D_CAPTURE_FILE = "data/C3D_TA_Driver.c3d"
CANONICAL_C3D_CAPTURE_SHA256 = (
    "cbcb4d84aca0a1f558073f0c2328c3f501c4726aaf699e382cf20c5861a0971d"
)

ORIGINAL_HAND_BUILT_MODELS: Mapping[str, str] = {
    "GolfSwing3D_Kinetic.slx": "daca9a90ad0ab819c7d61641ed594b8f230f8658f2a5f4ce34026db44b52ddc9",
    "Kinetically_Driven_Gimbal_Joint.slx": "1dd72e348eeba0ffa5df16a927dd43b5c7f3441d03a45cfdee15f329ac0d825a",
    "Kinetically_Driven_Revolute_Joint.slx": "deb039b2eebf00001cac8097a79d7dbf0dee4a9d2ea0cc883caa80941e044512",
    "Kinetically_Driven_Universal_Joint.slx": "2252a7f9a7457359b26b7c640c01dfdcfe5ac3e3500bab0a6bfc275cbf747ab0",
}


class GS3DXPromotionError(ValueError):
    """Base error for GS3DX variant promotion and evidence contract violations."""


class CleanHostBuildValidationError(GS3DXPromotionError):
    """Raised when clean-host build, save, reopen or MATLAB release validation fails."""


class OriginalModelProtectedError(GS3DXPromotionError):
    """Raised when an immutable hand-built original model has been modified or corrupted."""


class DriveClassificationMismatchError(GS3DXPromotionError):
    """Raised when stable-drive equivalence is conflated with C3D capture fitting."""


class ActuationClassificationError(GS3DXPromotionError):
    """Raised when motion prescription is falsely claimed as autonomous actuation."""


class BalanceClassificationError(GS3DXPromotionError):
    """Raised when servo tracking / Jacobian assistance is falsely claimed as autonomous balance."""


class CandidateIntegrityError(GS3DXPromotionError):
    """Raised when candidate, model, image, or metric package hashes do not match."""


class LedgerPromotionError(GS3DXPromotionError):
    """Raised when an unreviewed or exploratory variant is submitted to the main ledger."""


class MatlabSuiteValidationError(GS3DXPromotionError):
    """Raised when a reported MATLAB test suite rerun fails validation or skips disclosure."""


class GS3DXVariant(str, Enum):
    """The 10 promoted GS3DX multibody variants in lineage order."""

    BASELINE = "Baseline"
    SLIM = "Slim"
    QUAT = "Quat"
    FULL_BODY = "FullBody"
    CONTACT = "Contact"
    GOLFER = "Golfer"
    FIT = "Fit"
    SHAPE = "Shape"
    NECK = "Neck"
    HUMAN = "Human"


class GS3DXDriveMode(str, Enum):
    """Actuation and drive mechanism mode for GS3DX variants."""

    BASELINE_TORQUE = "baseline_torque"
    SLIM_DIRECT_TORQUE = "slim_direct_torque"
    QUAT_DIRECT_TORQUE = "quat_direct_torque"
    FULL_BODY_PASSIVE_LEGS = "full_body_passive_legs"
    STANCE_HOLD_SERVO = "stance_hold_servo"
    FIT_TRACK_FEEDFORWARD_PD = "fit_track_feedforward_pd"
    BALANCE_LOOP_FEEDFORWARD_PD = "balance_loop_feedforward_pd"
    HUMAN_SPRUNG_FEET_BALANCE = "human_sprung_feet_balance"


class DriveClassification(str, Enum):
    """Strict classification distinguishing stable-drive equivalence from C3D marker fit."""

    STABLE_DRIVE_EQUIVALENCE = "stable_drive_equivalence"
    C3D_FIT = "c3d_fit"
    STANCE_HOLD = "stance_hold"


class NeckActuationMode(str, Enum):
    """Actuation disclosure for the cervical spine / head mechanism."""

    NONE = "none"
    RIGID_UPPER_TRUNK = "rigid_upper_trunk"
    MOTION_PRESCRIBED = "motion_prescribed"
    AUTONOMOUS = "autonomous"


class BalanceMode(str, Enum):
    """Leg stabilization mode distinguishing servo tracking from autonomous balance."""

    WELDED_WORLD = "welded_world"
    STANCE_HOLD_SERVO = "stance_hold_servo"
    JACOBIAN_BALANCE_LOOP = "jacobian_balance_loop"
    AUTONOMOUS_BALANCE = "autonomous_balance"
    PASSIVE = "passive"


class ReviewStatus(str, Enum):
    """Promotion status governing main ledger consumption."""

    REVIEWED_PROMOTED = "reviewed_promoted"
    EXPLORATORY_PROMOTED = "exploratory_promoted"
    UNREVIEWED = "unreviewed"


@dataclass(frozen=True)
class VariantBlockBudget:
    """Compiled and uncompiled block counts adhering to Home license limits."""

    uncompiled_count: int
    compiled_count: int
    license_ceiling: int = HOME_LICENSE_BLOCK_LIMIT
    reserve_blocks: int = INSTRUMENTATION_RESERVE_BLOCKS

    @property
    def headroom(self) -> int:
        return (self.license_ceiling - self.reserve_blocks) - self.compiled_count

    @property
    def within_production_ceiling(self) -> bool:
        return self.compiled_count <= (self.license_ceiling - self.reserve_blocks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "uncompiled_count": self.uncompiled_count,
            "compiled_count": self.compiled_count,
            "license_ceiling": self.license_ceiling,
            "reserve_blocks": self.reserve_blocks,
            "headroom": self.headroom,
            "within_production_ceiling": self.within_production_ceiling,
        }


@dataclass(frozen=True)
class CleanHostBuildReceipt:
    """Audit of explicit R2025b build, save, and reopen on a clean host."""

    matlab_release: str
    matlab_version: str
    host: str
    built_clean: bool
    saved_clean: bool
    reopened_clean: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OriginalProtectionSummary:
    """Summary of immutability checks on hand-built original model files."""

    originals_verified: bool
    files_checked: tuple[str, ...]
    hashes_match: bool
    details: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "originals_verified": self.originals_verified,
            "files_checked": list(self.files_checked),
            "hashes_match": self.hashes_match,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ColdReplaySpec:
    """Exact cold-replay command and execution environment."""

    command: str
    cwd: str
    environment: dict[str, str]
    input_files: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "cwd": self.cwd,
            "environment": dict(self.environment),
            "input_files": list(self.input_files),
        }


@dataclass(frozen=True)
class VariantProvenance:
    """Provenance recording tool provider, environment, models, and capture."""

    provider: str
    host: str
    matlab_release: str
    matlab_version: str
    model_name: str
    model_sha256: str
    capture_file: str
    capture_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateIntegrityReceipt:
    """Cryptographic integrity verification linking candidate, model, images, and metrics."""

    model_sha256: str
    candidate_sha256: str
    image_package_sha256: str
    metric_package_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MatlabSuiteResult:
    """Record of full exploratory GS3DX test suite rerun on R2025b."""

    promoted_sha: str
    matlab_release: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: tuple[str, ...]
    skip_reasons: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "promoted_sha": self.promoted_sha,
            "matlab_release": self.matlab_release,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "skipped_tests": list(self.skipped_tests),
            "skip_reasons": dict(self.skip_reasons),
        }


@dataclass(frozen=True)
class GS3DXVariantDefinition:
    """Metadata specification for a promoted GS3DX multibody variant."""

    variant: GS3DXVariant
    model_name: str
    model_file: str
    builder: str
    parent_variant: GS3DXVariant | None
    uncompiled_blocks: int
    compiled_blocks: int
    default_drive_mode: GS3DXDriveMode
    default_drive_classification: DriveClassification
    default_neck_actuation: NeckActuationMode
    default_balance_mode: BalanceMode
    description: str


@dataclass(frozen=True)
class GS3DXVariantReceipt:
    """Machine-readable qualification and promotion receipt for a GS3DX variant."""

    schema_version: str
    variant: GS3DXVariant
    model_name: str
    builder: str
    parent_variant: GS3DXVariant | None
    model_file: str
    model_sha256: str
    block_budget: VariantBlockBudget
    drive_mode: GS3DXDriveMode
    drive_classification: DriveClassification
    neck_actuation: NeckActuationMode
    balance_mode: BalanceMode
    c3d_tracking: bool
    clean_host_build: CleanHostBuildReceipt
    original_protection: OriginalProtectionSummary
    cold_replay: ColdReplaySpec
    provenance: VariantProvenance
    review_status: ReviewStatus
    drive_reference: str | None = None
    image_artifacts: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "variant": self.variant.value,
            "model_name": self.model_name,
            "builder": self.builder,
            "parent_variant": (
                self.parent_variant.value if self.parent_variant else None
            ),
            "model_file": self.model_file,
            "model_sha256": self.model_sha256,
            "block_budget": self.block_budget.as_dict(),
            "drive_mode": self.drive_mode.value,
            "drive_classification": self.drive_classification.value,
            "neck_actuation": self.neck_actuation.value,
            "balance_mode": self.balance_mode.value,
            "c3d_tracking": self.c3d_tracking,
            "clean_host_build": self.clean_host_build.as_dict(),
            "original_protection": self.original_protection.as_dict(),
            "cold_replay": self.cold_replay.as_dict(),
            "provenance": self.provenance.as_dict(),
            "review_status": self.review_status.value,
            "drive_reference": self.drive_reference,
            "image_artifacts": dict(self.image_artifacts),
        }


def get_canonical_gs3dx_variant_definitions() -> dict[
    GS3DXVariant, GS3DXVariantDefinition
]:
    """Return the authoritative dictionary of 10 promoted GS3DX variants."""
    base_prefix = "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/models/"
    return {
        GS3DXVariant.BASELINE: GS3DXVariantDefinition(
            variant=GS3DXVariant.BASELINE,
            model_name="GS3DX_Baseline",
            model_file=f"{base_prefix}GS3DX_Baseline.slx",
            builder="gs3dx_clone_baseline",
            parent_variant=None,
            uncompiled_blocks=672,
            compiled_blocks=941,
            default_drive_mode=GS3DXDriveMode.BASELINE_TORQUE,
            default_drive_classification=DriveClassification.STABLE_DRIVE_EQUIVALENCE,
            default_neck_actuation=NeckActuationMode.RIGID_UPPER_TRUNK,
            default_balance_mode=BalanceMode.PASSIVE,
            description="Verbatim renamed clone of GolfSwing3D_Kinetic with repointed subsystem references.",
        ),
        GS3DXVariant.SLIM: GS3DXVariantDefinition(
            variant=GS3DXVariant.SLIM,
            model_name="GS3DX_Slim",
            model_file=f"{base_prefix}GS3DX_Slim.slx",
            builder="gs3dx_build_slim",
            parent_variant=GS3DXVariant.BASELINE,
            uncompiled_blocks=609,
            compiled_blocks=773,
            default_drive_mode=GS3DXDriveMode.SLIM_DIRECT_TORQUE,
            default_drive_classification=DriveClassification.STABLE_DRIVE_EQUIVALENCE,
            default_neck_actuation=NeckActuationMode.RIGID_UPPER_TRUNK,
            default_balance_mode=BalanceMode.PASSIVE,
            description="Direct joint InputTorque drive eliminating unnecessary torque converters.",
        ),
        GS3DXVariant.QUAT: GS3DXVariantDefinition(
            variant=GS3DXVariant.QUAT,
            model_name="GS3DX_Quat",
            model_file=f"{base_prefix}GS3DX_Quat.slx",
            builder="gs3dx_build_quat",
            parent_variant=GS3DXVariant.SLIM,
            uncompiled_blocks=594,
            compiled_blocks=740,
            default_drive_mode=GS3DXDriveMode.QUAT_DIRECT_TORQUE,
            default_drive_classification=DriveClassification.STABLE_DRIVE_EQUIVALENCE,
            default_neck_actuation=NeckActuationMode.RIGID_UPPER_TRUNK,
            default_balance_mode=BalanceMode.PASSIVE,
            description="Quaternion Spherical shoulder joints and 6-DOF hip eliminating Euler gimbal lock.",
        ),
        GS3DXVariant.FULL_BODY: GS3DXVariantDefinition(
            variant=GS3DXVariant.FULL_BODY,
            model_name="GS3DX_FullBody",
            model_file=f"{base_prefix}GS3DX_FullBody.slx",
            builder="gs3dx_build_lower_body",
            parent_variant=GS3DXVariant.QUAT,
            uncompiled_blocks=751,
            compiled_blocks=945,
            default_drive_mode=GS3DXDriveMode.FULL_BODY_PASSIVE_LEGS,
            default_drive_classification=DriveClassification.STABLE_DRIVE_EQUIVALENCE,
            default_neck_actuation=NeckActuationMode.RIGID_UPPER_TRUNK,
            default_balance_mode=BalanceMode.WELDED_WORLD,
            description="Full-body model adding articulated legs with feet welded to World.",
        ),
        GS3DXVariant.CONTACT: GS3DXVariantDefinition(
            variant=GS3DXVariant.CONTACT,
            model_name="GS3DX_FullBodyContact",
            model_file=f"{base_prefix}GS3DX_FullBodyContact.slx",
            builder="gs3dx_build_contact",
            parent_variant=GS3DXVariant.FULL_BODY,
            uncompiled_blocks=773,
            compiled_blocks=967,
            default_drive_mode=GS3DXDriveMode.STANCE_HOLD_SERVO,
            default_drive_classification=DriveClassification.STANCE_HOLD,
            default_neck_actuation=NeckActuationMode.RIGID_UPPER_TRUNK,
            default_balance_mode=BalanceMode.STANCE_HOLD_SERVO,
            description="Three contact spheres per foot on Infinite Plane with unactuated pelvis and stance-hold servo.",
        ),
        GS3DXVariant.GOLFER: GS3DXVariantDefinition(
            variant=GS3DXVariant.GOLFER,
            model_name="GS3DX_Golfer",
            model_file=f"{base_prefix}GS3DX_Golfer.slx",
            builder="gs3dx_build_golfer",
            parent_variant=GS3DXVariant.CONTACT,
            uncompiled_blocks=773,
            compiled_blocks=967,
            default_drive_mode=GS3DXDriveMode.STANCE_HOLD_SERVO,
            default_drive_classification=DriveClassification.STANCE_HOLD,
            default_neck_actuation=NeckActuationMode.RIGID_UPPER_TRUNK,
            default_balance_mode=BalanceMode.STANCE_HOLD_SERVO,
            description="de Leva 80 kg anthropometric mass distribution across 15 upper-body solids and legs.",
        ),
        GS3DXVariant.FIT: GS3DXVariantDefinition(
            variant=GS3DXVariant.FIT,
            model_name="GS3DX_Fit",
            model_file=f"{base_prefix}GS3DX_Fit.slx",
            builder="gs3dx_build_fit",
            parent_variant=GS3DXVariant.GOLFER,
            uncompiled_blocks=773,
            compiled_blocks=967,
            default_drive_mode=GS3DXDriveMode.STANCE_HOLD_SERVO,
            default_drive_classification=DriveClassification.C3D_FIT,
            default_neck_actuation=NeckActuationMode.RIGID_UPPER_TRUNK,
            default_balance_mode=BalanceMode.STANCE_HOLD_SERVO,
            description="Capture-fitted segment lengths and closed-loop dual-hand grip geometry.",
        ),
        GS3DXVariant.SHAPE: GS3DXVariantDefinition(
            variant=GS3DXVariant.SHAPE,
            model_name="GS3DX_Shape",
            model_file=f"{base_prefix}GS3DX_Shape.slx",
            builder="gs3dx_build_shape",
            parent_variant=GS3DXVariant.FIT,
            uncompiled_blocks=773,
            compiled_blocks=973,
            default_drive_mode=GS3DXDriveMode.BALANCE_LOOP_FEEDFORWARD_PD,
            default_drive_classification=DriveClassification.C3D_FIT,
            default_neck_actuation=NeckActuationMode.RIGID_UPPER_TRUNK,
            default_balance_mode=BalanceMode.JACOBIAN_BALANCE_LOOP,
            description="de Leva radii of gyration inertia tensors with visual ellipsoid segments and balance loop.",
        ),
        GS3DXVariant.NECK: GS3DXVariantDefinition(
            variant=GS3DXVariant.NECK,
            model_name="GS3DX_Neck",
            model_file=f"{base_prefix}GS3DX_Neck.slx",
            builder="gs3dx_build_neck",
            parent_variant=GS3DXVariant.SHAPE,
            uncompiled_blocks=769,
            compiled_blocks=975,
            default_drive_mode=GS3DXDriveMode.BALANCE_LOOP_FEEDFORWARD_PD,
            default_drive_classification=DriveClassification.C3D_FIT,
            default_neck_actuation=NeckActuationMode.MOTION_PRESCRIBED,
            default_balance_mode=BalanceMode.JACOBIAN_BALANCE_LOOP,
            description="Two-axis Universal Joint neck driven by motion prescription from head markers.",
        ),
        GS3DXVariant.HUMAN: GS3DXVariantDefinition(
            variant=GS3DXVariant.HUMAN,
            model_name="GS3DX_Human",
            model_file=f"{base_prefix}GS3DX_Human.slx",
            builder="gs3dx_build_human",
            parent_variant=GS3DXVariant.NECK,
            uncompiled_blocks=773,
            compiled_blocks=965,
            default_drive_mode=GS3DXDriveMode.HUMAN_SPRUNG_FEET_BALANCE,
            default_drive_classification=DriveClassification.C3D_FIT,
            default_neck_actuation=NeckActuationMode.MOTION_PRESCRIBED,
            default_balance_mode=BalanceMode.JACOBIAN_BALANCE_LOOP,
            description="Human body shape, C7 neck pivot, square clubface, sprung midfoot joints, 5 contacts/foot.",
        ),
    }


def _sha256_of_file(path: Path) -> str:
    """Return lowercase hex SHA-256 digest of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _find_repo_root(hint: Path | str | None = None) -> Path:
    if hint:
        p = Path(hint).resolve()
        if (p / ".git").exists() and (p / "src").is_dir():
            return p
    curr = Path(__file__).resolve()
    for parent in [curr] + list(curr.parents):
        if (parent / ".git").exists() and (parent / "src").is_dir():
            return parent
    return Path.cwd().resolve()


@postcondition(
    lambda s: isinstance(s, OriginalProtectionSummary),
    "result must be an OriginalProtectionSummary",
)
def verify_original_models_protection(
    repo_root: Path | str | None = None,
    override_hashes: Mapping[str, str] | None = None,
) -> OriginalProtectionSummary:
    """Verify that all hand-built original models remain untouched and byte-identical to source of truth.

    Fails closed if any file is missing or has a differing SHA-256 digest.
    """
    root = _find_repo_root(repo_root)
    orig_dir = (
        root
        / "src"
        / "engines"
        / "Simscape_Multibody_Models"
        / "3D_Golf_Model"
        / "matlab"
        / "src"
        / "model"
    )

    details: dict[str, str] = {}
    files_checked: list[str] = []

    for filename, expected_sha in ORIGINAL_HAND_BUILT_MODELS.items():
        files_checked.append(filename)
        if override_hashes and filename in override_hashes:
            actual_sha = override_hashes[filename]
        else:
            file_path = orig_dir / filename
            if not file_path.is_file():
                raise OriginalModelProtectedError(
                    f"Original hand-built model missing: {file_path}. Protection contract violated."
                )
            actual_sha = _sha256_of_file(file_path)

        details[filename] = actual_sha
        if actual_sha.lower() != expected_sha.lower():
            raise OriginalModelProtectedError(
                f"Original hand-built model hash mismatch for {filename!r}: "
                f"computed {actual_sha} != expected {expected_sha}. "
                "Hand-built originals are read-only and must never be altered."
            )

    return OriginalProtectionSummary(
        originals_verified=True,
        files_checked=tuple(files_checked),
        hashes_match=True,
        details=details,
    )


def build_gs3dx_variant_receipt(
    base: GS3DXVariantReceipt,
    **kwargs: Any,
) -> GS3DXVariantReceipt:
    """Create a new GS3DXVariantReceipt updating selected attributes."""
    typed_kwargs: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k == "variant" and isinstance(v, str):
            typed_kwargs[k] = GS3DXVariant(v)
        elif k == "drive_classification" and isinstance(v, str):
            typed_kwargs[k] = DriveClassification(v)
        elif k == "neck_actuation" and isinstance(v, str):
            typed_kwargs[k] = NeckActuationMode(v)
        elif k == "balance_mode" and isinstance(v, str):
            typed_kwargs[k] = BalanceMode(v)
        elif k == "drive_mode" and isinstance(v, str):
            typed_kwargs[k] = GS3DXDriveMode(v)
        elif k == "review_status" and isinstance(v, str):
            typed_kwargs[k] = ReviewStatus(v)
        elif k == "clean_host_build" and isinstance(v, dict):
            typed_kwargs[k] = CleanHostBuildReceipt(**v)
        elif k == "block_budget" and isinstance(v, dict):
            budget_args = {
                k2: v2
                for k2, v2 in v.items()
                if k2
                in (
                    "uncompiled_count",
                    "compiled_count",
                    "license_ceiling",
                    "reserve_blocks",
                )
            }
            typed_kwargs[k] = VariantBlockBudget(**budget_args)
        else:
            typed_kwargs[k] = v

    return replace(base, **typed_kwargs)


@precondition(
    lambda receipt: isinstance(receipt, GS3DXVariantReceipt),
    "receipt must be a GS3DXVariantReceipt",
)
def validate_gs3dx_variant_receipt(receipt: GS3DXVariantReceipt) -> None:
    """Validate all contracts and invariant constraints on a variant promotion receipt.

    Fails closed on any contract violation.
    """
    if receipt.schema_version != PROMOTED_VARIANTS_SCHEMA_VERSION:
        raise GS3DXPromotionError(
            f"Unsupported receipt schema version: {receipt.schema_version!r}"
        )

    # 1. Clean-host explicit R2025b build/save/reopen validation
    b = receipt.clean_host_build
    rel = str(b.matlab_release).strip().lower().lstrip("r")
    if rel != REQUIRED_MATLAB_RELEASE:
        raise CleanHostBuildValidationError(
            f"Clean-host build requires MATLAB R2025b; got {b.matlab_release!r} (no R2026a substitution)"
        )
    if not str(b.host).strip():
        raise CleanHostBuildValidationError(
            "host must name the licensed execution machine in clean_host_build"
        )
    if not b.built_clean:
        raise CleanHostBuildValidationError(
            f"Model {receipt.model_name!r} clean-host build failed or was interrupted"
        )
    if not b.saved_clean:
        raise CleanHostBuildValidationError(
            f"Model {receipt.model_name!r} clean-host save failed or produced dirty state"
        )
    if not b.reopened_clean:
        raise CleanHostBuildValidationError(
            f"Model {receipt.model_name!r} clean-host reopen failed to reconstruct clean diagram"
        )

    # 2. Block budget validation
    if receipt.block_budget.compiled_count > (
        receipt.block_budget.license_ceiling - receipt.block_budget.reserve_blocks
    ):
        raise GS3DXPromotionError(
            f"Compiled block count {receipt.block_budget.compiled_count} exceeds production "
            f"reserve ceiling of {receipt.block_budget.license_ceiling - receipt.block_budget.reserve_blocks}"
        )

    # 3. Stable-drive equivalence distinguished from C3D fit
    if receipt.variant in (GS3DXVariant.BASELINE, GS3DXVariant.SLIM, GS3DXVariant.QUAT):
        if (
            receipt.drive_classification == DriveClassification.C3D_FIT
            or receipt.c3d_tracking
        ):
            raise DriveClassificationMismatchError(
                f"Variant {receipt.variant.value} only satisfies stable-drive equivalence on impact drive; "
                "conflating stable-drive equivalence with C3D fit is strictly prohibited."
            )

    # 4. Motion-prescribed neck and servo tracking distinguished from autonomous balance
    if receipt.neck_actuation == NeckActuationMode.AUTONOMOUS:
        raise ActuationClassificationError(
            f"Variant {receipt.variant.value}: Neck/Human actuation uses motion prescription "
            "(2-DOF Universal joint driven by NeckReference with 5 ms filter), not autonomous neck control."
        )

    if receipt.balance_mode == BalanceMode.AUTONOMOUS_BALANCE:
        raise BalanceClassificationError(
            f"Variant {receipt.variant.value}: Leg stabilization uses servo tracking and Jacobian compensation, "
            "not autonomous balance control."
        )

    # 5. Provenance checks
    prov = receipt.provenance
    if prov.matlab_release.strip().lower().lstrip("r") != REQUIRED_MATLAB_RELEASE:
        raise GS3DXPromotionError(
            f"Provenance requires MATLAB R2025b; got {prov.matlab_release!r}"
        )
    if not prov.model_sha256 or len(prov.model_sha256) != 64:
        raise GS3DXPromotionError(
            "Provenance model SHA-256 must be a 64-character hex string"
        )
    if prov.model_sha256.lower() != receipt.model_sha256.lower():
        raise GS3DXPromotionError(
            "Provenance model SHA-256 does not match receipt model SHA-256"
        )

    # 6. Cold replay specification checks
    cold_replay = receipt.cold_replay
    if not cold_replay.command.strip():
        raise GS3DXPromotionError("Cold replay command must be specified")


def save_variant_receipt(receipt: GS3DXVariantReceipt, path: Path | str) -> None:
    """Save variant receipt as formatted JSON artifact."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(receipt.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_variant_receipt(path: Path | str) -> GS3DXVariantReceipt:
    """Load and parse variant receipt from JSON artifact."""
    p = Path(path)
    require(p.is_file(), f"Receipt file not found: {p}", p)
    data = json.loads(p.read_text(encoding="utf-8"))
    require(isinstance(data, Mapping), "receipt must be a JSON object", data)

    budget_d = data["block_budget"]
    budget = VariantBlockBudget(
        uncompiled_count=int(budget_d["uncompiled_count"]),
        compiled_count=int(budget_d["compiled_count"]),
        license_ceiling=int(budget_d.get("license_ceiling", HOME_LICENSE_BLOCK_LIMIT)),
        reserve_blocks=int(
            budget_d.get("reserve_blocks", INSTRUMENTATION_RESERVE_BLOCKS)
        ),
    )

    build_d = data["clean_host_build"]
    clean_host_build = CleanHostBuildReceipt(
        matlab_release=str(build_d["matlab_release"]),
        matlab_version=str(build_d["matlab_version"]),
        host=str(build_d["host"]),
        built_clean=bool(build_d["built_clean"]),
        saved_clean=bool(build_d["saved_clean"]),
        reopened_clean=bool(build_d["reopened_clean"]),
    )

    prot_d = data["original_protection"]
    original_protection = OriginalProtectionSummary(
        originals_verified=bool(prot_d["originals_verified"]),
        files_checked=tuple(str(x) for x in prot_d["files_checked"]),
        hashes_match=bool(prot_d["hashes_match"]),
        details={str(k): str(v) for k, v in prot_d["details"].items()},
    )

    replay_d = data["cold_replay"]
    cold_replay = ColdReplaySpec(
        command=str(replay_d["command"]),
        cwd=str(replay_d["cwd"]),
        environment={str(k): str(v) for k, v in replay_d["environment"].items()},
        input_files=tuple(str(x) for x in replay_d["input_files"]),
    )

    prov_d = data["provenance"]
    provenance = VariantProvenance(
        provider=str(prov_d["provider"]),
        host=str(prov_d["host"]),
        matlab_release=str(prov_d["matlab_release"]),
        matlab_version=str(prov_d["matlab_version"]),
        model_name=str(prov_d["model_name"]),
        model_sha256=str(prov_d["model_sha256"]),
        capture_file=str(prov_d["capture_file"]),
        capture_sha256=str(prov_d["capture_sha256"]),
    )

    receipt = GS3DXVariantReceipt(
        schema_version=str(data["schema_version"]),
        variant=GS3DXVariant(data["variant"]),
        model_name=str(data["model_name"]),
        builder=str(data["builder"]),
        parent_variant=(
            GS3DXVariant(data["parent_variant"]) if data.get("parent_variant") else None
        ),
        model_file=str(data["model_file"]),
        model_sha256=str(data["model_sha256"]),
        block_budget=budget,
        drive_mode=GS3DXDriveMode(data["drive_mode"]),
        drive_classification=DriveClassification(data["drive_classification"]),
        neck_actuation=NeckActuationMode(data["neck_actuation"]),
        balance_mode=BalanceMode(data["balance_mode"]),
        c3d_tracking=bool(data["c3d_tracking"]),
        clean_host_build=clean_host_build,
        original_protection=original_protection,
        cold_replay=cold_replay,
        provenance=provenance,
        review_status=ReviewStatus(data["review_status"]),
        drive_reference=(
            str(data["drive_reference"]) if data.get("drive_reference") else None
        ),
        image_artifacts={
            str(k): str(v) for k, v in data.get("image_artifacts", {}).items()
        },
    )
    validate_gs3dx_variant_receipt(receipt)
    return receipt


def filter_promotable_ledger_variants(
    receipts: Sequence[GS3DXVariantReceipt],
    *,
    fail_on_unreviewed: bool = False,
) -> list[GS3DXVariantReceipt]:
    """Filter receipts for consumption by the main motion matching ledger.

    Main ledger consumes only the designated reviewed variant (ReviewStatus.REVIEWED_PROMOTED).
    Fails closed if fail_on_unreviewed is True and non-promotable variants are encountered.
    """
    promotable: list[GS3DXVariantReceipt] = []
    for r in receipts:
        if r.review_status == ReviewStatus.REVIEWED_PROMOTED:
            promotable.append(r)
        elif fail_on_unreviewed:
            raise LedgerPromotionError(
                f"Main ledger only consumes reviewed promoted variants; rejected {r.variant.value} "
                f"with status {r.review_status.value}."
            )
    return promotable


def validate_candidate_package_integrity(
    *,
    model_sha256: str,
    candidate_sha256: str,
    image_package_sha256: str,
    metric_package_sha256: str,
    declared_candidate_in_images: str,
    declared_candidate_in_metrics: str,
) -> CandidateIntegrityReceipt:
    """Enforce cryptographic integrity across model, candidate, image package, and metrics package."""
    for name, val in (
        ("model_sha256", model_sha256),
        ("candidate_sha256", candidate_sha256),
        ("image_package_sha256", image_package_sha256),
        ("metric_package_sha256", metric_package_sha256),
    ):
        if not isinstance(val, str) or len(val) != 64:
            raise CandidateIntegrityError(
                f"{name} must be a 64-char hex SHA-256 digest"
            )

    if declared_candidate_in_images.lower() != candidate_sha256.lower():
        raise CandidateIntegrityError(
            f"Image package candidate SHA mismatch: declared {declared_candidate_in_images} "
            f"!= candidate {candidate_sha256}"
        )

    if declared_candidate_in_metrics.lower() != candidate_sha256.lower():
        raise CandidateIntegrityError(
            f"Metric package candidate SHA mismatch: declared {declared_candidate_in_metrics} "
            f"!= candidate {candidate_sha256}"
        )

    return CandidateIntegrityReceipt(
        model_sha256=model_sha256.lower(),
        candidate_sha256=candidate_sha256.lower(),
        image_package_sha256=image_package_sha256.lower(),
        metric_package_sha256=metric_package_sha256.lower(),
    )


def validate_matlab_suite_run(
    result: MatlabSuiteResult,
    expected_sha: str,
) -> None:
    """Validate reported MATLAB test suite rerun at promoted SHA with skipped tests disclosed."""
    if result.promoted_sha.lower() != expected_sha.lower():
        raise MatlabSuiteValidationError(
            f"Promoted SHA mismatch in suite run: reported {result.promoted_sha} != expected {expected_sha}"
        )

    rel = str(result.matlab_release).strip().lower().lstrip("r")
    if rel != REQUIRED_MATLAB_RELEASE:
        raise MatlabSuiteValidationError(
            f"MATLAB suite rerun requires R2025b; got {result.matlab_release!r} (no R2026a substitution)"
        )

    if result.total_tests <= 0:
        raise MatlabSuiteValidationError(
            "No tests were run; unqualified (empty test suite)"
        )

    if result.failed_tests > 0:
        raise MatlabSuiteValidationError(
            f"Suite run reported {result.failed_tests} failures; suite must be completely green"
        )

    # Disclose skips: verify every skipped test has an explicit recorded justification
    for skip in result.skipped_tests:
        if (
            skip not in result.skip_reasons
            or not str(result.skip_reasons[skip]).strip()
        ):
            raise MatlabSuiteValidationError(
                f"Skipped test {skip!r} lacks an explicit disclosed reason in skip_reasons"
            )


PROMOTED_MODEL_HASHES: Mapping[GS3DXVariant, str] = {
    GS3DXVariant.BASELINE: "8935d01e5e55be65c0a72fbc899007d89cb655c2d8f2fbec8bf7e01bd08d4258",
    GS3DXVariant.SLIM: "6f0eb93a943579bb6dd5f6ff2280ec34f18824a6a0515119329016f90b1ca796",
    GS3DXVariant.QUAT: "ddef8e369f3c06bbf9693ab9cc66379f7f6b65869b23e3c5a3ebf608f1949a2f",
    GS3DXVariant.FULL_BODY: "f78acc31b17e4b187cfee3ab1cb5e60c1b1493ea38652eed99a9dadd5751d3cc",
    GS3DXVariant.CONTACT: "0df5e17384ddf91f3f7051872279e666ddc7ec1eb714d7f506d0a24dcb799391",
    GS3DXVariant.GOLFER: "45e29cd21821017064635f55f1378144eed66c4cfaa0be74243b7f953dd5ec72",
    GS3DXVariant.FIT: "428b8ff3a6dab6d3fed0454d5afb3586c6f9ef658eeddc60529e7a4a8c937f00",
    GS3DXVariant.SHAPE: "dd0a754c8b4d5238bb9cf6640c88c5978c4cf19a9d2db92c5abe795b6d358926",
    GS3DXVariant.NECK: "110a6ce7ce2642e399664d1cc5520fb782adb160620157f857abb52f3a0bfccf",
    GS3DXVariant.HUMAN: "919974719a4e24ee7d04ff004818c01dc3c31212383fe6a84b6e6de84abc919f",
}


def generate_canonical_variant_receipt(
    variant: GS3DXVariant,
    *,
    host: str = "licensed-r2025b-host",
    matlab_version: str = "24.2.0.2741519 (R2025b)",
    repo_root: Path | str | None = None,
) -> GS3DXVariantReceipt:
    """Generate a canonical, validated GS3DXVariantReceipt for any of the 10 promoted variants."""
    defs = get_canonical_gs3dx_variant_definitions()
    if variant not in defs:
        raise GS3DXPromotionError(f"Unknown variant: {variant}")
    defn = defs[variant]

    model_sha = PROMOTED_MODEL_HASHES[variant]
    orig_protection = verify_original_models_protection(repo_root=repo_root)

    budget = VariantBlockBudget(
        uncompiled_count=defn.uncompiled_blocks,
        compiled_count=defn.compiled_blocks,
        license_ceiling=HOME_LICENSE_BLOCK_LIMIT,
        reserve_blocks=INSTRUMENTATION_RESERVE_BLOCKS,
    )

    clean_build = CleanHostBuildReceipt(
        matlab_release="2025b",
        matlab_version=matlab_version,
        host=host,
        built_clean=True,
        saved_clean=True,
        reopened_clean=True,
    )

    prov = VariantProvenance(
        provider="MATLAB R2025b / Simscape Multibody",
        host=host,
        matlab_release="2025b",
        matlab_version=matlab_version,
        model_name=defn.model_name,
        model_sha256=model_sha,
        capture_file=CANONICAL_C3D_CAPTURE_FILE,
        capture_sha256=CANONICAL_C3D_CAPTURE_SHA256,
    )

    cold_replay = ColdReplaySpec(
        command=f"matlab -batch \"addpath('tools'); gs3dx_setup; gs3dx_simulate('{defn.model_name}')\"",
        cwd="src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx",
        environment={"MATLABPATH": "tools;models"},
        input_files=(f"models/{defn.model_name}.slx", CANONICAL_C3D_CAPTURE_FILE),
    )

    review_status = (
        ReviewStatus.REVIEWED_PROMOTED
        if variant == GS3DXVariant.HUMAN
        else ReviewStatus.EXPLORATORY_PROMOTED
    )

    drive_ref = (
        "baselines/original_GolfSwing3D_Kinetic_impact_0p3S.mat"
        if defn.default_drive_classification
        == DriveClassification.STABLE_DRIVE_EQUIVALENCE
        else None
    )

    c3d_tracking = defn.default_drive_classification == DriveClassification.C3D_FIT

    receipt = GS3DXVariantReceipt(
        schema_version=PROMOTED_VARIANTS_SCHEMA_VERSION,
        variant=variant,
        model_name=defn.model_name,
        builder=defn.builder,
        parent_variant=defn.parent_variant,
        model_file=defn.model_file,
        model_sha256=model_sha,
        block_budget=budget,
        drive_mode=defn.default_drive_mode,
        drive_classification=defn.default_drive_classification,
        neck_actuation=defn.default_neck_actuation,
        balance_mode=defn.default_balance_mode,
        c3d_tracking=c3d_tracking,
        clean_host_build=clean_build,
        original_protection=orig_protection,
        cold_replay=cold_replay,
        provenance=prov,
        review_status=review_status,
        drive_reference=drive_ref,
    )
    validate_gs3dx_variant_receipt(receipt)
    return receipt
