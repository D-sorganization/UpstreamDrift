"""Static catalog of known neural corpora, checkpoints and training claims."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.shared.python.motion_matching.surrogate.artifact_paths import (
    DEFAULT_PRODUCTION_CHECKPOINT_REL,
    DEFAULT_SURROGATE_CHECKPOINT_REL,
    DOCUMENTED_TEN_THOUSAND_FILES,
    SWEEP_SYNTHETIC_REL,
)

from .types import ArtifactKind, ArtifactRole

# Re-export for callers that historically imported from catalog.
__all__ = [
    "CatalogEntry",
    "DOCUMENTED_TEN_THOUSAND_FILES",
    "HISTORICAL_DESIGN_ISSUES",
    "SWEEP_REQUIRED_CHANNELS",
    "default_catalog",
]

SWEEP_REQUIRED_CHANNELS: tuple[str, ...] = (
    "trial_id",
    "t",
    "q",
    "qd",
    "qdd",
    "tau",
)


@dataclass(frozen=True)
class CatalogEntry:
    """Declarative description of an artifact to audit at a repo root."""

    artifact_id: str
    kind: ArtifactKind
    role: ArtifactRole
    # Absolute documented path OR path relative to repo root.
    path_spec: Path
    path_is_absolute: bool
    is_synthetic_fixture: bool
    schema_version: str | None
    engine: str | None
    units: str | None
    required_channels: tuple[str, ...]
    label_availability: dict[str, bool]
    related_issues: tuple[str, ...]
    retrieval_instructions: str
    notes: str = ""
    # When True, path is a directory with trials.parquet + timesteps.parquet.
    is_sweep_folder: bool = False


def _ten_thousand_entry() -> CatalogEntry:
    return CatalogEntry(
        artifact_id="corpus.ten_thousand_files",
        kind=ArtifactKind.REMOTE_POINTER,
        role=ArtifactRole.COMPACT_CORPUS,
        path_spec=DOCUMENTED_TEN_THOUSAND_FILES,
        path_is_absolute=True,
        is_synthetic_fixture=False,
        schema_version=None,
        engine="simscape",
        units="SI (joint radians, torques N·m; club positions metres)",
        required_channels=(),
        label_availability={
            "coefficients": False,
            "torques": False,
            "club_kinematics": False,
        },
        related_issues=("#4075", "#4076", "#10615"),
        retrieval_instructions=(
            "Locate the host-local Simscape dump historically documented at "
            f"{DOCUMENTED_TEN_THOUSAND_FILES}. If present, compute SHA-256, "
            "inspect parquet metadata/row groups with "
            "neural_motion.inspect_parquet_bounded, and record generation "
            "provenance (model release, seeds, geometry, contact) before any "
            "training claim. Do not invent 10k training results when absent."
        ),
        notes=(
            "Documented in MachineLearning README / CONTROL_STRATEGY and "
            "surrogate/perstep/extract_dataset.py DEFAULT_SOURCE. REVIEW.md "
            "confirmed absence on the planning host."
        ),
    )


def _sweep_synthetic_entry() -> CatalogEntry:
    return CatalogEntry(
        artifact_id="fixture.sweep_synthetic",
        kind=ArtifactKind.FIXTURE,
        role=ArtifactRole.SWEEP_CORPUS,
        path_spec=SWEEP_SYNTHETIC_REL,
        path_is_absolute=False,
        is_synthetic_fixture=True,
        schema_version="0.1.0",
        engine="synthetic",
        units="SI shape-valid only; not physically meaningful",
        required_channels=SWEEP_REQUIRED_CHANNELS,
        label_availability={
            "coefficients": True,
            "torques": True,
            "club_kinematics": True,
        },
        related_issues=("#10615",),
        retrieval_instructions=(
            "Tracked under data/sweep_synthetic/{trials,timesteps}.parquet. "
            "Load with motion_matching.dataset.load_sweep_dataset. Use only "
            "for software-contract tests; never as native physical supervision."
        ),
        notes="Toy fixture generated for loader validation.",
        is_sweep_folder=True,
    )


def _checkpoint_best_entry() -> CatalogEntry:
    return CatalogEntry(
        artifact_id="checkpoint.surrogate_best_default",
        kind=ArtifactKind.CHECKPOINT,
        role=ArtifactRole.FORWARD_SURROGATE,
        path_spec=DEFAULT_SURROGATE_CHECKPOINT_REL,
        path_is_absolute=False,
        is_synthetic_fixture=False,
        schema_version=None,
        engine="pytorch",
        units=None,
        required_channels=(
            "model_state_dict",
            "input_columns",
            "target_columns",
            "x_mean",
            "x_std",
            "y_mean",
            "y_std",
            "config",
        ),
        label_availability={},
        related_issues=("#4075", "#10615"),
        retrieval_instructions=(
            "Default PROJECT_SPEC path output/surrogate/checkpoint_best.pt. "
            "If present, load only via motion_matching._checkpoint_artifacts "
            "(weights_only=True). Prefer checkpoint_production.pt when both "
            "exist. Unsafe pickle payloads require MIGRATE."
        ),
        notes="Default training output path; not a qualified artifact.",
    )


def _checkpoint_production_entry() -> CatalogEntry:
    return CatalogEntry(
        artifact_id="checkpoint.surrogate_production_default",
        kind=ArtifactKind.CHECKPOINT,
        role=ArtifactRole.FORWARD_SURROGATE,
        path_spec=DEFAULT_PRODUCTION_CHECKPOINT_REL,
        path_is_absolute=False,
        is_synthetic_fixture=False,
        schema_version=None,
        engine="pytorch",
        units=None,
        required_channels=(
            "model_state_dict",
            "input_columns",
            "target_columns",
            "x_mean",
            "x_std",
            "y_mean",
            "y_std",
            "config",
        ),
        label_availability={},
        related_issues=("#4075", "#10615"),
        retrieval_instructions=(
            "Default production checkpoint path "
            "output/surrogate/checkpoint_production.pt. Same safe-load rules "
            "as checkpoint_best."
        ),
    )


def default_catalog() -> tuple[CatalogEntry, ...]:
    """Return the fixed NM-00 inventory surface (datasets, pointers, claims)."""
    return (
        _ten_thousand_entry(),
        _sweep_synthetic_entry(),
        _checkpoint_best_entry(),
        _checkpoint_production_entry(),
    )


# Historical issue ids that supply prior designs, not current speed evidence.
HISTORICAL_DESIGN_ISSUES: tuple[str, ...] = (
    "#4075",
    "#4076",
    "#3999",
    "#4000",
    "#6014",
    "#5419",
)
