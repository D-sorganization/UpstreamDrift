"""Training-claim classification: notes vs reproduced evidence (NM-00 #10615)."""

from __future__ import annotations

from pathlib import Path

from src.shared.python.core.contracts import postcondition, precondition

from .types import (
    ArtifactIdentity,
    ArtifactKind,
    ArtifactRole,
    ClaimStatus,
    Disposition,
)

# Source notes that document mean-baseline plateaus. These are NOT reproduced
# loss curves; NM-00 keeps them ClaimStatus.NOTE_ONLY until curves/hashes exist.
_PLATEAU_CLAIMS: dict[str, dict[str, str | tuple[str, ...]]] = {
    "inverse_cvae_mean_baseline_plateau": {
        "role": ArtifactRole.INVERSE_CVAE,
        "notes": (
            "Source notes in motion_matching/inverse/__init__.py and "
            "PROJECT_SPEC §5.2 report a hard reconstruction plateau near the "
            "mean-prediction baseline on the compact corpus."
        ),
        "blockers": (
            "No versioned loss-curve artifact or checkpoint hash reproduced in NM-00",
            "mean-baseline plateau is a note, not current training evidence",
        ),
        "related_issues": ("#4076", "#4240", "#10615"),
        "source_paths": (
            "src/shared/python/motion_matching/inverse/__init__.py",
            "src/engines/Simscape_Multibody_Models/3D_Golf_Model/PROJECT_SPEC.md",
        ),
    },
    "inverse_regressor_mean_baseline_plateau": {
        "role": ArtifactRole.INVERSE_REGRESSOR,
        "notes": (
            "Source notes in motion_matching/inverse/regressor.py report "
            "val_recon stuck at the mean-prediction baseline for 18+ epochs; "
            "inverse_timestep notes the same for trajectory→189 models."
        ),
        "blockers": (
            "No reproduced training curves or checkpoint hash under NM-00",
            "mean-baseline plateau note cannot authorize speed/accuracy claims",
        ),
        "related_issues": ("#4076", "#4267", "#10615"),
        "source_paths": (
            "src/shared/python/motion_matching/inverse/regressor.py",
            "src/shared/python/motion_matching/inverse_timestep/__init__.py",
            "src/shared/python/motion_matching/inverse_timestep/training.py",
        ),
    },
}


@precondition(
    lambda claim_id, repo_root=None: (
        isinstance(claim_id, str) and bool(claim_id.strip())
    ),
    "claim_id must be a non-empty string",
)
@postcondition(
    lambda result: result.claim_status != ClaimStatus.NATIVE_QUALIFIED,
    "plateau/historical claims cannot be native-qualified by note classification",
)
def classify_training_claim(
    claim_id: str,
    *,
    repo_root: Path | None = None,
) -> ArtifactIdentity:
    """Classify a named historical training claim without inventing evidence.

    Unknown claim ids are rejected. Known plateau notes return
    ``ClaimStatus.NOTE_ONLY`` with ``Disposition.QUARANTINE``.
    """
    key = claim_id.strip()
    if key not in _PLATEAU_CLAIMS:
        raise ValueError(f"unknown training claim id: {claim_id!r}")

    meta = _PLATEAU_CLAIMS[key]
    root = repo_root.resolve() if repo_root is not None else None
    source_paths = tuple(str(p) for p in meta["source_paths"])  # type: ignore[arg-type]
    if root is not None:
        missing = [p for p in source_paths if not (root / Path(p)).is_file()]
        blockers = list(meta["blockers"])  # type: ignore[arg-type]
        if missing:
            blockers.append(f"missing source note paths: {', '.join(missing)}")
    else:
        blockers = list(meta["blockers"])  # type: ignore[arg-type]

    return ArtifactIdentity(
        artifact_id=f"claim.{key}",
        kind=ArtifactKind.TRAINING_CLAIM,
        role=meta["role"],  # type: ignore[arg-type]
        path=None,
        exists=False,
        content_sha256=None,
        schema_version=None,
        model_ids=(),
        engine=None,
        source_revision=None,
        units=None,
        seeds=(),
        trial_ancestry=None,
        required_channels=(),
        label_availability={},
        is_synthetic_fixture=False,
        disposition=Disposition.QUARANTINE,
        claim_status=ClaimStatus.NOTE_ONLY,
        blockers=tuple(blockers),
        retrieval_instructions=(
            "Read the cited source note paths; reproduce curves under a frozen "
            "seed/split before promoting claim_status beyond NOTE_ONLY."
        ),
        notes=str(meta["notes"]),
        related_issues=tuple(meta["related_issues"]),  # type: ignore[arg-type]
    )


def list_known_claim_ids() -> tuple[str, ...]:
    """Return claim ids that ``classify_training_claim`` accepts."""
    return tuple(sorted(_PLATEAU_CLAIMS))
