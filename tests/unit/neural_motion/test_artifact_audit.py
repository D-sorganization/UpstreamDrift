"""NM-00 (#10615): fail-closed dataset/checkpoint/training-claim audit contracts.

Acceptance cases from the issue:
- identity mismatch is rejected
- synthetic fixtures cannot certify native supervision
- absent files are recorded explicitly (never invented 10k results)
- exact source revision and required channels are recorded
- representative legacy fixture loads through the real adapter
- missing provenance is quarantined
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.shared.python.motion_matching.dataset import (
    SCHEMA_VERSION as SWEEP_SCHEMA_VERSION,
    load_sweep_dataset,
)
from src.shared.python.neural_motion import (
    AUDIT_SCHEMA,
    ArtifactKind,
    ClaimStatus,
    Disposition,
    audit_neural_artifacts,
    classify_training_claim,
    generate_neural_coverage_matrix,
    inspect_parquet_bounded,
)
from src.shared.python.tour_baselines.registry import list_golf_models

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SWEEP_SYNTHETIC = REPO_ROOT / "data" / "sweep_synthetic"
DOCUMENTED_10K = Path(r"C:\Users\diete\Repositories\data\TenThousandFiles.parquet")


def test_absent_ten_thousand_files_is_explicit_quarantine() -> None:
    """Documented 10k parquet must not invent training results when absent."""
    receipt = audit_neural_artifacts(REPO_ROOT)
    row = receipt.artifact("corpus.ten_thousand_files")
    assert row.path == DOCUMENTED_10K
    assert row.exists is False
    assert row.content_sha256 is None
    assert row.disposition == Disposition.QUARANTINE
    assert row.claim_status == ClaimStatus.UNSUPPORTED
    assert "absent" in " ".join(row.blockers).lower()
    assert receipt.as_dict()["schema"] == AUDIT_SCHEMA


def test_synthetic_sweep_cannot_certify_native() -> None:
    """Tracked synthetic fixtures validate software contracts only."""
    receipt = audit_neural_artifacts(REPO_ROOT)
    row = receipt.artifact("fixture.sweep_synthetic")
    assert row.exists is True
    assert row.is_synthetic_fixture is True
    assert row.claim_status == ClaimStatus.SOFTWARE_CONTRACT_ONLY
    assert row.disposition in {Disposition.RETAIN, Disposition.REPAIR}
    assert ClaimStatus.NATIVE_QUALIFIED not in {
        a.claim_status for a in receipt.artifacts if a.is_synthetic_fixture
    }


def test_dataset_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    """Declared content hash that does not match the file fails closed."""
    payload = b"not-a-real-sweep-corpus"
    fake = tmp_path / "trials.parquet"
    fake.write_bytes(payload)
    wrong_sha = "0" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        inspect_parquet_bounded(
            fake,
            expected_sha256=wrong_sha,
            max_rows=8,
        )


def test_missing_provenance_is_quarantined() -> None:
    """Artifacts without model/engine/revision provenance cannot supervise."""
    receipt = audit_neural_artifacts(REPO_ROOT)
    quarantined = [
        a
        for a in receipt.artifacts
        if a.disposition == Disposition.QUARANTINE
        and a.kind != ArtifactKind.REMOTE_POINTER
    ]
    # At least the documented-but-unproven historical training claims quarantine.
    assert any(a.artifact_id.startswith("claim.") for a in receipt.artifacts)
    for claim in receipt.artifacts:
        if claim.kind == ArtifactKind.TRAINING_CLAIM:
            assert claim.claim_status in {
                ClaimStatus.NOTE_ONLY,
                ClaimStatus.UNSUPPORTED,
                ClaimStatus.SOFTWARE_CONTRACT_ONLY,
            }
            assert claim.claim_status != ClaimStatus.NATIVE_QUALIFIED
    assert quarantined or any(
        a.disposition == Disposition.QUARANTINE for a in receipt.artifacts
    )


def test_exact_source_revision_and_required_channels_on_fixture() -> None:
    """Inventory rows record schema/channels; synthetic fixture loads via adapter."""
    receipt = audit_neural_artifacts(REPO_ROOT)
    row = receipt.artifact("fixture.sweep_synthetic")
    assert row.schema_version == SWEEP_SCHEMA_VERSION
    assert row.source_revision is not None and len(row.source_revision) > 0
    assert "q" in row.required_channels
    assert "tau" in row.required_channels
    assert row.label_availability.get("coefficients") is True

    dataset = load_sweep_dataset(SWEEP_SYNTHETIC)
    assert dataset.n_trials() >= 1
    assert dataset.schema_version == SWEEP_SCHEMA_VERSION


def test_representative_legacy_fixture_bounded_inspect() -> None:
    """Inspect metadata/row groups and a bounded sample before bulk reads."""
    trials = SWEEP_SYNTHETIC / "trials.parquet"
    sample = inspect_parquet_bounded(trials, max_rows=4)
    assert sample.exists is True
    assert sample.num_row_groups >= 1
    assert sample.sampled_rows <= 4
    assert sample.content_sha256 == hashlib.sha256(trials.read_bytes()).hexdigest()
    assert sample.schema_names  # non-empty column inventory


def test_plateau_notes_are_note_only_not_reproduced_evidence() -> None:
    """CVAE/regressor mean-baseline plateau prose is note-only until curves exist."""
    cvae = classify_training_claim("inverse_cvae_mean_baseline_plateau")
    regressor = classify_training_claim("inverse_regressor_mean_baseline_plateau")
    assert cvae.claim_status == ClaimStatus.NOTE_ONLY
    assert regressor.claim_status == ClaimStatus.NOTE_ONLY
    assert cvae.disposition == Disposition.QUARANTINE
    assert (
        "mean" in " ".join(cvae.blockers).lower()
        or "plateau" in " ".join(cvae.blockers).lower()
    )


def test_neural_coverage_matrix_covers_every_registered_model() -> None:
    """Per-model dataset/checkpoint coverage table from the TB-00 roster."""
    models = list_golf_models()
    matrix = generate_neural_coverage_matrix(REPO_ROOT)
    model_ids = {m.model_id for m in models}
    assert {cell.model_id for cell in matrix} == model_ids
    for cell in matrix:
        assert cell.disposition in Disposition
        assert cell.dataset_id or cell.checkpoint_id or cell.blockers
        # No silent native certification from this audit alone.
        assert cell.claim_status != ClaimStatus.NATIVE_QUALIFIED


def test_audit_receipt_serializes_and_records_retrieval_instructions(
    tmp_path: Path,
) -> None:
    receipt = audit_neural_artifacts(REPO_ROOT)
    out = tmp_path / "neural_artifact_audit_receipt.json"
    receipt.write_json(out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == AUDIT_SCHEMA
    ten_k = next(
        a
        for a in payload["artifacts"]
        if a["artifact_id"] == "corpus.ten_thousand_files"
    )
    assert "retrieval_instructions" in ten_k
    assert ten_k["retrieval_instructions"]
    assert ten_k["exists"] is False
