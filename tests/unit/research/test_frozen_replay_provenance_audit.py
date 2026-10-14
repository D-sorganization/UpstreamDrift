"""Unit tests for frozen research replay provenance auditor and ledger (#12398)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.frozen_replay_provenance_audit import (
    CANONICAL_MEASURED_BOUNDS,
    FROZEN_ARTIFACT_DIGESTS,
    audit_frozen_replay_studies,
    capture_runtime_provenance,
    quantify_replay_discrepancy_ledger,
    reconcile_source_authority_amendment,
    verify_frozen_artifact_digests,
)

pytestmark = pytest.mark.unit


def test_capture_runtime_provenance_distinguishes_lockfile_pins() -> None:
    record = capture_runtime_provenance()

    assert record.provenance_kind == "verified_runtime"
    assert record.is_runtime_verified is True
    assert record.lockfile_pinned is False
    assert len(record.python_version) > 0
    assert len(record.python_implementation) > 0
    assert len(record.platform_system) > 0
    assert len(record.platform_release) > 0
    assert len(record.numpy_version) > 0
    assert len(record.scipy_version) > 0
    assert len(record.blas_lapack_fingerprint) == 64


def test_verify_frozen_artifact_digests_matches_immutable_hashes() -> None:
    results = verify_frozen_artifact_digests()

    assert set(results.keys()) == set(FROZEN_ARTIFACT_DIGESTS.keys())
    for name, expected_sha in FROZEN_ARTIFACT_DIGESTS.items():
        res = results[name]
        assert res.filename == name
        assert res.expected_sha256 == expected_sha
        if res.exists_on_disk:
            assert res.actual_sha256 == expected_sha
            assert res.verified is True


def test_verify_frozen_artifact_digests_detects_tampering_and_missing(
    tmp_path: Path,
) -> None:
    tampered_file = tmp_path / "bounded_event_reachability.json"
    tampered_file.write_bytes(b'{"tampered": true}')

    results = verify_frozen_artifact_digests(data_dir=tmp_path)

    tampered_res = results["bounded_event_reachability.json"]
    assert tampered_res.exists_on_disk is True
    assert tampered_res.verified is False
    assert tampered_res.actual_sha256 != tampered_res.expected_sha256

    missing_res = results["forward_two_arm_study.json"]
    assert missing_res.exists_on_disk is False
    assert missing_res.verified is False
    assert missing_res.actual_sha256 is None


def test_reconcile_source_authority_amendment_identifies_pure_amendment() -> None:
    registered: dict[str, Any] = {
        "planar_closed_loop": {"rank": 4, "coordinate_scale_m": 0.75},
        "source_authority": {
            "path/a.json": {"schema_version": "v1", "sha256": "old_sha_a"},
            "path/b.json": {"schema_version": "v1", "sha256": "old_sha_b"},
        },
    }
    candidate: dict[str, Any] = {
        "planar_closed_loop": {"rank": 4, "coordinate_scale_m": 0.75},
        "source_authority": {
            "path/a.json": {"schema_version": "v1", "sha256": "new_sha_a"},
            "path/b.json": {"schema_version": "v1", "sha256": "new_sha_b"},
        },
    }

    result = reconcile_source_authority_amendment(registered, candidate)

    assert result.is_pure_amendment is True
    assert result.amended_sha_count == 2
    assert result.non_authority_discrepancies == 0
    assert result.classification == "pure_source_authority_amendment"
    amended_paths = {a.target_path for a in result.amendments}
    assert amended_paths == {"path/a.json", "path/b.json"}


def test_reconcile_source_authority_amendment_rejects_numerical_drift() -> None:
    registered: dict[str, Any] = {
        "planar_closed_loop": {"rank": 4, "coordinate_scale_m": 0.75},
        "source_authority": {
            "path/a.json": {"schema_version": "v1", "sha256": "old_sha_a"},
        },
    }
    candidate: dict[str, Any] = {
        "planar_closed_loop": {"rank": 3, "coordinate_scale_m": 0.75},
        "source_authority": {
            "path/a.json": {"schema_version": "v1", "sha256": "new_sha_a"},
        },
    }

    result = reconcile_source_authority_amendment(registered, candidate)

    assert result.is_pure_amendment is False
    assert result.amended_sha_count == 1
    assert result.non_authority_discrepancies == 1
    assert result.classification == "numerical_categorical_drift"


def test_reconcile_source_authority_amendment_identifies_identical() -> None:
    report: dict[str, Any] = {
        "metrics": {"rank": 4},
        "source_authority": {
            "path/a.json": {"schema_version": "v1", "sha256": "same_sha"},
        },
    }

    result = reconcile_source_authority_amendment(report, report)

    assert result.is_pure_amendment is False
    assert result.amended_sha_count == 0
    assert result.non_authority_discrepancies == 0
    assert result.classification == "identical"


def test_canonical_measured_bounds_integrity_and_non_relaxation() -> None:
    assert set(CANONICAL_MEASURED_BOUNDS.keys()) == {
        "bounded_event_reachability",
        "forward_two_arm_study",
        "constraint_internal_force_diagnostics",
    }

    reachability = CANONICAL_MEASURED_BOUNDS["bounded_event_reachability"]
    assert reachability.json_differing_leaves == 139
    assert reachability.json_max_abs_diff == 7e-6
    assert reachability.array_differing_count == 19
    assert reachability.array_max_abs_diff == 9.19169347857185e-7
    assert reachability.source_authority_sha_amendments == 0

    forward = CANONICAL_MEASURED_BOUNDS["forward_two_arm_study"]
    assert forward.json_differing_leaves == 48
    assert forward.json_max_abs_diff == 1.3997517594684788e-11
    assert forward.array_differing_count == 37
    assert forward.array_max_abs_diff == 1.3446310731524136e-10
    assert forward.source_authority_sha_amendments == 0

    constraint = CANONICAL_MEASURED_BOUNDS["constraint_internal_force_diagnostics"]
    assert constraint.json_differing_leaves == 0
    assert constraint.json_max_abs_diff == 0.0
    assert constraint.array_differing_count == 0
    assert constraint.array_max_abs_diff == 0.0
    assert constraint.source_authority_sha_amendments == 2


def test_quantify_replay_discrepancy_ledger_evaluates_bounds() -> None:
    reg_json = {"score": 10.0, "status": "nominal"}
    cand_json = {"score": 10.000005, "status": "nominal"}
    reg_arrays = {"q": np.array([1.0, 2.0]), "v": np.array([0.1])}
    cand_arrays = {"q": np.array([1.0000005, 2.0]), "v": np.array([0.1])}

    entry = quantify_replay_discrepancy_ledger(
        registered_json=reg_json,
        candidate_json=cand_json,
        registered_arrays=reg_arrays,
        candidate_arrays=cand_arrays,
        study_name="bounded_event_reachability",
    )

    assert entry.study_name == "bounded_event_reachability"
    assert entry.json_differing_leaves == 1
    assert entry.json_max_abs_diff == pytest.approx(5e-6)
    assert entry.array_differing_count == 1
    assert entry.array_max_abs_diff == pytest.approx(5e-7)
    assert entry.tolerances_relaxed is False
    assert entry.within_canonical_bounds is True
    assert entry.canonical_bound is not None


def test_quantify_replay_discrepancy_ledger_exceeding_bounds() -> None:
    reg_json = {"score": 10.0}
    cand_json = {"score": 20.0}

    entry = quantify_replay_discrepancy_ledger(
        registered_json=reg_json,
        candidate_json=cand_json,
        study_name="forward_two_arm_study",
    )

    assert entry.within_canonical_bounds is False
    assert entry.tolerances_relaxed is False


def test_audit_frozen_replay_studies_structure() -> None:
    audit = audit_frozen_replay_studies()

    assert audit["schema_version"] == "frozen-replay-provenance-audit-v1"
    assert "runtime_provenance" in audit
    assert "artifact_digests" in audit
    assert "canonical_bounds" in audit
    assert audit["runtime_provenance"]["is_runtime_verified"] is True
