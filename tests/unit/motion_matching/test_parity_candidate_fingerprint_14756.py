"""Candidate identity in parity reports includes provenance and all arrays."""

from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.motion_matching.candidate import (
    CandidateAuxiliary,
    CandidateMarkers,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)
from src.shared.python.motion_matching.candidate_io import save_candidate
from src.shared.python.motion_matching.parity_report import build_parity_report

pytestmark = pytest.mark.unit


def _candidate(*, metadata=None, **arrays) -> MatchedSwingCandidate:
    values = {
        "time_s": np.array([0.0, 0.1, 0.2]),
        "q": np.zeros((3, 2)),
        "v": np.zeros((3, 2)),
        "tau": np.zeros((3, 2)),
    }
    values.update(arrays)
    return MatchedSwingCandidate(
        metadata=metadata
        or CandidateMetadata(
            coordinate_names=("a", "b"),
            profile=(
                CandidateProfile.DYNAMIC
                if values["tau"] is not None
                else CandidateProfile.KINEMATIC
            ),
        ),
        **values,
    )


def _fingerprint(candidate) -> str:
    # No native engine or private capture is needed for candidate identity.
    return build_parity_report(
        candidate, engines=["identity-test"], reference_engine="identity-test", spec={}
    ).candidate_sha256


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_sha256", "model-b"),
        ("source_c3d_sha256", "capture-b"),
        ("document_sha256", "document-b"),
        ("coordinate_names", ("b", "a")),
        ("solver_settings", {"dt": 0.002}),
        ("extra", {"receipt_sha256": "receipt-b"}),
    ],
)
def test_identical_q_different_provenance_has_distinct_identity(field, value):
    original = _candidate()
    changed = _candidate(metadata=replace(original.metadata, **{field: value}))
    assert _fingerprint(original) != _fingerprint(changed)


@pytest.mark.parametrize(
    "arrays",
    [
        {"time_s": np.array([0.0, 0.2, 0.4])},
        {"v": np.ones((3, 2))},
        {"tau": np.ones((3, 2))},
        {"v": None, "tau": None},
        {"markers": CandidateMarkers(target_markers_m=np.ones((3, 1, 3)))},
        {"auxiliary": CandidateAuxiliary(external_forces=np.ones((3, 3)))},
    ],
)
def test_identical_q_different_evidence_has_distinct_identity(arrays):
    assert _fingerprint(_candidate()) != _fingerprint(_candidate(**arrays))


def test_identity_is_full_sha256_and_survives_archive_roundtrip(tmp_path):
    candidate = _candidate()
    path = tmp_path / "candidate.npz"
    save_candidate(candidate, path)
    fingerprint = _fingerprint(candidate)
    assert len(fingerprint) == 64
    assert int(fingerprint, 16) >= 0
    assert _fingerprint(path) == fingerprint


def test_metadata_mapping_order_does_not_change_identity():
    left = CandidateMetadata(
        profile=CandidateProfile.DYNAMIC, extra={"run": "r", "receipt": "s"}
    )
    right = replace(left, extra={"receipt": "s", "run": "r"})
    assert _fingerprint(_candidate(metadata=left)) == _fingerprint(
        _candidate(metadata=right)
    )
