"""MS-60 (#10347): Simscape returned-102 candidate conversion and run-manifest schema."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.candidate import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateProfile,
)
from src.shared.python.motion_matching.candidate_convert import (
    convert_simscape_returned_replay,
)
from src.shared.python.motion_matching.candidate_io import (
    load_candidate,
    save_candidate,
)
from src.shared.python.motion_matching.simscape_run_manifest import (
    RUN_MANIFEST_SCHEMA_VERSION,
    SimscapeRunManifestRequest,
    build_simscape_run_manifest,
    validate_simscape_run_manifest,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_DIR = (
    REPO_ROOT
    / "docs"
    / "development"
    / "simscape_tour_matching"
    / "native_evidence"
    / "two_window_fit_9967_102"
)
RETURNED_REPLAY_NPZ = EVIDENCE_DIR / "returned-replay.npz"
RETURNED_CANDIDATE_JSON = EVIDENCE_DIR / "returned-candidate.json"
QUALIFIED_REPLAY_JSON = EVIDENCE_DIR / "qualified_candidate_replay.json"
RECEIPT_JSON = EVIDENCE_DIR / "receipt.json"
COMMITTED_CANDIDATE_NPZ = EVIDENCE_DIR / "candidate.npz"
COMMITTED_MANIFEST = EVIDENCE_DIR / "run_manifest.json"
COMMITTED_PLAYBACK_GIF = EVIDENCE_DIR / "playback.gif"


def test_convert_simscape_returned_replay_lossless() -> None:
    """Converter must map run-102 returned-replay.npz into MatchedSwingCandidate."""
    assert RETURNED_REPLAY_NPZ.is_file(), f"Missing fixture: {RETURNED_REPLAY_NPZ}"
    assert RETURNED_CANDIDATE_JSON.is_file(), f"Missing: {RETURNED_CANDIDATE_JSON}"

    candidate_doc = json.loads(RETURNED_CANDIDATE_JSON.read_text(encoding="utf-8"))
    candidate = convert_simscape_returned_replay(
        RETURNED_REPLAY_NPZ,
        candidate_doc=candidate_doc,
        engine="simscape",
    )

    assert len(candidate.time_s) == 307
    assert candidate.metadata.schema_version == CANDIDATE_SCHEMA_VERSION
    assert candidate.metadata.profile == CandidateProfile.KINEMATIC
    assert candidate.metadata.engine == "simscape"
    assert candidate.q.shape == (307, 27)
    assert candidate.v is not None and candidate.v.shape == (307, 27)
    assert candidate.tau is None
    assert "tau" in candidate.metadata.missing_fields
    assert candidate.metadata.coordinate_names == tuple(
        candidate_doc["coordinate_names"]
    )
    assert candidate.metadata.marker_names == tuple(candidate_doc["marker_labels"])
    assert candidate.metadata.model_sha256 == candidate_doc["model_sha256"]
    assert candidate.model_markers_m is not None
    assert candidate.model_markers_m.shape == (307, 25, 3)
    assert candidate.target_markers_m is not None
    assert candidate.target_markers_m.shape == (307, 25, 3)
    assert candidate.marker_validity is not None
    assert candidate.marker_validity.shape == (307, 25)

    with np.load(RETURNED_REPLAY_NPZ, allow_pickle=False) as raw:
        np.testing.assert_allclose(candidate.time_s, raw["time_s"])
        np.testing.assert_allclose(candidate.model_markers_m, raw["markers_m"])
        np.testing.assert_allclose(candidate.target_markers_m, raw["target_m"])


def test_convert_simscape_roundtrip_candidate_io(tmp_path: Path) -> None:
    """Converted run-102 package must round-trip through candidate_io."""
    candidate_doc = json.loads(RETURNED_CANDIDATE_JSON.read_text(encoding="utf-8"))
    candidate = convert_simscape_returned_replay(
        RETURNED_REPLAY_NPZ,
        candidate_doc=candidate_doc,
        engine="simscape",
    )
    out = tmp_path / "candidate.npz"
    save_candidate(candidate, out)
    loaded = load_candidate(out)
    assert loaded.metadata.engine == "simscape"
    assert loaded.q.shape == candidate.q.shape
    np.testing.assert_allclose(loaded.time_s, candidate.time_s)


def test_simscape_run_manifest_schema_validation() -> None:
    """Run manifest must require R2025b identity, SHAs, host, and wall-clock."""
    candidate_doc = json.loads(RETURNED_CANDIDATE_JSON.read_text(encoding="utf-8"))
    qualified = json.loads(QUALIFIED_REPLAY_JSON.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_JSON.read_text(encoding="utf-8"))
    replay_sha = "efec8a6ecf9479c1bf0f5604d21add5224373423f1441c6250c1c3cd484e25b5"
    manifest = build_simscape_run_manifest(
        SimscapeRunManifestRequest(
            run_id="two_window_fit_9967_102",
            matlab_release=qualified["matlab_release"],
            matlab_version=qualified["matlab_version"],
            host="DeskComputer",
            model_sha256=candidate_doc["model_sha256"],
            candidate_sha256=str(receipt["returned_sha256"]),
            replay_npz_sha256=replay_sha,
            wall_clock_s=float(qualified["elapsed_s"]),
            qualification=qualified["qualification"],
            evidence_dir=(
                "docs/development/simscape_tour_matching/native_evidence/"
                "two_window_fit_9967_102"
            ),
            artifacts={
                "candidate_npz": "candidate.npz",
                "playback_gif": "playback.gif",
                "returned_replay_npz": "returned-replay.npz",
                "qualified_replay_json": "qualified_candidate_replay.json",
            },
            issue="#10347",
        )
    )
    assert manifest["schema_version"] == RUN_MANIFEST_SCHEMA_VERSION
    assert manifest["matlab_release"] == "2025b"
    validate_simscape_run_manifest(manifest)

    bad = dict(manifest)
    bad["matlab_release"] = "2026a"
    with pytest.raises(ValueError, match="R2025b"):
        validate_simscape_run_manifest(bad)


def test_committed_run102_artifacts_present_and_consistent() -> None:
    """Acceptance: candidate.npz, run_manifest.json, and playback.gif are in-tree."""
    assert COMMITTED_CANDIDATE_NPZ.is_file(), "Missing committed candidate.npz"
    assert COMMITTED_MANIFEST.is_file(), "Missing committed run_manifest.json"
    assert COMMITTED_PLAYBACK_GIF.is_file(), "Missing committed playback.gif"
    assert COMMITTED_PLAYBACK_GIF.stat().st_size > 0

    loaded = load_candidate(COMMITTED_CANDIDATE_NPZ)
    assert loaded.metadata.engine == "simscape"
    assert len(loaded.time_s) == 307
    assert loaded.q.shape == (307, 27)

    manifest = json.loads(COMMITTED_MANIFEST.read_text(encoding="utf-8"))
    validate_simscape_run_manifest(manifest)
    assert manifest["run_id"] == "two_window_fit_9967_102"
    assert manifest["matlab_release"] == "2025b"
    assert manifest["model_sha256"] == loaded.metadata.model_sha256
