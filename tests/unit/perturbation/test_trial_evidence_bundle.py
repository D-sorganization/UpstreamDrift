"""Tests for atomic, digest-verifiable canonical trial bundles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python import perturbation
from src.shared.python.perturbation.trial_evidence import (
    CanonicalTrialEvidence,
    ClosestApproach,
    SampledInput,
    TrialTrace,
)
from src.shared.python.perturbation.trial_evidence_bundle import (
    TrialEvidenceBundleSummary,
    load_trial_evidence_bundle,
    validate_trial_evidence_bundle,
    write_trial_evidence_bundle,
)

pytestmark = pytest.mark.unit

_PLAN_SHA = "a" * 64
_SCENARIO_SHA = "d" * 64
_CONFIG_SHA = "e" * 64
_TOOLS_REVISION = "b" * 40
_ENGINE_REVISION = "c" * 40


def _trace(*, complete: bool = True) -> TrialTrace:
    return TrialTrace(
        times_s=np.array([0.0, 0.01, 0.02]),
        q=np.array([[0.0, 0.0], [0.1, 0.2], [0.2, 0.3]]),
        v=np.array([[0.0, 0.0], [1.0, 2.0], [1.5, 1.0]]),
        coordinate_ids=("joint.shoulder", "joint.wrist"),
        coordinate_units=("rad", "rad"),
        velocity_units=("rad/s", "rad/s"),
        markers_m=np.array(
            [
                [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                [[0.1, 0.0, 0.0], [0.6, 0.1, 0.0]],
                [[0.2, 0.0, 0.0], [0.7, 0.2, 0.0]],
            ]
        ),
        marker_ids=("body.lead_hand", "body.clubhead"),
        frame_id="world-z-up",
        alignment_id="downswing-start/v1",
        complete=complete,
    )


def _record(index: int, outcome: str) -> CanonicalTrialEvidence:
    common = {
        "trial_index": index,
        "seed": 31,
        "plan_sha256": _PLAN_SHA,
        "scenario_sha256": _SCENARIO_SHA,
        "execution_config_sha256": _CONFIG_SHA,
        "tools_revision": _TOOLS_REVISION,
        "engine_id": "mujoco-articulated",
        "engine_revision": _ENGINE_REVISION,
        "model_id": "bilateral-upper-body/v1",
        "sampled_inputs": (
            SampledInput("swing_sim.swing.damping_shoulder", 1.5 + index, "N·m·s"),
        ),
    }
    if outcome == "no_impact":
        return CanonicalTrialEvidence(
            **common,
            outcome="no_impact",
            trace=_trace(),
            closest_approach=ClosestApproach(
                0.02, 0.014, "body.clubhead", "body.ball", False
            ),
        )
    if outcome == "partial_valid_trace":
        return CanonicalTrialEvidence(
            **common,
            outcome="partial_valid_trace",
            trace=_trace(complete=False),
            failure_reason="Integrator diverged after a finite prefix.",
        )
    return CanonicalTrialEvidence(
        **common,
        outcome="numerical_failure",
        trace=None,
        failure_reason="FloatingPointError: non-finite acceleration",
    )


def test_bundle_api_is_public() -> None:
    assert perturbation.TrialEvidenceBundleSummary is TrialEvidenceBundleSummary
    assert perturbation.write_trial_evidence_bundle is write_trial_evidence_bundle
    assert perturbation.load_trial_evidence_bundle is load_trial_evidence_bundle


def test_round_trip_preserves_complete_miss_partial_trace_and_failure(
    tmp_path: Path,
) -> None:
    records = (
        _record(0, "no_impact"),
        _record(1, "partial_valid_trace"),
        _record(2, "numerical_failure"),
    )
    destination = tmp_path / "campaign"

    summary = write_trial_evidence_bundle(destination, records)
    loaded = load_trial_evidence_bundle(destination)
    validated = validate_trial_evidence_bundle(destination)

    assert summary == validated
    assert summary.trial_count == 3
    assert len(summary.content_sha256) == 64
    assert tuple(record.outcome for record in loaded) == (
        "no_impact",
        "partial_valid_trace",
        "numerical_failure",
    )
    assert loaded[0].trace is not None
    np.testing.assert_array_equal(loaded[0].trace.q, records[0].trace.q)
    assert loaded[0].trace.q.flags.writeable is False
    assert loaded[1].trace is not None and loaded[1].trace.complete is False
    assert loaded[2].trace is None

    manifest = json.loads((destination / "manifest.json").read_text("utf-8"))
    assert manifest["schema_version"] == "upstream-tools-variation-bundle/v1"
    assert manifest["identity"]["plan_sha256"] == _PLAN_SHA
    assert manifest["outcome_counts"] == {
        "hit": 0,
        "no_impact": 1,
        "numerical_failure": 1,
        "partial_valid_trace": 1,
    }


def test_content_digest_is_reproducible_across_output_locations(
    tmp_path: Path,
) -> None:
    records = (_record(0, "no_impact"), _record(1, "numerical_failure"))

    first = write_trial_evidence_bundle(tmp_path / "first", records)
    second = write_trial_evidence_bundle(tmp_path / "second", records)

    assert first.content_sha256 == second.content_sha256
    assert (tmp_path / "first" / "manifest.json").read_bytes() == (
        tmp_path / "second" / "manifest.json"
    ).read_bytes()


def test_loader_rejects_tampered_missing_or_extra_content(tmp_path: Path) -> None:
    records = (_record(0, "no_impact"),)

    tampered = tmp_path / "tampered"
    write_trial_evidence_bundle(tampered, records)
    (tampered / "arrays" / "trial-000000-q.npy").write_bytes(b"altered")
    with pytest.raises(ValueError, match="digest"):
        load_trial_evidence_bundle(tampered)

    missing = tmp_path / "missing"
    write_trial_evidence_bundle(missing, records)
    (missing / "arrays" / "trial-000000-v.npy").unlink()
    with pytest.raises(ValueError, match="file inventory"):
        load_trial_evidence_bundle(missing)

    extra = tmp_path / "extra"
    write_trial_evidence_bundle(extra, records)
    (extra / "unregistered.txt").write_text("not evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="file inventory"):
        load_trial_evidence_bundle(extra)


def test_writer_rejects_identity_drift_and_never_overwrites(tmp_path: Path) -> None:
    drifted = CanonicalTrialEvidence(
        **{
            **_record(1, "numerical_failure").__dict__,
            "scenario_sha256": "f" * 64,
        }
    )
    destination = tmp_path / "drift"
    with pytest.raises(ValueError, match="execution identity"):
        write_trial_evidence_bundle(destination, (_record(0, "no_impact"), drifted))
    assert not destination.exists()

    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "retain.txt"
    sentinel.write_text("retain", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        write_trial_evidence_bundle(existing, (_record(0, "no_impact"),))
    assert sentinel.read_text("utf-8") == "retain"
