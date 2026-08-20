"""Contracts for fail-closed articulated checkpoint audits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_checkpoint_audit import (
    audit_ground_checkpoint_directory,
)

pytestmark = pytest.mark.scientific
DESIGN_DIGEST = "a" * 64


def _write_checkpoint(
    path: Path,
    *,
    state_slot: int,
    kind: str,
    branch_slot: int,
    design_digest: str = DESIGN_DIGEST,
    value: float = 1.0,
    shape: tuple[int, ...] = (2,),
    state: tuple[int, int] | None = None,
) -> None:
    metadata = {
        "state_slot": state_slot,
        "state": list(state or (state_slot, 0)),
        "kind": kind,
        "branch_slot": branch_slot,
        "schema_version": "articulated-ground-branch-checkpoint/v1",
        "design_digest": design_digest,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            __metadata__=np.asarray(json.dumps(metadata, sort_keys=True)),
            response=np.full(shape, value, dtype=float),
            gate=np.ones(shape, dtype=bool),
        )


def _write_state(
    directory: Path,
    state_slot: int = 0,
    *,
    state: tuple[int, int] | None = None,
) -> None:
    for kind, count in (("primary", 4), ("control", 2)):
        for branch_slot in range(count):
            _write_checkpoint(
                directory / f"state-{state_slot:02d}-{kind}-{branch_slot:02d}.npz",
                state_slot=state_slot,
                kind=kind,
                branch_slot=branch_slot,
                state=state,
            )


def test_checkpoint_audit_distinguishes_complete_and_partial_sets(tmp_path) -> None:
    complete = tmp_path / "complete"
    _write_state(complete)
    reference = complete / "state-00-primary-00.npz"

    report = audit_ground_checkpoint_directory(
        complete, reference=reference, expected_count=6
    )

    assert report["status"] == "complete"
    assert report["checkpoint_count"] == 6
    assert report["unique_identity_count"] == 6
    assert report["observed_state_slot_count"] == 1
    assert report["complete_state_slot_count"] == 1
    assert report["design_digest"] == DESIGN_DIGEST
    assert len(report["checkpoint_set_sha256"]) == 64

    partial = tmp_path / "partial"
    _write_checkpoint(
        partial / "state-00-primary-00.npz",
        state_slot=0,
        kind="primary",
        branch_slot=0,
    )
    partial_report = audit_ground_checkpoint_directory(
        partial, reference=reference, expected_count=6, allow_partial=True
    )
    assert partial_report["status"] == "partial"
    assert partial_report["observed_state_slot_count"] == 1
    assert partial_report["complete_state_slot_count"] == 0
    with pytest.raises(RuntimeError, match="count is incomplete"):
        audit_ground_checkpoint_directory(
            partial, reference=reference, expected_count=6
        )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ({"design_digest": "b" * 64}, "design digest"),
        ({"design_digest": "z" * 64}, "design digest"),
        ({"value": float("nan")}, "must be finite"),
        ({"shape": (3,)}, "shape or dtype"),
    ],
)
def test_checkpoint_audit_rejects_mixed_or_invalid_payloads(
    tmp_path, tamper, message
) -> None:
    directory = tmp_path / "checkpoints"
    _write_state(directory)
    reference = directory / "state-00-primary-00.npz"
    target = directory / "state-00-primary-01.npz"
    _write_checkpoint(
        target,
        state_slot=0,
        kind="primary",
        branch_slot=1,
        **tamper,
    )

    with pytest.raises(RuntimeError, match=message):
        audit_ground_checkpoint_directory(
            directory, reference=reference, expected_count=6
        )


def test_checkpoint_audit_rejects_metadata_filename_disagreement(tmp_path) -> None:
    directory = tmp_path / "checkpoints"
    _write_state(directory)
    reference = directory / "state-00-primary-00.npz"
    _write_checkpoint(
        directory / "state-00-primary-01.npz",
        state_slot=0,
        kind="primary",
        branch_slot=2,
    )

    with pytest.raises(RuntimeError, match="filename does not match"):
        audit_ground_checkpoint_directory(
            directory, reference=reference, expected_count=6
        )


def test_checkpoint_audit_rejects_duplicate_state_mapping(tmp_path) -> None:
    directory = tmp_path / "checkpoints"
    _write_state(directory, 0)
    _write_state(directory, 1, state=(0, 0))
    reference = directory / "state-00-primary-00.npz"

    with pytest.raises(RuntimeError, match="map to unique states"):
        audit_ground_checkpoint_directory(
            directory, reference=reference, expected_count=12
        )
