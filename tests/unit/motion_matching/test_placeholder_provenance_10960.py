"""Unit tests for #10960 P2: fail-closed remediation of zero-hash and zero-work placeholders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.club_only.hub_accounting import (
    HubMode,
    account_external_hub_work,
)
from src.shared.python.motion_matching.loaders.body_json import load_body_target_json

pytestmark = pytest.mark.unit


def _make_body_json_payload(
    *,
    source_override: dict[str, Any] | None = None,
    schema_override: str | None = None,
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "filename": "test_body.json",
        "format": "body_target_json_v1",
        "subject_id": "test_subject",
        "trial_id": "test_trial",
    }
    if source_override is not None:
        source = dict(source_override)
    return {
        "schema": schema_override
        if schema_override is not None
        else "body_target_json_v1",
        "time_s": [0.0, 0.01],
        "marker_names": ["pelvis", "wrist", "elbow"],
        "marker_xyz": [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        "impact_idx": 0,
        "events": [{"label": "address", "frame": 0, "time_s": 0.0}],
        "source": source,
        "coordinate_frame": "z_up_right_handed",
    }


def test_body_json_without_sha256_computes_file_digest(tmp_path: Path) -> None:
    """A body JSON without sha256 gets the true file-bytes SHA-256 digest."""
    payload = _make_body_json_payload(
        source_override={
            "filename": "sample.json",
            "format": "test_format",
        }
    )
    json_path = tmp_path / "sample.json"
    raw_text = json.dumps(payload, indent=2)
    json_path.write_text(raw_text, encoding="utf-8")

    expected_sha256 = hashlib.sha256(json_path.read_bytes()).hexdigest()

    target = load_body_target_json(json_path)

    assert target.source.sha256 == expected_sha256
    assert target.source.sha256 != "0" * 64
    assert len(target.source.sha256) == 64


def test_body_json_all_zero_sha256_raises(tmp_path: Path) -> None:
    """An all-zero sha256 placeholder in the source record raises ValueError."""
    payload = _make_body_json_payload(
        source_override={
            "filename": "all_zero.json",
            "format": "test_format",
            "sha256": "0" * 64,
        }
    )
    json_path = tmp_path / "all_zero.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="sha256"):
        load_body_target_json(json_path)


def test_body_json_invalid_sha256_format_raises(tmp_path: Path) -> None:
    """Non-64-character or non-hex sha256 in source record raises ValueError."""
    for invalid_sha in ["", "not_a_hash", "0" * 63, "g" * 64]:
        payload = _make_body_json_payload(
            source_override={
                "filename": "invalid.json",
                "format": "test_format",
                "sha256": invalid_sha,
            }
        )
        json_path = tmp_path / f"invalid_{len(invalid_sha)}.json"
        json_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="sha256"):
            load_body_target_json(json_path)


def test_body_json_valid_declared_sha256_is_preserved(tmp_path: Path) -> None:
    """A valid non-zero 64-hex digest is accepted and preserved in lowercase."""
    valid_sha = "1" * 64
    payload = _make_body_json_payload(
        source_override={
            "filename": "valid.json",
            "format": "test_format",
            "sha256": valid_sha.upper(),
        }
    )
    json_path = tmp_path / "valid.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    target = load_body_target_json(json_path)
    assert target.source.sha256 == valid_sha


def test_body_json_missing_format_is_derived_from_schema(tmp_path: Path) -> None:
    """When format is missing from source, it is derived from file schema rather than defaulting to synthetic."""
    payload = _make_body_json_payload(
        source_override={
            "filename": "no_format.json",
        }
    )
    json_path = tmp_path / "no_format.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    target = load_body_target_json(json_path)

    # Derived from file schema "body_target_json_v1", never placeholder "synthetic"
    assert target.source.format == "body_target_json_v1"
    assert target.source.format != "synthetic"


def test_body_json_explicit_synthetic_format_allowed(tmp_path: Path) -> None:
    """When format is explicitly declared as synthetic, it is accepted."""
    payload = _make_body_json_payload(
        source_override={
            "filename": "explicit_synthetic.json",
            "format": "synthetic",
        }
    )
    json_path = tmp_path / "explicit_synthetic.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    target = load_body_target_json(json_path)
    assert target.source.format == "synthetic"


def test_unmeasured_hub_work_is_not_reported_as_zero() -> None:
    """Fixed-pivot external work without measurement reports None, never 0.0."""
    times = np.array([0.0, 0.01, 0.02])
    positions = np.zeros((3, 2))
    forces = np.zeros((3, 2))

    motion = account_external_hub_work(
        times=times,
        hub_positions=positions,
        hub_reaction_forces=forces,
        hub_mode=HubMode.FIXED_PIVOT,
    )

    assert motion.total_work_joules is None
    assert motion.total_work_joules != 0.0


def test_explicit_measured_hub_work_is_recorded() -> None:
    """Explicitly measured work passed for fixed pivot is recorded accurately."""
    times = np.array([0.0, 0.01, 0.02])
    positions = np.zeros((3, 2))
    forces = np.zeros((3, 2))

    motion = account_external_hub_work(
        times=times,
        hub_positions=positions,
        hub_reaction_forces=forces,
        hub_mode=HubMode.FIXED_PIVOT,
        measured_work_joules=12.5,
    )

    assert motion.total_work_joules == 12.5
