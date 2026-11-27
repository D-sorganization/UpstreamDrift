"""A1 reference fixture pins capture-A and rejects a drifted best-match bundle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.a1_reference_fixture import (
    A1FixtureError,
    BUNDLE_SCHEMA,
    verify_best_match_bundle,
    verify_public_pins,
)
from src.shared.python.contracts import PreconditionError

pytestmark = pytest.mark.unit


def _bundle(path: Path, *, schema: str, dt_s: float, spec_sha: str = "ab" * 32) -> None:
    """Write a same-input bundle with the given manifest fields to ``path``."""
    manifest = {
        "schema": schema,
        "spec_sha256": spec_sha,
        "dt_s": dt_s,
        "coordinate_order": ["q0"],
    }
    np.savez_compressed(
        path,
        manifest=np.array(json.dumps(manifest)),
        reference_q=np.zeros((4, 2)),
    )


def test_public_pins_match_capture_a_and_canonical_v2() -> None:
    """The committed fixture pins capture-A, the marker schema, and canonical-v2."""
    fixture = verify_public_pins()
    assert fixture["schema"] == "opensim-catchup-a1/v1"
    assert fixture["capture"]["id"] == "capture-A"
    assert fixture["capture"]["frame_count"] == 654
    assert fixture["capture"]["sample_rate_hz"] == 360.0
    assert fixture["markers"]["count"] == 38
    assert fixture["model"]["convention"] == "canonical-v2"
    assert fixture["model"]["quaternion_order"] == "wxyz"
    assert fixture["reference_trajectory"]["in_repo_demo_is_best_match"] is False
    assert (
        fixture["reference_trajectory"]["best_match_env"] == "UD_DEFAULT_MATCH_BUNDLE"
    )


def test_best_match_bundle_on_capture_timebase(tmp_path: Path) -> None:
    """A bundle on the capture timebase is accepted and its receipt is typed."""
    path = tmp_path / "best.npz"
    _bundle(path, schema=BUNDLE_SCHEMA, dt_s=1.0 / 360.0)
    receipt = verify_best_match_bundle(path)
    assert receipt["schema"] == BUNDLE_SCHEMA
    assert receipt["reference_steps"] == 4
    assert receipt["spec_sha256"] == "ab" * 32


def test_best_match_bundle_rejects_other_schema(tmp_path: Path) -> None:
    """A bundle with a foreign schema is rejected."""
    path = tmp_path / "other.npz"
    _bundle(path, schema="same-input-bundle/v0", dt_s=1.0 / 360.0)
    with pytest.raises((A1FixtureError, PreconditionError)):
        verify_best_match_bundle(path)


def test_best_match_bundle_rejects_other_timebase(tmp_path: Path) -> None:
    """A bundle on a different timebase is rejected."""
    path = tmp_path / "slow.npz"
    _bundle(path, schema=BUNDLE_SCHEMA, dt_s=1.0 / 240.0)
    with pytest.raises((A1FixtureError, PreconditionError)):
        verify_best_match_bundle(path)
