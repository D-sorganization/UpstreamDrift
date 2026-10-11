"""Staleness guard for the finite-ground post-hoc sensitivity artifact.

Kept separate from ``test_articulated_ground_posthoc_sensitivity.py`` because
that file's own digest is recorded in the artifact's ``source_sha256``.
"""

from __future__ import annotations

import json

import pytest

from scripts.research.proximal_distal_energy.articulated_ground_posthoc_sensitivity import (
    DATA,
    build_ground_posthoc_sensitivity,
)

pytestmark = pytest.mark.unit


def test_committed_record_matches_regenerated_record() -> None:
    """The committed record must equal what the documented generator emits."""
    committed = json.loads(
        (DATA / "articulated_ground_posthoc_sensitivity.json").read_text(
            encoding="utf-8"
        )
    )
    assert build_ground_posthoc_sensitivity() == committed, (
        "stale artifact: rerun scripts.research.proximal_distal_energy."
        "run_articulated_ground_posthoc_sensitivity"
    )
