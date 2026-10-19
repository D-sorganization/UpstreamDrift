"""Sealed NM records still document u_dim=5; addendum supersedes (#12670)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.neural_motion.golfer_udim_addendum import (
    GOLFER_MODEL_ID,
    load_golfer_udim_addendum,
)
from src.shared.python.tour_baselines.registry import get_golf_model

pytestmark = pytest.mark.unit

_EVIDENCE_DIR = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "plans"
    / "neural_motion_matching"
    / "evidence"
)
_FROZEN_ROSTER = _EVIDENCE_DIR / "nm01_model_roster.json"
_FROZEN_MATRIX_RECEIPT = _EVIDENCE_DIR / "nm09_checkpoint_matrix_receipt.json"


def test_frozen_roster_still_records_u_dim_five_for_audit() -> None:
    payload = json.loads(_FROZEN_ROSTER.read_text(encoding="utf-8"))
    golfer = next(e for e in payload["entries"] if e["model_id"] == GOLFER_MODEL_ID)
    assert golfer["independent_dof"] == 5
    assert payload["content_digest"] == (
        "1451cfb6cdcd1d8a7e951ec7fa5296f2cc5ee286904a7bb5c062dbb14ad59c6c"
    )


def test_addendum_supersedes_sealed_dof_and_invalidates_u_dim_five_checkpoints() -> (
    None
):
    addendum = load_golfer_udim_addendum()
    identity = get_golf_model(GOLFER_MODEL_ID)
    assert identity.independent_dof == 6
    assert addendum.corrected_independent_dof == identity.independent_dof
    assert addendum.corrected_q_dim == identity.dof
    assert addendum.checkpoint_invalidation_status == "INVALIDATED"
    assert addendum.supersedes_roster_content_digest == (
        "1451cfb6cdcd1d8a7e951ec7fa5296f2cc5ee286904a7bb5c062dbb14ad59c6c"
    )
    receipt = json.loads(_FROZEN_MATRIX_RECEIPT.read_text(encoding="utf-8"))
    assert addendum.supersedes_matrix_digest == receipt["matrix_digest"]


def test_addendum_content_digest_is_self_consistent() -> None:
    path = _EVIDENCE_DIR / "nm12670_golfer_independent_dof_addendum.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    recorded = raw.pop("content_digest")
    from src.shared.python.neural_motion.golfer_udim_addendum import (
        content_digest_for_payload,
    )

    assert recorded == content_digest_for_payload(raw)
