"""Freshness and scope gates for the distributed #9153 smoke."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.research.proximal_distal_energy.articulated_forward_distributed_plan import (
    DistributedAttributionStudyPlan,
)
from scripts.research.proximal_distal_energy.articulated_forward_attribution_runner import (
    build_registered_cases,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN = (
    ROOT
    / "docs"
    / "research"
    / "proximal_distal_energy_transfer"
    / "data"
    / "articulated_forward_distributed_plan.json"
)


def test_distributed_plan_is_serial_event_explicit_and_nonhuman() -> None:
    manifest = DistributedAttributionStudyPlan(
        source_revision="a" * 40,
        source_data_sha256="b" * 64,
    ).to_manifest()

    assert manifest["execution"]["worker_count"] == 1
    assert manifest["execution"]["launch_status"] == "not_started"
    assert len(build_registered_cases(manifest)) == 42
    assert manifest["design"]["event_kinds"] == [
        "opening",
        "reattachment",
        "friction_limit_entry",
        "friction_limit_exit",
    ]
    assert (
        manifest["design"]["distributed_contact_law"]["static_stick_modeled"] is False
    )
    assert manifest["promotion"]["human_or_coaching_claims"] is False
    assert manifest["promotion"]["cross_engine_parity_required"] is True
    assert manifest["promotion"]["original_rigid_smoke_failure_erased"] is False


def test_committed_distributed_plan_matches_generator() -> None:
    committed = json.loads(PLAN.read_text(encoding="utf-8"))
    generated = DistributedAttributionStudyPlan(
        source_revision=committed["identity"]["source_revision"],
        source_data_sha256=committed["identity"]["source_data_sha256"],
    ).to_manifest()

    assert committed == generated
