"""Semantic guards for numeric-contract scaffolding (#8918)."""

from __future__ import annotations

import json

import pytest

from scripts.research.proximal_distal_energy.scaffold_numeric_claim_contracts import (
    ARTICLE,
    CONTRACT_PATH,
    REPORTED_PATH,
    _has_semantic_pointer_match,
    _pointer_matches_declared_quantity,
    _retain_reviewed_entry,
    _scale_is_semantically_valid,
    build_scaffold,
)


pytestmark = pytest.mark.unit


def test_scaffold_reproduces_registered_documents() -> None:
    contracts, reported = build_scaffold(ARTICLE.parents[2])

    assert contracts == json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert reported == json.loads(REPORTED_PATH.read_text(encoding="utf-8"))


def test_reviewed_reported_entry_requires_exact_statement_and_reindexes() -> None:
    claim = {"claim_id": "PD-CLAIM-TEST"}
    literal = {"literal_id": "5#1"}
    entry = {
        "literal_id": "5#1",
        "artifact": str(REPORTED_PATH),
        "json_pointer": "/claims/PD-CLAIM-TEST/4/value",
    }
    record = {"literal_id": "5#1", "value": 5.0}
    authority = {("PD-CLAIM-TEST", "reviewed-digest", "5#1"): (entry, record)}
    reported: dict[str, list[dict[str, object]]] = {}

    assert (
        _retain_reviewed_entry(
            claim=claim,
            statement_sha256="changed-digest",
            literal=literal,
            authority=authority,
            reported=reported,
        )
        is None
    )
    retained = _retain_reviewed_entry(
        claim=claim,
        statement_sha256="reviewed-digest",
        literal=literal,
        authority=authority,
        reported=reported,
    )

    assert retained is not None
    assert retained["json_pointer"] == "/claims/PD-CLAIM-TEST/0/value"
    assert reported == {"PD-CLAIM-TEST": [record]}


def test_delivery_event_context_matches_time_pointer() -> None:
    assert _has_semantic_pointer_match(
        "/impact/time_s", "reaches the declared delivery event at 0.3493 s"
    )


def test_generic_grid_context_does_not_authorize_unrelated_time_pointer() -> None:
    assert not _has_semantic_pointer_match(
        "/robustness_grid/rows/1/torque_cut_time_s",
        "The 120-case grid spans several outcomes",
    )


def test_millisecond_transform_requires_time_quantity() -> None:
    assert _scale_is_semantically_valid(
        1000.0, "/negative_interval_duration_s", "lasts 37.5 ms"
    )
    assert not _scale_is_semantically_valid(
        1000.0, "/hand_mass_factor", "through 50 ms"
    )


def test_radian_transform_does_not_match_unrelated_count() -> None:
    import math

    assert not _scale_is_semantically_valid(
        180.0 / math.pi, "/terminal_q_distance_rad", "hashes 13 source inputs"
    )


def test_quantity_guard_rejects_equal_but_unrelated_values() -> None:
    assert not _pointer_matches_declared_quantity(
        "/rows/65/force_work_difference_j", before="whereas ", after=" ms versus"
    )
    assert not _pointer_matches_declared_quantity(
        "/programs/126/program_index", before="contains ", after=" rate cases"
    )
    assert not _pointer_matches_declared_quantity(
        "/configuration/station_count_per_hand", before="within ", after="% for peak"
    )
    assert not _pointer_matches_declared_quantity(
        "/robustness_grid/damping_values_nms_rad/0",
        before="phi2=dphi2=",
        after=", while",
    )


def test_quantity_guard_accepts_matching_units_and_counts() -> None:
    assert _pointer_matches_declared_quantity(
        "/impact/clubhead_speed_m_s", before="at ", after=" m/s"
    )
    assert _pointer_matches_declared_quantity(
        "/design/trajectory_count", before="all ", after=" trajectories"
    )
