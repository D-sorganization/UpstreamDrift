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
    _scale_is_semantically_valid,
    build_scaffold,
)

pytestmark = pytest.mark.unit


def test_scaffold_preserves_registered_coverage_and_reviewed_overrides() -> None:
    contracts, reported = build_scaffold(ARTICLE.parents[2])
    registered_contracts = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    registered_reported = json.loads(REPORTED_PATH.read_text(encoding="utf-8"))

    scaffold_by_id = {row["claim_id"]: row for row in contracts["claims"]}
    registered_by_id = {row["claim_id"]: row for row in registered_contracts["claims"]}
    assert set(scaffold_by_id) == set(registered_by_id)
    assert {
        claim_id
        for claim_id in scaffold_by_id
        if scaffold_by_id[claim_id] != registered_by_id[claim_id]
    } == {
        "PD-CLAIM-313",
        "PD-CLAIM-314",
        "PD-CLAIM-315",
        "PD-CLAIM-316",
    }

    scaffold_reported = reported["claims"]
    reviewed_reported = registered_reported["claims"]
    assert set(scaffold_reported) - set(reviewed_reported) == {
        "PD-CLAIM-313",
        "PD-CLAIM-315",
        "PD-CLAIM-316",
    }
    assert set(reviewed_reported) <= set(scaffold_reported)
    assert all(
        reviewed_reported[claim_id] == scaffold_reported[claim_id]
        for claim_id in reviewed_reported
    )


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
