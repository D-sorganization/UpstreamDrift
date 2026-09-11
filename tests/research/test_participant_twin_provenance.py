"""Privacy and governance contracts for the participant twin cohort record."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.participant_twin_provenance import (
    HeldOutAccessError,
    assign_participant_holdout,
    assign_trajectory_roles,
    build_holdout_barrier,
    load_and_validate_cohort,
    personalized_recommendation,
    public_facade,
    validate_cohort,
)

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.scientific
COHORT = (
    ROOT / "docs/research/proximal_distal_energy_transfer/data/"
    "participant_twin_cohort.json"
)


def _record() -> dict[str, object]:
    return json.loads(COHORT.read_text(encoding="utf-8"))


def test_committed_cohort_is_identity_safe_and_human_blocked() -> None:
    record, summary = load_and_validate_cohort(COHORT)

    assert summary["data_class"] == "synthetic_benchmark"
    assert summary["human_calibration_authority"] == (
        "blocked_no_governed_participant_data"
    )
    assert summary["personalized_recommendation_authority"] == "prohibited"
    assert summary["participant_count"] >= 4
    assert summary["participant_held_out"]
    assert set(summary["participant_held_out"]).isdisjoint(
        summary["training_participants"]
    )
    assert record["split"]["frozen_before_outcome_access"] is True
    assert record["split"]["outcome_fields_present"] is False


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("participant_04.c3d", "source filename"),
        ("coach@example.test", "e-mail"),
        ("C:/lab/session", "filesystem path"),
        ("1994-03-11", "calendar date"),
    ],
)
def test_identity_bearing_values_are_rejected(value: str, message: str) -> None:
    record = _record()
    record["cohort_id"] = value

    with pytest.raises(ValueError, match=message):
        validate_cohort(record)


def test_identity_bearing_keys_are_rejected() -> None:
    record = _record()
    record["participants"][0]["source_filename"] = "a-recording"

    with pytest.raises(ValueError, match="identity-bearing field"):
        validate_cohort(record)


def test_pseudonyms_may_not_encode_a_record_index() -> None:
    record = _record()
    record["participants"][0]["participant_pseudonym"] = "pt-0000000000000001"

    with pytest.raises(ValueError, match="record index"):
        validate_cohort(record)


def test_record_order_may_not_carry_identity() -> None:
    record = _record()
    record["participants"].reverse()

    with pytest.raises(ValueError, match="order carries no identity"):
        validate_cohort(record)


def test_equipment_may_not_single_out_one_participant() -> None:
    record = _record()
    lone_club = record["club_pool"][0]
    for participant in record["participants"][1:]:
        participant["club_pseudonyms"] = sorted(
            club for club in participant["club_pseudonyms"] if club != lone_club
        ) or [record["club_pool"][1]]

    with pytest.raises(ValueError, match="shared by at least"):
        validate_cohort(record)


def test_split_must_reproduce_from_the_frozen_salt() -> None:
    record = _record()
    record["split"]["participant_held_out"] = [
        record["participants"][0]["participant_pseudonym"]
    ]

    with pytest.raises(ValueError, match="does not reproduce from the salt"):
        validate_cohort(record)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("frozen_before_outcome_access", False, "frozen before any outcome"),
        ("outcome_fields_present", True, "must not carry outcome fields"),
    ],
)
def test_split_must_predate_outcome_access(key: str, value: bool, message: str) -> None:
    record = _record()
    record["split"][key] = value

    with pytest.raises(ValueError, match=message):
        validate_cohort(record)


def test_holdout_and_role_assignment_ignore_input_order() -> None:
    record = _record()
    pseudonyms = [row["participant_pseudonym"] for row in record["participants"]]
    fraction = record["split"]["holdout_fraction"]
    salt = record["split"]["split_salt"]

    forward = assign_participant_holdout(
        pseudonyms, split_salt=salt, holdout_fraction=fraction
    )
    reversed_order = assign_participant_holdout(
        list(reversed(pseudonyms)), split_salt=salt, holdout_fraction=fraction
    )

    assert forward == reversed_order
    trajectories = record["participants"][0]["trajectory_ids"]
    assert assign_trajectory_roles(
        trajectories, split_salt=salt, calibration_trajectory_count=3
    ) == assign_trajectory_roles(
        list(reversed(trajectories)), split_salt=salt, calibration_trajectory_count=3
    )


def test_barrier_refuses_every_held_out_outcome() -> None:
    record = _record()
    barrier = build_holdout_barrier(record)
    outcomes = dict.fromkeys(record["split"]["trajectory_roles"], 1.0)
    calibration = min(barrier.calibration_trajectories)
    evaluation = min(barrier.evaluation_trajectories)

    assert barrier.calibration_outcome(calibration, outcomes) == 1.0
    with pytest.raises(HeldOutAccessError, match="held out"):
        barrier.calibration_outcome(evaluation, outcomes)
    with pytest.raises(HeldOutAccessError, match="not a registered trajectory"):
        barrier.calibration_outcome("tj-deadbeefdeadbeef", outcomes)
    assert barrier.held_out_outcomes_read() == 0
    assert barrier.access_ledger() == (calibration,)


def test_public_facade_suppresses_small_cells_and_hides_pseudonyms() -> None:
    record = _record()

    facade = public_facade(record)

    payload = json.dumps(facade)
    assert "participants" not in facade
    assert "pt-" not in payload
    assert "tj-" not in payload
    assert "cl-" not in payload
    suppressed = [
        (field, level)
        for field, levels in facade["stratum_counts"].items()
        for level, count in levels.items()
        if count == "suppressed_small_cell"
    ]
    assert suppressed
    assert facade["human_calibration_authority"] == (
        "blocked_no_governed_participant_data"
    )


def test_personalized_recommendation_is_refused_without_governed_authority() -> None:
    record = _record()
    pseudonym = record["participants"][0]["participant_pseudonym"]

    with pytest.raises(ValueError, match="outside the declared evidence domain"):
        personalized_recommendation(record, pseudonym, "increase lead-hand grip force")


def test_governed_human_class_requires_every_satisfied_gate() -> None:
    record = copy.deepcopy(_record())
    record["governance"]["data_class"] = "governed_human"
    record["governance"]["pseudonym_derivation"] = (
        "private_random_map_held_outside_repository"
    )

    with pytest.raises(ValueError, match="requires a satisfied"):
        validate_cohort(record)


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "cohort.json"
    path.write_text('{"cohort_id": "a", "cohort_id": "b"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_and_validate_cohort(path)


@pytest.mark.parametrize("name", ["declared_scope", "prohibited_inferences"])
def test_evidence_domain_must_declare_statements(name: str) -> None:
    record = _record()
    record["evidence_domain"][name] = []

    with pytest.raises(ValueError, match="nonempty list of statements"):
        validate_cohort(record)
