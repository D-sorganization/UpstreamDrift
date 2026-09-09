"""Identity-safe participant, session, and club provenance for digital twins.

The record frozen by this module is the only admissible cohort authority for
participant-calibrated twin calibration. It carries pseudonyms, coarse
stratification levels, a governance block, and a split that is frozen before
any outcome is read. It deliberately cannot carry a name, a filename, a device
serial, a session date, or any other value from which a participant could be
re-identified, and it refuses records whose ordering or equipment assignment
would leak identity.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "participant-twin-provenance/v1"

_TOP_KEYS = frozenset(
    {
        "schema_version",
        "cohort_id",
        "registered_at_utc",
        "governance",
        "club_pool",
        "participants",
        "split",
        "stratification_vocabulary",
        "evidence_domain",
        "inference_boundary",
    }
)
_GOVERNANCE_KEYS = frozenset(
    {
        "data_class",
        "ethics_reference",
        "consent_and_reuse_basis",
        "private_data_authority",
        "calibration_records",
        "time_synchronization_records",
        "analysis_release_authorization",
        "identity_policy",
        "pseudonym_derivation",
    }
)
_PARTICIPANT_KEYS = frozenset(
    {
        "participant_pseudonym",
        "session_pseudonyms",
        "club_pseudonyms",
        "strata",
        "trajectory_ids",
        "holdout_role",
    }
)
_SPLIT_KEYS = frozenset(
    {
        "split_id",
        "split_salt",
        "holdout_fraction",
        "calibration_trajectory_count",
        "frozen_before_outcome_access",
        "outcome_fields_present",
        "participant_held_out",
        "trajectory_roles",
    }
)
_EVIDENCE_DOMAIN_KEYS = frozenset({"declared_scope", "prohibited_inferences"})

_DATA_CLASSES = frozenset({"synthetic_benchmark", "governed_human"})
_GATE_STATUSES = frozenset(
    {"satisfied", "not_satisfied", "not_applicable_synthetic_benchmark"}
)
_GATE_FIELDS = (
    "ethics_reference",
    "consent_and_reuse_basis",
    "private_data_authority",
    "calibration_records",
    "time_synchronization_records",
    "analysis_release_authorization",
)
_IDENTITY_POLICY = "pseudonym_only_no_identity_inference"
_PSEUDONYM_DERIVATIONS = {
    "synthetic_benchmark": "synthetic_salted_digest_no_real_identity",
    "governed_human": "private_random_map_held_outside_repository",
}

STRATIFICATION_VOCABULARY: dict[str, tuple[str, ...]] = {
    "anthropometry_band": ("band_a", "band_b", "band_c", "withheld"),
    "skill_band": ("recreational", "competitive_amateur", "elite", "withheld"),
    "sex": ("female", "male", "other_or_undisclosed", "withheld"),
    "age_band": ("18_29", "30_44", "45_59", "60_plus", "withheld"),
    "handedness": ("left", "right", "withheld"),
    "injury_history": ("none_reported", "prior_upper_limb", "prior_lumbar", "withheld"),
    "impairment": (
        "none_reported",
        "limited_shoulder_range",
        "limited_lumbar_range",
        "withheld",
    ),
    "club_class": ("driver", "iron", "wedge", "putter", "withheld"),
    "task": ("full_swing_driver", "full_swing_iron", "half_swing", "withheld"),
}

MIN_PUBLIC_CELL_COUNT = 3
MIN_CLUB_SHARING_PARTICIPANTS = 2

_PSEUDONYM = re.compile(r"(pt|ss|cl|tj)-[0-9a-f]{16}")
_HEX_SALT = re.compile(r"[0-9a-f]{32}")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_FILENAME = re.compile(
    r"\.(c3d|trc|tsv|mot|sto|csv|npz|mp4|mov|avi|xlsx|jpg|png)\b", re.IGNORECASE
)
_PATH_LIKE = re.compile(r"(^[A-Za-z]:[\\/])|(^/[A-Za-z])|(\\\\)")
_CALENDAR_DATE = re.compile(r"(19|20)\d{2}-\d{2}-\d{2}")

_BANNED_KEYS = frozenset(
    {
        "address",
        "birth_date",
        "capture_date",
        "club_serial",
        "date_of_birth",
        "device_serial",
        "email",
        "family_name",
        "file_path",
        "filename",
        "given_name",
        "initials",
        "jersey",
        "membership_id",
        "mrn",
        "name",
        "national_id",
        "path",
        "phone",
        "photo",
        "serial_number",
        "session_date",
        "source_filename",
        "video_url",
    }
)
_DATE_EXEMPT_KEYS = frozenset({"registered_at_utc"})


class HeldOutAccessError(RuntimeError):
    """Raised when calibration attempts to read a held-out outcome."""


def _require_exact_keys(record: Any, expected: frozenset[str], name: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{name} must be an object")
    if set(record) != set(expected):
        raise ValueError(f"{name} fields do not match the registered schema")


def _digest_rank(salt: str, kind: str, token: str) -> str:
    payload = f"{salt}|{kind}|{token}".encode()
    return hashlib.sha256(payload).hexdigest()


def scan_for_identity_leakage(node: Any, *, path: str = "record") -> None:
    """Reject any key or value from which a participant could be identified.

    Preconditions: ``node`` is a JSON-compatible structure.
    Postcondition: returns ``None`` only when no banned key, e-mail address,
    source filename, filesystem path, or calendar date is reachable anywhere
    inside ``node``.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            if key.lower() in _BANNED_KEYS:
                raise ValueError(f"{path}.{key} is an identity-bearing field")
            scan_for_identity_leakage(value, path=f"{path}.{key}")
        return
    if isinstance(node, list):
        for index, value in enumerate(node):
            scan_for_identity_leakage(value, path=f"{path}[{index}]")
        return
    if not isinstance(node, str):
        return
    if _EMAIL.search(node):
        raise ValueError(f"{path} contains an e-mail address")
    if _FILENAME.search(node):
        raise ValueError(f"{path} contains a source filename")
    if _PATH_LIKE.search(node):
        raise ValueError(f"{path} contains a filesystem path")
    leaf = path.rsplit(".", 1)[-1]
    if leaf not in _DATE_EXEMPT_KEYS and _CALENDAR_DATE.search(node):
        raise ValueError(f"{path} contains a calendar date")


def _validate_pseudonym(value: object, prefix: str, name: str) -> str:
    if not isinstance(value, str) or _PSEUDONYM.fullmatch(value) is None:
        raise ValueError(f"{name} must be a registered pseudonym")
    if not value.startswith(f"{prefix}-"):
        raise ValueError(f"{name} must use the {prefix} pseudonym namespace")
    suffix = value.split("-", 1)[1]
    if suffix.lstrip("0").isdigit() or set(suffix) == {"0"}:
        raise ValueError(f"{name} must not encode a record index")
    return value


def _validate_pseudonym_list(
    values: object, prefix: str, name: str, *, minimum: int
) -> tuple[str, ...]:
    if not isinstance(values, list) or len(values) < minimum:
        raise ValueError(f"{name} must list at least {minimum} pseudonyms")
    checked = tuple(
        _validate_pseudonym(value, prefix, f"{name} entry") for value in values
    )
    if len(set(checked)) != len(checked):
        raise ValueError(f"{name} must not repeat a pseudonym")
    if list(checked) != sorted(checked):
        raise ValueError(f"{name} must be sorted so record order carries no identity")
    return checked


def _validate_strata(strata: object) -> dict[str, str]:
    _require_exact_keys(strata, frozenset(STRATIFICATION_VOCABULARY), "strata")
    assert isinstance(strata, dict)
    for field_name, levels in STRATIFICATION_VOCABULARY.items():
        if strata[field_name] not in levels:
            raise ValueError(f"strata.{field_name} is not a registered level")
    return dict(strata)


def _validate_governance(governance: object) -> dict[str, str]:
    _require_exact_keys(governance, _GOVERNANCE_KEYS, "governance")
    assert isinstance(governance, dict)
    data_class = governance["data_class"]
    if data_class not in _DATA_CLASSES:
        raise ValueError("governance.data_class is not registered")
    for gate in _GATE_FIELDS:
        if governance[gate] not in _GATE_STATUSES:
            raise ValueError(f"governance.{gate} is not a registered status")
        if data_class == "governed_human" and governance[gate] != "satisfied":
            raise ValueError(f"governed human data requires a satisfied {gate}")
    if governance["identity_policy"] != _IDENTITY_POLICY:
        raise ValueError("governance.identity_policy is not the registered policy")
    if governance["pseudonym_derivation"] != _PSEUDONYM_DERIVATIONS[data_class]:
        raise ValueError("governance.pseudonym_derivation does not match data_class")
    return dict(governance)


def assign_participant_holdout(
    participant_pseudonyms: list[str] | tuple[str, ...],
    *,
    split_salt: str,
    holdout_fraction: float,
) -> tuple[str, ...]:
    """Return the participant-held-out set implied by the salt, not by order.

    Preconditions: at least two unique pseudonyms and a ``holdout_fraction``
    strictly inside ``(0, 1)``.
    Postcondition: the returned set is sorted, non-empty, and always leaves at
    least one training participant.
    """
    pseudonyms = tuple(participant_pseudonyms)
    if len(pseudonyms) < 2:
        raise ValueError("a participant holdout needs at least two participants")
    if len(set(pseudonyms)) != len(pseudonyms):
        raise ValueError("participant pseudonyms must be unique")
    if not 0.0 < float(holdout_fraction) < 1.0:
        raise ValueError("holdout_fraction must lie strictly between 0 and 1")
    count = max(1, min(len(pseudonyms) - 1, round(len(pseudonyms) * holdout_fraction)))
    ranked = sorted(
        pseudonyms, key=lambda token: _digest_rank(split_salt, "participant", token)
    )
    return tuple(sorted(ranked[:count]))


def assign_trajectory_roles(
    trajectory_ids: list[str] | tuple[str, ...],
    *,
    split_salt: str,
    calibration_trajectory_count: int,
) -> dict[str, str]:
    """Return the calibration or evaluation role of every trajectory.

    Roles are ranked by a salted digest, so neither file order nor acquisition
    order can decide which trajectories a twin may calibrate on.
    """
    identifiers = tuple(trajectory_ids)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("trajectory identifiers must be unique")
    if calibration_trajectory_count < 1:
        raise ValueError("calibration_trajectory_count must be positive")
    if len(identifiers) <= calibration_trajectory_count:
        raise ValueError("every participant needs at least one evaluation trajectory")
    ranked = sorted(
        identifiers, key=lambda token: _digest_rank(split_salt, "trajectory", token)
    )
    calibration = set(ranked[:calibration_trajectory_count])
    return {
        identifier: ("calibration" if identifier in calibration else "evaluation")
        for identifier in sorted(identifiers)
    }


@dataclass(frozen=True, slots=True)
class HoldoutBarrier:
    """Outcome gate that fails closed on every held-out trajectory."""

    calibration_trajectories: frozenset[str]
    evaluation_trajectories: frozenset[str]
    _access_ledger: set[str] = field(default_factory=set)

    def calibration_outcome(self, trajectory_id: str, outcomes: dict[str, Any]) -> Any:
        """Return one calibration outcome and record that the access happened.

        Raises ``HeldOutAccessError`` for any evaluation or unregistered
        trajectory, so a calibration path cannot read a held-out outcome even
        by accident.
        """
        if trajectory_id in self.evaluation_trajectories:
            raise HeldOutAccessError(
                f"{trajectory_id} is held out; calibration cannot read its outcome"
            )
        if trajectory_id not in self.calibration_trajectories:
            raise HeldOutAccessError(f"{trajectory_id} is not a registered trajectory")
        self._access_ledger.add(trajectory_id)
        return outcomes[trajectory_id]

    def access_ledger(self) -> tuple[str, ...]:
        """Return every trajectory whose outcome calibration actually read."""
        return tuple(sorted(self._access_ledger))

    def held_out_outcomes_read(self) -> int:
        """Return how many held-out outcomes were read, which is always zero."""
        return len(self._access_ledger & self.evaluation_trajectories)


def build_holdout_barrier(record: dict[str, Any]) -> HoldoutBarrier:
    """Build the outcome barrier implied by a validated cohort record."""
    roles = record["split"]["trajectory_roles"]
    calibration = {key for key, role in roles.items() if role == "calibration"}
    evaluation = {key for key, role in roles.items() if role == "evaluation"}
    return HoldoutBarrier(frozenset(calibration), frozenset(evaluation))


def _validate_clubs(participants: list[dict[str, Any]], club_pool: object) -> None:
    if not isinstance(club_pool, list) or not club_pool:
        raise ValueError("club_pool must be a nonempty list")
    pool = _validate_pseudonym_list(club_pool, "cl", "club_pool", minimum=1)
    usage: dict[str, int] = dict.fromkeys(pool, 0)
    for participant in participants:
        for club in participant["club_pseudonyms"]:
            if club not in usage:
                raise ValueError("club_pseudonyms must be drawn from club_pool")
            usage[club] += 1
    singleton = sorted(
        club
        for club, count in usage.items()
        if 0 < count < MIN_CLUB_SHARING_PARTICIPANTS
    )
    if singleton:
        raise ValueError(
            "every used club pseudonym must be shared by at least "
            f"{MIN_CLUB_SHARING_PARTICIPANTS} participants; {singleton[0]} is not"
        )


def _validate_participants(
    participants: object, club_pool: object
) -> list[dict[str, Any]]:
    if not isinstance(participants, list) or len(participants) < 4:
        raise ValueError("participants must list at least four records")
    checked: list[dict[str, Any]] = []
    for participant in participants:
        _require_exact_keys(participant, _PARTICIPANT_KEYS, "participant")
        assert isinstance(participant, dict)
        _validate_pseudonym(
            participant["participant_pseudonym"], "pt", "participant_pseudonym"
        )
        _validate_pseudonym_list(
            participant["session_pseudonyms"], "ss", "session_pseudonyms", minimum=1
        )
        _validate_pseudonym_list(
            participant["club_pseudonyms"], "cl", "club_pseudonyms", minimum=1
        )
        _validate_strata(participant["strata"])
        _validate_pseudonym_list(
            participant["trajectory_ids"], "tj", "trajectory_ids", minimum=2
        )
        if participant["holdout_role"] not in {"training", "participant_held_out"}:
            raise ValueError("holdout_role is not registered")
        checked.append(participant)
    pseudonyms = [row["participant_pseudonym"] for row in checked]
    if len(set(pseudonyms)) != len(pseudonyms):
        raise ValueError("participant pseudonyms must be unique")
    if pseudonyms != sorted(pseudonyms):
        raise ValueError("participants must be sorted so order carries no identity")
    _validate_clubs(checked, club_pool)
    trajectories = [
        row for participant in checked for row in participant["trajectory_ids"]
    ]
    if len(set(trajectories)) != len(trajectories):
        raise ValueError("trajectory identifiers must be unique across the cohort")
    return checked


def _validate_split(
    split: object, participants: list[dict[str, Any]]
) -> dict[str, Any]:
    _require_exact_keys(split, _SPLIT_KEYS, "split")
    assert isinstance(split, dict)
    salt = split["split_salt"]
    if not isinstance(salt, str) or _HEX_SALT.fullmatch(salt) is None:
        raise ValueError("split_salt must be a 32-character lowercase hexadecimal salt")
    if split["frozen_before_outcome_access"] is not True:
        raise ValueError("the split must be frozen before any outcome is accessed")
    if split["outcome_fields_present"] is not False:
        raise ValueError("a cohort record must not carry outcome fields")
    count = split["calibration_trajectory_count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("calibration_trajectory_count must be a positive integer")
    pseudonyms = [row["participant_pseudonym"] for row in participants]
    expected_holdout = assign_participant_holdout(
        pseudonyms,
        split_salt=salt,
        holdout_fraction=float(split["holdout_fraction"]),
    )
    if tuple(split["participant_held_out"]) != expected_holdout:
        raise ValueError("participant_held_out does not reproduce from the salt")
    expected_roles: dict[str, str] = {}
    for participant in participants:
        implied = (
            "participant_held_out"
            if participant["participant_pseudonym"] in expected_holdout
            else "training"
        )
        if participant["holdout_role"] != implied:
            raise ValueError("holdout_role does not reproduce from the frozen split")
        expected_roles.update(
            assign_trajectory_roles(
                participant["trajectory_ids"],
                split_salt=salt,
                calibration_trajectory_count=count,
            )
        )
    if split["trajectory_roles"] != expected_roles:
        raise ValueError("trajectory_roles does not reproduce from the salt")
    return dict(split)


def validate_cohort(record: dict[str, Any]) -> dict[str, Any]:
    """Validate one cohort record and return a recomputed, derived summary.

    Postcondition: every summary field is derived from the record rather than
    copied from it, and the human-calibration authority is reported as blocked
    unless the governance block declares satisfied governed human data.
    """
    _require_exact_keys(record, _TOP_KEYS, "cohort")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("cohort schema_version is unsupported")
    if not isinstance(record["cohort_id"], str) or not record["cohort_id"].strip():
        raise ValueError("cohort_id must be a nonempty string")
    boundary = record["inference_boundary"]
    if not isinstance(boundary, str) or "cannot" not in boundary.lower():
        raise ValueError("inference_boundary must state what this cohort cannot prove")
    scan_for_identity_leakage(record)
    governance = _validate_governance(record["governance"])
    if record["stratification_vocabulary"] != {
        key: list(values) for key, values in STRATIFICATION_VOCABULARY.items()
    }:
        raise ValueError("stratification_vocabulary does not match the registered set")
    _require_exact_keys(
        record["evidence_domain"], _EVIDENCE_DOMAIN_KEYS, "evidence_domain"
    )
    for name in ("declared_scope", "prohibited_inferences"):
        entries = record["evidence_domain"][name]
        if (
            not isinstance(entries, list)
            or not entries
            or any(not isinstance(row, str) or not row.strip() for row in entries)
        ):
            raise ValueError(
                f"evidence_domain.{name} must be a nonempty list of statements"
            )
    participants = _validate_participants(record["participants"], record["club_pool"])
    split = _validate_split(record["split"], participants)
    held_out = tuple(split["participant_held_out"])
    return {
        "cohort_id": record["cohort_id"],
        "data_class": governance["data_class"],
        "participant_count": len(participants),
        "trajectory_count": len(split["trajectory_roles"]),
        "training_participants": tuple(
            row["participant_pseudonym"]
            for row in participants
            if row["participant_pseudonym"] not in held_out
        ),
        "participant_held_out": held_out,
        "calibration_trajectory_count": split["calibration_trajectory_count"],
        "human_calibration_authority": (
            "available"
            if governance["data_class"] == "governed_human"
            else "blocked_no_governed_participant_data"
        ),
        "personalized_recommendation_authority": "prohibited",
    }


def load_and_validate_cohort(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a cohort JSON without accepting duplicate keys and validate it."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    record = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    return record, validate_cohort(record)


def public_facade(
    record: dict[str, Any], *, min_cell_count: int = MIN_PUBLIC_CELL_COUNT
) -> dict[str, Any]:
    """Return the public facade of a validated cohort.

    The facade exposes counts only. It carries no pseudonym, no trajectory
    identifier, and no stratum cell smaller than ``min_cell_count``, so a
    reader cannot single out a participant from the published record.
    """
    summary = validate_cohort(record)
    strata: dict[str, dict[str, Any]] = {}
    for field_name in STRATIFICATION_VOCABULARY:
        counts: dict[str, int] = {}
        for participant in record["participants"]:
            level = participant["strata"][field_name]
            counts[level] = counts.get(level, 0) + 1
        strata[field_name] = {
            level: (count if count >= min_cell_count else "suppressed_small_cell")
            for level, count in sorted(counts.items())
        }
    return {
        "cohort_id": summary["cohort_id"],
        "data_class": summary["data_class"],
        "participant_count": summary["participant_count"],
        "trajectory_count": summary["trajectory_count"],
        "participant_held_out_count": len(summary["participant_held_out"]),
        "stratum_counts": strata,
        "governance": dict(record["governance"]),
        "human_calibration_authority": summary["human_calibration_authority"],
        "personalized_recommendation_authority": "prohibited",
        "inference_boundary": record["inference_boundary"],
    }


def personalized_recommendation(
    record: dict[str, Any], participant_pseudonym: str, statement: str
) -> str:
    """Refuse every personalized recommendation outside the evidence domain.

    No governed participant evidence exists in this program, so this function
    makes the refusal executable and testable rather than editorial. It raises
    unless the cohort is governed human data whose governance gates are all
    satisfied and whose declared evidence domain already admits ``statement``.
    """
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("statement must be a nonempty string")
    summary = validate_cohort(record)
    if participant_pseudonym not in {
        row["participant_pseudonym"] for row in record["participants"]
    }:
        raise ValueError("participant_pseudonym is not in the cohort")
    if summary["human_calibration_authority"] != "available":
        raise ValueError(
            "personalized recommendation is outside the declared evidence domain: "
            "no governed participant authority"
        )
    if statement not in record["evidence_domain"]["declared_scope"]:
        raise ValueError(
            "personalized recommendation is outside the declared evidence domain"
        )
    return statement
