"""CI gate for the #15794 B1 parity capability matrix and tolerances."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.validate_parity_capability_matrix import (
    MATRIX_PATH,
    TOLERANCES_PATH,
    main,
    validate_matrix,
    validate_paths,
)

pytestmark = pytest.mark.unit

ENGINES = (
    "drake",
    "mujoco",
    "myosuite",
    "opensim",
    "pinocchio",
    "simscape",
)
CAPABILITIES = (
    "prescribed_kinematics",
    "instantaneous_dynamics",
    "grip_flexible_shaft",
    "ground_contact",
)
EPIC_MAXIMA = {
    "prescribed_kinematics": {
        "landmark_error_mm": 0.1,
        "angular_error_mrad": 1.0,
    },
    "instantaneous_dynamics": {
        "relative_l2": 1.0e-3,
        "torque_floor_nm": 0.01,
    },
    "grip_flexible_shaft": {
        "loop_closure_mm": 1.0,
        "tip_deflection_mm": 2.0,
        "modal_frequency_relative_error": 0.01,
    },
    "ground_contact": {
        "net_impulse_relative_error": 0.03,
        "cop_rms_mm": 10.0,
        "vertical_load_threshold_n": 50.0,
    },
}


def _changelog() -> dict:
    """Return an approval-empty changelog document."""
    return {"schema_version": "2.0.0", "entries": []}


def _tolerances() -> dict:
    """Return tolerances built from the epic maxima."""
    regimes = {}
    for regime, bounds in EPIC_MAXIMA.items():
        regimes[regime] = {
            "bounds": {name: {"max": value} for name, value in bounds.items()}
        }
    return {"schema_version": "2.0.0", "version": 1, "regimes": regimes}


def _matrix() -> dict:
    """Return a claimed baseline matrix covering every engine."""
    cells = {engine: {"status": "claimed", "claimed_cycle": 1} for engine in ENGINES}
    return {
        "schema_version": "2.0.0",
        "release_cycle": 1,
        "claim_expiry_cycles": 2,
        "engines": list(ENGINES),
        "capabilities": {
            name: {"tolerance_id": name, "cells": copy.deepcopy(cells)}
            for name in CAPABILITIES
        },
    }


def _codes(findings: list[str]) -> set[str]:
    """Extract the leading finding codes from a findings list."""
    return {item.split(":", 1)[0] for item in findings}


def test_unverified_baseline_is_accepted() -> None:
    """All six engines may start claimed inside the expiry window."""
    assert validate_matrix(_matrix(), _tolerances(), _changelog()) == []


def test_verified_cell_requires_fixture_and_known_tolerance() -> None:
    """A verified cell without a passing fixture is not evidence."""
    matrix = _matrix()
    matrix["capabilities"]["prescribed_kinematics"]["cells"]["drake"] = {
        "status": "verified",
        "tolerance_id": "prescribed_kinematics",
    }
    assert "verified_missing_fixture" in _codes(
        validate_matrix(matrix, _tolerances(), _changelog())
    )

    matrix["capabilities"]["prescribed_kinematics"]["cells"]["drake"]["fixture_id"] = (
        "capture-a-drake"
    )
    assert validate_matrix(matrix, _tolerances(), _changelog()) == []

    matrix["capabilities"]["prescribed_kinematics"]["cells"]["drake"][
        "tolerance_id"
    ] = "not-a-regime"
    assert "unknown_tolerance" in _codes(
        validate_matrix(matrix, _tolerances(), _changelog())
    )


def test_shared_capability_must_name_every_engine() -> None:
    """A shared feature is incomplete until all six engines have a status."""
    matrix = _matrix()
    del matrix["capabilities"]["ground_contact"]["cells"]["simscape"]
    assert "missing_engine" in _codes(
        validate_matrix(matrix, _tolerances(), _changelog())
    )


def test_unsupported_cell_requires_a_linked_issue() -> None:
    """Unsupported is a recorded gap, not a blank cell."""
    matrix = _matrix()
    matrix["capabilities"]["ground_contact"]["cells"]["pinocchio"] = {
        "status": "unsupported"
    }
    assert "unsupported_missing_issue" in _codes(
        validate_matrix(matrix, _tolerances(), _changelog())
    )
    matrix["capabilities"]["ground_contact"]["cells"]["pinocchio"]["issue"] = 15776
    assert validate_matrix(matrix, _tolerances(), _changelog()) == []


def test_claimed_cell_expires_after_two_cycles_without_a_fixture() -> None:
    """Claimed evidence expires two release cycles after it was claimed."""
    matrix = _matrix()
    matrix["release_cycle"] = 3
    assert "claim_expired" in _codes(
        validate_matrix(matrix, _tolerances(), _changelog())
    )
    matrix["capabilities"]["instantaneous_dynamics"]["cells"]["mujoco"][
        "fixture_id"
    ] = "capture-a-mujoco"
    findings = validate_matrix(matrix, _tolerances(), _changelog())
    assert "claim_expired:instantaneous_dynamics.mujoco" not in findings
    assert "claim_expired" in _codes(findings)


def test_loosening_or_downgrade_requires_codeowners_approval() -> None:
    """A looser bound or a lower status needs an approved changelog entry."""
    current = _matrix()
    previous = _matrix()
    tolerances = _tolerances()
    looser = copy.deepcopy(tolerances)
    looser["regimes"]["prescribed_kinematics"]["bounds"]["landmark_error_mm"]["max"] = (
        0.2
    )
    assert "tolerance_loosened" in _codes(
        validate_matrix(
            current,
            looser,
            _changelog(),
            baseline_matrix=previous,
            baseline_tolerances=tolerances,
        )
    )
    approved = _changelog()
    approved["entries"].append(
        {
            "kind": "tolerance_loosening",
            "regime": "prescribed_kinematics",
            "bound": "landmark_error_mm",
            "codeowners_approval": True,
        }
    )
    assert (
        validate_matrix(
            current,
            looser,
            approved,
            baseline_matrix=previous,
            baseline_tolerances=tolerances,
        )
        == []
    )

    downgraded = _matrix()
    downgraded["capabilities"]["grip_flexible_shaft"]["cells"]["opensim"] = {
        "status": "unsupported",
        "issue": 12691,
    }
    previous_verified = _matrix()
    previous_verified["capabilities"]["grip_flexible_shaft"]["cells"]["opensim"] = {
        "status": "verified",
        "fixture_id": "shaft-opensim",
        "tolerance_id": "grip_flexible_shaft",
    }
    assert "status_downgraded" in _codes(
        validate_matrix(
            downgraded,
            tolerances,
            _changelog(),
            baseline_matrix=previous_verified,
            baseline_tolerances=tolerances,
        )
    )
    approved_downgrade = _changelog()
    approved_downgrade["entries"].append(
        {
            "kind": "status_downgrade",
            "capability": "grip_flexible_shaft",
            "engine": "opensim",
            "codeowners_approval": True,
        }
    )
    assert (
        validate_matrix(
            downgraded,
            tolerances,
            approved_downgrade,
            baseline_matrix=previous_verified,
            baseline_tolerances=tolerances,
        )
        == []
    )


def test_committed_parity_package_matches_epic_gates() -> None:
    """The files CI reads are the #15794 provisional gates and pass the validator."""
    assert validate_paths(MATRIX_PATH, TOLERANCES_PATH) == []
    import yaml

    document = yaml.safe_load(TOLERANCES_PATH.read_text(encoding="utf-8"))
    for regime, bounds in EPIC_MAXIMA.items():
        recorded = document["regimes"][regime]["bounds"]
        for name, maximum in bounds.items():
            assert recorded[name]["max"] == maximum
    assert main([]) == 0


def test_cli_rejects_a_broken_matrix(tmp_path: Path) -> None:
    """The validator process exits non-zero when a verified cell has no fixture."""
    matrix = _matrix()
    matrix["capabilities"]["prescribed_kinematics"]["cells"]["drake"] = {
        "status": "verified",
        "tolerance_id": "prescribed_kinematics",
    }
    matrix_path = tmp_path / "matrix.json"
    tolerances_path = tmp_path / "tolerances.yaml"
    changelog_path = tmp_path / "changelog.yaml"
    matrix_path.write_text(__import__("json").dumps(matrix), encoding="utf-8")
    import yaml

    tolerances_path.write_text(yaml.safe_dump(_tolerances()), encoding="utf-8")
    changelog_path.write_text(yaml.safe_dump(_changelog()), encoding="utf-8")
    assert (
        main(
            [
                "--matrix",
                str(matrix_path),
                "--tolerances",
                str(tolerances_path),
                "--changelog",
                str(changelog_path),
            ]
        )
        == 1
    )
