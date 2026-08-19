from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_inertia_cross_engine import (
    ArticulatedInertiaConfig,
    finite_difference_kinematics,
)
from scripts.research.proximal_distal_energy.register_articulated_inertia_claims import (
    _reconcile,
)

pytestmark = pytest.mark.scientific

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def test_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="mass_matrix_relative_tolerance"):
        ArticulatedInertiaConfig(mass_matrix_relative_tolerance=0.0)
    with pytest.raises(ValueError, match="inverse_dynamics_relative_tolerance"):
        ArticulatedInertiaConfig(inverse_dynamics_relative_tolerance=np.inf)
    with pytest.raises(ValueError, match="minimum_eigenvalue_tolerance"):
        ArticulatedInertiaConfig(minimum_eigenvalue_tolerance=-1.0)


def test_finite_difference_kinematics_recovers_quadratic_interior() -> None:
    time = np.linspace(0.0, 0.24, 13)
    coefficients = np.arange(1.0, 5.0)
    position = time[:, None] ** 2 * coefficients[None, :]
    velocity, acceleration = finite_difference_kinematics(position, time)
    np.testing.assert_allclose(
        velocity[1:-1], 2.0 * time[1:-1, None] * coefficients, atol=1.0e-12
    )
    np.testing.assert_allclose(
        acceleration[1:-1],
        np.broadcast_to(2.0 * coefficients, acceleration[1:-1].shape),
        atol=1.0e-11,
    )


def test_committed_articulated_inertia_evidence_is_complete_and_bounded() -> None:
    record = json.loads(
        (DATA_DIR / "articulated_inertia_cross_engine.json").read_text(encoding="utf-8")
    )
    assert record["schema_version"] == "articulated-inertia-cross-engine/v1"
    assert record["design"]["profile_count"] == 6
    assert record["design"]["grip_span_count"] == 3
    assert record["design"]["state_count"] == 234
    assert record["results"]["all_registered_gates_passed"] is True
    assert record["results"]["failed_state_count"] == 0
    assert record["claim_boundary"]["forward_contact"] == "not_established"
    assert record["claim_boundary"]["human_strategy"] == "untested"
    tolerances = record["tolerances"]
    results = record["results"]
    assert (
        results["maximum_mass_matrix_relative_error"]
        <= tolerances["mass_matrix_relative_tolerance"]
    )
    assert (
        results["maximum_bias_relative_error"] <= tolerances["bias_relative_tolerance"]
    )
    assert (
        results["maximum_inverse_dynamics_relative_error"]
        <= tolerances["inverse_dynamics_relative_tolerance"]
    )
    for path, digest in record["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


def test_committed_articulated_inertia_arrays_are_finite() -> None:
    with np.load(DATA_DIR / "articulated_inertia_cross_engine.npz") as arrays:
        assert arrays["mass_matrix_relative_error"].shape == (18, 13)
        assert arrays["bias_relative_error"].shape == (18, 13)
        assert arrays["inverse_dynamics_relative_error"].shape == (18, 13)
        assert arrays["minimum_mass_matrix_eigenvalue"].shape == (18, 13, 2)
        assert arrays["engine_names"].tolist() == ["mujoco", "pinocchio"]
        for key in arrays.files:
            if arrays[key].dtype.kind in "fc":
                assert np.all(np.isfinite(arrays[key])), key


def test_claim_registration_prunes_superseded_candidate_reviews() -> None:
    current = {
        "candidate_id": "PD-CAND-current",
        "source_path": "paper.qmd",
        "line_start": 1,
    }
    registry = {
        "claims": [],
        "candidate_reviews": [
            {
                "candidate_id": "PD-CAND-stale",
                "disposition": "material_claims_mapped",
                "claim_ids": ["PD-CLAIM-276"],
            }
        ],
        "release_claim_inventory": [],
        "audit_scope": {},
        "paper": {},
    }
    inventory = {"candidates": [current], "source_digest": "digest"}

    _reconcile(
        registry,
        inventory,
        [],
        {
            "design": current,
            "methods": current,
            "figure": current,
            "result": current,
            "boundary": current,
        },
    )

    assert [review["candidate_id"] for review in registry["candidate_reviews"]] == [
        "PD-CAND-current"
    ]
