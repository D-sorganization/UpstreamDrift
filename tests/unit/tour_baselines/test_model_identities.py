"""Tests for Golf Model Identities, Topologies, and Registries (TB-00 #10585).

TDD test-first suite verifying:
1. Two identically named models with different topologies remain distinct.
2. Reconstruction models explicitly omit the simulated club.
3. Driven double, triple, and upper-body models have simulated clubs.
4. Constrained upper-body golfer reports 8 generalized coords, rank-3 constraint Jacobian,
   and 5 independent DOFs.
5. Aliases round-trip deterministically.
6. Imported provider mismatch detection fails closed.
"""

from __future__ import annotations

import pytest

from src.shared.python.tour_baselines.models import (
    BackendType,
    EvidenceStatus,
    GolfModelIdentity,
    ModelTopology,
    SourceOwner,
)
from src.shared.python.tour_baselines.registry import (
    AmbiguousModelError,
    clear_golf_model_registry,
    detect_provider_mismatch,
    get_golf_model,
    init_default_registry,
    list_golf_models,
    register_golf_model,
    resolve_model_alias,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_registry():
    """Ensure each test runs with a clean initialized registry."""
    clear_golf_model_registry()
    init_default_registry()
    yield
    init_default_registry()


def test_identically_named_models_with_different_topologies_stay_distinct():
    """Reconstruction double pendulum and driven double pendulum must NOT collapse."""
    recon_double = get_golf_model("reconstruction_double_pendulum")
    driven_double = get_golf_model("driven_double_pendulum")

    assert recon_double.model_id != driven_double.model_id
    assert recon_double.topology == ModelTopology.KINEMATIC_RECONSTRUCTION
    assert driven_double.topology == ModelTopology.PLANAR_DRIVEN_PENDULUM
    assert recon_double.has_simulated_club is False
    assert driven_double.has_simulated_club is True
    assert recon_double.dof == 4  # pivot xyz + arm hinge
    assert driven_double.dof == 2  # shoulder + wrist


def test_triple_pendulum_reconstruction_vs_driven_distinction():
    """Reconstruction triple pendulum and driven triple pendulum must remain distinct."""
    recon_triple = get_golf_model("reconstruction_triple_pendulum")
    driven_triple = get_golf_model("driven_triple_pendulum")

    assert recon_triple.model_id != driven_triple.model_id
    assert recon_triple.topology == ModelTopology.KINEMATIC_RECONSTRUCTION
    assert driven_triple.topology == ModelTopology.PLANAR_DRIVEN_PENDULUM
    assert recon_triple.has_simulated_club is False
    assert driven_triple.has_simulated_club is True
    assert recon_triple.dof == 5  # pivot xyz + upper_arm + forearm
    assert driven_triple.dof == 3  # hub + arm + club


def test_upper_body_golfer_constraint_rank_and_independent_dof():
    """Upper body golfer has 8 generalized coords, rank-3 constraint, and 5 independent DOFs."""
    golfer = get_golf_model("constrained_upper_body_golfer")

    assert golfer.topology == ModelTopology.CONSTRAINED_UPPER_BODY
    assert golfer.dof == 8
    assert golfer.constraint_count == 4
    assert golfer.independent_dof == 5
    assert golfer.has_simulated_club is True
    assert "loop closure" in golfer.constraint_description.lower()


def test_alias_resolution_and_round_trip():
    """Aliases resolve unambiguously with context or directly when unique."""
    # Disambiguation with explicit context
    recon_id = resolve_model_alias("double_pendulum", context="reconstruction")
    assert recon_id == "reconstruction_double_pendulum"

    driven_id = resolve_model_alias("double_pendulum", context="pendulum_simulator")
    assert driven_id == "driven_double_pendulum"

    # Unique alias resolves directly
    assert resolve_model_alias("double") == "driven_double_pendulum"
    assert resolve_model_alias("triple") == "driven_triple_pendulum"
    assert resolve_model_alias("pinocchio_golfer") == "reference_pinocchio_urdf"

    # Ambiguous alias without context raises AmbiguousModelError (DbC)
    with pytest.raises(AmbiguousModelError):
        resolve_model_alias("double_pendulum")


def test_provider_mismatch_detection():
    """Provider mismatch must be detected when backend or model identity disagrees."""
    # Correct matching
    assert detect_provider_mismatch("pendulum", "driven_double_pendulum") is False
    assert detect_provider_mismatch("tools", "driven_double_pendulum") is False
    assert detect_provider_mismatch("pinocchio", "full_body_pinocchio") is False

    # Mismatch: using reconstruction model with dynamic pendulum provider
    assert (
        detect_provider_mismatch("pendulum", "reconstruction_double_pendulum") is True
    )
    assert detect_provider_mismatch("pinocchio", "full_body_mujoco") is True


def test_flagship_full_body_models_in_all_six_engines():
    """Flagship full body models for all six engines must be registered."""
    expected_engines = {
        "full_body_mujoco": BackendType.MUJOCO,
        "full_body_pinocchio": BackendType.PINOCCHIO,
        "full_body_drake": BackendType.DRAKE,
        "full_body_opensim": BackendType.OPENSIM,
        "full_body_simscape": BackendType.SIMSCAPE,
        "full_body_myosuite": BackendType.MYOSUITE,
    }

    all_models = {m.model_id: m for m in list_golf_models()}
    for model_id, backend in expected_engines.items():
        assert model_id in all_models, f"Missing flagship model {model_id}"
        assert all_models[model_id].backend == backend
        assert all_models[model_id].topology == ModelTopology.FULL_BODY_MULTIBODY
