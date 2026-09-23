"""Unit and behavioral tests for NM-09 checkpoint matrix and qualification (#10624).

Acceptance criteria (issue copy):
- One model cannot load another model card accidentally (assert_model_checkpoint_compatible).
- Variable nq/nv/nu dimensions bound to registered GolfModelIdentity.
- Native replay per checkpoint with full horizon and contact/loop-closure validity.
- Kinematic reconstruction models receive kinematic proposal network or named missing-dynamics prerequisite, never fabricated torques.
- Optional runtimes missing stay blocked (fail-closed, explicit named blockers).
- Roster completeness covering every registered #10585 model.
- Checkpoint hash chain integrity: dataset_hash -> split_hash -> weight_digest -> card_digest -> matrix_digest.
"""

from __future__ import annotations

import math
from typing import Any
import pytest

from src.shared.python.tour_baselines.models import ModelTopology
from src.shared.python.tour_baselines.registry import list_golf_models, get_golf_model
from src.shared.python.neural_motion.matrix.types import (
    CHECKPOINT_SCHEMA,
    BenefitResult,
    ModelCheckpointCard,
    ModelCheckpointStatus,
    NativeReplayReceipt,
    ThreeSeedEvidence,
    assert_model_checkpoint_compatible,
)
from src.shared.python.neural_motion.matrix.builder import (
    MATRIX_SCHEMA,
    NeuralCheckpointMatrix,
    build_checkpoint_matrix,
)
from src.shared.python.neural_motion.matrix.replay import (
    verify_checkpoint_native_replay,
)
from src.shared.python.neural_motion.matrix.adapters import (
    is_runtime_available_for_model,
)

pytestmark = pytest.mark.unit


def _dummy_evidence() -> ThreeSeedEvidence:
    return ThreeSeedEvidence(
        seed_losses=((11, 0.042), (22, 0.041), (33, 0.045)),
        mean_loss=0.042667,
        std_loss=0.001699,
        converged=True,
    )


def _dummy_receipt(
    *,
    is_valid: bool = True,
    replay_rmse: float = 0.012,
    max_constraint_violation: float = 0.001,
) -> NativeReplayReceipt:
    return NativeReplayReceipt(
        is_valid=is_valid,
        replay_rmse=replay_rmse,
        max_constraint_violation=max_constraint_violation,
        horizon_s=0.6,
        time_step_s=0.01,
        backend="scipy_ode",
        native_engine_version="1.0.0",
        receipt_digest="a" * 16,
    )


def _dummy_benefit() -> BenefitResult:
    return BenefitResult(
        speedup_factor=4.5,
        loss_reduction_pct=15.2,
        break_even_queries=12,
        verdict="favorable_speedup",
    )


def _dummy_card(
    *,
    model_id: str = "driven_double_pendulum",
    q_dim: int = 2,
    v_dim: int = 2,
    u_dim: int = 2,
    constraint_count: int = 0,
    control_basis: str = "joint_torque",
    status: ModelCheckpointStatus = ModelCheckpointStatus.QUALIFIED_NATIVE,
) -> ModelCheckpointCard:
    return ModelCheckpointCard(
        schema=CHECKPOINT_SCHEMA,
        model_id=model_id,
        backend="scipy_ode",
        topology="planar_driven_pendulum",
        q_dim=q_dim,
        v_dim=v_dim,
        u_dim=u_dim,
        constraint_count=constraint_count,
        control_basis=control_basis,
        conditioning_schema="conditioning/1.0",
        generator_adapter="planar_double_pendulum",
        dataset_hash="1" * 16,
        split_hash="2" * 16,
        weight_digest="3" * 16,
        three_seed_evidence=_dummy_evidence(),
        native_replay_receipt=_dummy_receipt(),
        benefit_result=_dummy_benefit(),
        status=status,
        blockers=(),
        governing_issues=("#10585", "#10624"),
    )


class TestModelCardCompatibility:
    """Accidental cross-model loading rejection."""

    def test_exact_match_succeeds(self) -> None:
        card = _dummy_card(model_id="driven_double_pendulum", u_dim=2)
        assert_model_checkpoint_compatible(
            card=card,
            expected_model_id="driven_double_pendulum",
            expected_u_dim=2,
            expected_control_basis="joint_torque",
        )

    def test_mismatched_model_id_raises_value_error(self) -> None:
        card = _dummy_card(model_id="driven_double_pendulum")
        with pytest.raises(ValueError, match="incompatible checkpoint model_id"):
            assert_model_checkpoint_compatible(
                card=card,
                expected_model_id="driven_triple_pendulum",
                expected_u_dim=2,
                expected_control_basis="joint_torque",
            )

    def test_mismatched_u_dim_raises_value_error(self) -> None:
        card = _dummy_card(model_id="driven_double_pendulum", u_dim=2)
        with pytest.raises(ValueError, match="incompatible checkpoint u_dim"):
            assert_model_checkpoint_compatible(
                card=card,
                expected_model_id="driven_double_pendulum",
                expected_u_dim=3,
                expected_control_basis="joint_torque",
            )

    def test_mismatched_control_basis_raises_value_error(self) -> None:
        card = _dummy_card(
            model_id="driven_double_pendulum", control_basis="joint_torque"
        )
        with pytest.raises(ValueError, match="incompatible checkpoint control_basis"):
            assert_model_checkpoint_compatible(
                card=card,
                expected_model_id="driven_double_pendulum",
                expected_u_dim=2,
                expected_control_basis="generalized_force",
            )

    def test_invalid_schema_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="incompatible checkpoint schema"):
            bad_card = ModelCheckpointCard(
                schema="invalid-schema/0.0",
                model_id="driven_double_pendulum",
                backend="scipy_ode",
                topology="planar_driven_pendulum",
                q_dim=2,
                v_dim=2,
                u_dim=2,
                constraint_count=0,
                control_basis="joint_torque",
                conditioning_schema="c/1",
                generator_adapter="g/1",
                dataset_hash="1" * 16,
                split_hash="2" * 16,
                weight_digest="3" * 16,
                three_seed_evidence=_dummy_evidence(),
                native_replay_receipt=_dummy_receipt(),
                benefit_result=_dummy_benefit(),
                status=ModelCheckpointStatus.QUALIFIED_NATIVE,
                blockers=(),
                governing_issues=("#10624",),
            )
            assert_model_checkpoint_compatible(
                card=bad_card,
                expected_model_id="driven_double_pendulum",
                expected_u_dim=2,
                expected_control_basis="joint_torque",
            )


class TestKinematicModelsNoFabricatedTorques:
    """Reconstruction models must use kinematic proposals and never fabricate torques."""

    def test_kinematic_reconstruction_models_have_kinematic_basis(self) -> None:
        matrix = build_checkpoint_matrix()
        for model_id in [
            "reconstruction_golfer",
            "reconstruction_double_pendulum",
            "reconstruction_triple_pendulum",
        ]:
            card = matrix.get_card(model_id)
            assert card.control_basis != "joint_torque", (
                f"{model_id} must not fabricate torque control basis"
            )
            assert card.control_basis in (
                "kinematic_joint_angle",
                "kinematic_marker_trajectory",
            )
            assert (
                card.status == ModelCheckpointStatus.KINEMATIC_PROPOSAL
                or "lacks torque-driven supervision" in " ".join(card.blockers)
            )


class TestRosterCompletenessAndHashChain:
    """Completeness over #10585 roster and cryptographic hash chain."""

    def test_all_registered_models_present(self) -> None:
        all_models = list_golf_models()
        matrix = build_checkpoint_matrix()
        registered_ids = {m.model_id for m in all_models}
        matrix_ids = {card.model_id for card in matrix.cards}
        assert registered_ids == matrix_ids, (
            f"Matrix missing models: {registered_ids - matrix_ids}"
        )

    def test_dimensions_match_registered_identities(self) -> None:
        matrix = build_checkpoint_matrix()
        for card in matrix.cards:
            identity = get_golf_model(card.model_id)
            assert card.q_dim == identity.dof
            assert card.v_dim == identity.dof
            assert card.u_dim == identity.independent_dof
            assert card.constraint_count == identity.constraint_count
            assert card.backend == identity.backend.value

    def test_matrix_hash_chain_is_deterministic_and_unique(self) -> None:
        matrix1 = build_checkpoint_matrix()
        matrix2 = build_checkpoint_matrix()
        digest1 = matrix1.matrix_digest()
        digest2 = matrix2.matrix_digest()
        assert digest1 == digest2
        assert len(digest1) == 64
        # Verify card hashes are unique across distinct models
        card_hashes = [card.checkpoint_hash() for card in matrix1.cards]
        assert len(card_hashes) == len(set(card_hashes))


class TestNativeReplayVerification:
    """Native forward dynamics replay and constraint validation."""

    def test_driven_double_pendulum_native_replay(self) -> None:
        matrix = build_checkpoint_matrix()
        card = matrix.get_card("driven_double_pendulum")
        assert card.status == ModelCheckpointStatus.QUALIFIED_NATIVE
        receipt = verify_checkpoint_native_replay(card)
        assert receipt.is_valid
        assert receipt.replay_rmse < 0.05
        assert receipt.max_constraint_violation == 0.0  # unconstrained mechanism

    def test_driven_triple_pendulum_native_replay(self) -> None:
        matrix = build_checkpoint_matrix()
        card = matrix.get_card("driven_triple_pendulum")
        assert card.status == ModelCheckpointStatus.QUALIFIED_NATIVE
        receipt = verify_checkpoint_native_replay(card)
        assert receipt.is_valid
        assert receipt.replay_rmse < 0.05
        assert receipt.max_constraint_violation == 0.0

    def test_constrained_upper_body_native_replay_and_loop_closure(self) -> None:
        matrix = build_checkpoint_matrix()
        card = matrix.get_card("constrained_upper_body_golfer")
        assert card.status in (
            ModelCheckpointStatus.QUALIFIED_NATIVE,
            ModelCheckpointStatus.TRAINED_SURROGATE,
        )
        assert card.constraint_count == 4
        receipt = verify_checkpoint_native_replay(card)
        assert receipt.is_valid
        assert receipt.max_constraint_violation < 1e-4

    def test_nonfinite_trajectory_fails_closed(self) -> None:
        bad_card = _dummy_card(
            model_id="driven_double_pendulum",
            status=ModelCheckpointStatus.QUALIFIED_NATIVE,
        )
        # Verify that corrupted/non-finite native replay fails closed
        with pytest.raises(ValueError, match="non-finite|invalid"):
            verify_checkpoint_native_replay(bad_card, inject_nan=True)


class TestOptionalRuntimeBlockers:
    """Missing optional physics runtimes remain fail-closed with named blockers."""

    def test_missing_runtimes_explicitly_blocked(self) -> None:
        matrix = build_checkpoint_matrix()

        # Simscape requires MATLAB R2025b
        simscape_card = matrix.get_card("full_body_simscape")
        if not is_runtime_available_for_model("full_body_simscape"):
            assert simscape_card.status == ModelCheckpointStatus.BLOCKED_PREREQUISITE
            assert any(
                "MATLAB R2025b" in b or "Simscape" in b for b in simscape_card.blockers
            )

        # MyoSuite fail-closed per MS-50
        myosuite_card = matrix.get_card("full_body_myosuite")
        assert myosuite_card.status == ModelCheckpointStatus.BLOCKED_PREREQUISITE
        assert any("MS-50" in b or "MyoSuite" in b for b in myosuite_card.blockers)

        # OpenSim requires Moco
        opensim_card = matrix.get_card("full_body_opensim")
        if not is_runtime_available_for_model("full_body_opensim"):
            assert opensim_card.status == ModelCheckpointStatus.BLOCKED_PREREQUISITE
            assert any(
                "OpenSim" in b or "moco" in b.lower() for b in opensim_card.blockers
            )

    def test_pendulum_model_registry_checkpoint_resolution(self) -> None:
        from src.shared.python.pendulum_simulator.model_registry import (
            resolve_model_checkpoint,
        )

        card_double = resolve_model_checkpoint("double")
        assert card_double.model_id == "driven_double_pendulum"
        assert card_double.status == ModelCheckpointStatus.QUALIFIED_NATIVE

        card_triple = resolve_model_checkpoint("triple")
        assert card_triple.model_id == "driven_triple_pendulum"
        assert card_triple.status == ModelCheckpointStatus.QUALIFIED_NATIVE

        card_golfer = resolve_model_checkpoint("golfer")
        assert card_golfer.model_id == "constrained_upper_body_golfer"
        assert card_golfer.status == ModelCheckpointStatus.QUALIFIED_NATIVE
