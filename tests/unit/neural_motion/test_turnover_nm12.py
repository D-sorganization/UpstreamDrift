"""Unit and behavioral tests for NM-12 model cards, reproduction commands and turnover (#10627).

Acceptance criteria (issue copy):
- Complete reproduction catalog covering all 20 registered models in list_golf_models().
- Clean reproduction commands for 5 lifecycle phases: generate, train, evaluate, infer, replay.
- Model reproduction cards with schema validation, provenance, economics, and limitations.
- Promotion verdicts correctly assigned across promoted, research, blocked, and reference categories.
- End-to-end user flow verification validating all 5 steps from registration to physical replay.
- JSON round-trip serialization and manifest generation.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.shared.python.neural_motion.turnover import (
    EndToEndFlowReport,
    FlowStepOutcome,
    FlowStepStatus,
    ModelReproductionCard,
    PromotionVerdict,
    build_reproduction_catalog,
    generate_reproduction_commands,
    load_reproduction_card,
    save_reproduction_catalog,
    verify_end_to_end_flow,
)
from src.shared.python.tour_baselines.registry import list_golf_models

pytestmark = pytest.mark.unit


class TestModelReproductionCardSchema:
    """Validate ModelReproductionCard schema constraints and error handling."""

    def test_valid_card_creation_and_to_dict(self) -> None:
        commands = generate_reproduction_commands("driven_double_pendulum")
        card = ModelReproductionCard(
            model_id="driven_double_pendulum",
            intended_task="Fast neural motion proposal for double pendulum",
            input_observations=("q", "v"),
            physical_assumptions=("planar_motion", "rigid_bodies"),
            dataset_split_provenance={"seed": "42", "episodes": "100"},
            native_validation={"is_valid": True, "rmse": 0.012},
            performance_economics={"speedup": 12.3},
            limits_and_licensing=("MIT License",),
            commands=commands,
            verdict=PromotionVerdict.PROMOTED,
        )
        assert card.model_id == "driven_double_pendulum"
        assert card.verdict == PromotionVerdict.PROMOTED
        data = card.to_dict()
        assert data["model_id"] == "driven_double_pendulum"
        assert data["verdict"] == "PROMOTED"
        assert data["schema_version"] == "neural-model-reproduction-card/1.0.0"

    def test_round_trip_serialization(self) -> None:
        commands = generate_reproduction_commands("driven_double_pendulum")
        card = ModelReproductionCard(
            model_id="driven_double_pendulum",
            intended_task="Fast neural motion proposal",
            input_observations=("q", "v"),
            physical_assumptions=("planar_motion",),
            dataset_split_provenance={"seed": "42"},
            native_validation={"is_valid": True},
            performance_economics={"speedup": 10.0},
            limits_and_licensing=("MIT",),
            commands=commands,
            verdict=PromotionVerdict.PROMOTED,
        )
        data = card.to_dict()
        restored = ModelReproductionCard.from_dict(data)
        assert restored.model_id == card.model_id
        assert restored.intended_task == card.intended_task
        assert restored.verdict == card.verdict
        assert restored.commands == card.commands

    def test_rejects_empty_model_id(self) -> None:
        commands = generate_reproduction_commands("dummy")
        with pytest.raises(ValueError, match="model_id must be a non-empty string"):
            ModelReproductionCard(
                model_id="",
                intended_task="Task",
                input_observations=("q",),
                physical_assumptions=("assumptions",),
                dataset_split_provenance={},
                native_validation={},
                performance_economics={},
                limits_and_licensing=(),
                commands=commands,
                verdict=PromotionVerdict.RESEARCH_ONLY,
            )

    def test_rejects_empty_intended_task(self) -> None:
        commands = generate_reproduction_commands("dummy")
        with pytest.raises(
            ValueError, match="intended_task must be a non-empty string"
        ):
            ModelReproductionCard(
                model_id="dummy",
                intended_task="",
                input_observations=("q",),
                physical_assumptions=("assumptions",),
                dataset_split_provenance={},
                native_validation={},
                performance_economics={},
                limits_and_licensing=(),
                commands=commands,
                verdict=PromotionVerdict.RESEARCH_ONLY,
            )

    def test_rejects_empty_observations(self) -> None:
        commands = generate_reproduction_commands("dummy")
        with pytest.raises(
            ValueError, match="input_observations must be a non-empty tuple"
        ):
            ModelReproductionCard(
                model_id="dummy",
                intended_task="Task",
                input_observations=(),
                physical_assumptions=("assumptions",),
                dataset_split_provenance={},
                native_validation={},
                performance_economics={},
                limits_and_licensing=(),
                commands=commands,
                verdict=PromotionVerdict.RESEARCH_ONLY,
            )

    def test_rejects_missing_command_key(self) -> None:
        bad_commands = {"generate": "echo 1", "train": "echo 2"}
        with pytest.raises(ValueError, match="commands mapping missing required key"):
            ModelReproductionCard(
                model_id="dummy",
                intended_task="Task",
                input_observations=("q",),
                physical_assumptions=("assumptions",),
                dataset_split_provenance={},
                native_validation={},
                performance_economics={},
                limits_and_licensing=(),
                commands=bad_commands,
                verdict=PromotionVerdict.RESEARCH_ONLY,
            )

    def test_from_dict_rejects_invalid_schema(self) -> None:
        commands = generate_reproduction_commands("dummy")
        bad_dict = {
            "schema_version": "invalid-schema/2.0",
            "model_id": "dummy",
            "intended_task": "Task",
            "input_observations": ["q"],
            "physical_assumptions": ["assumptions"],
            "dataset_split_provenance": {},
            "native_validation": {},
            "performance_economics": {},
            "limits_and_licensing": [],
            "commands": commands,
            "verdict": "RESEARCH_ONLY",
        }
        with pytest.raises(ValueError, match="Unsupported schema_version"):
            ModelReproductionCard.from_dict(bad_dict)


class TestReproductionCatalog:
    """Validate catalog generation, coverage, and verdicts."""

    def test_catalog_covers_all_registered_models(self) -> None:
        registered = list_golf_models()
        catalog = build_reproduction_catalog()
        assert len(catalog) == len(registered)
        assert len(catalog) == 20

        registered_ids = {m.model_id for m in registered}
        catalog_ids = {c.model_id for c in catalog}
        assert registered_ids == catalog_ids

    def test_promoted_models_classification(self) -> None:
        catalog = {c.model_id: c for c in build_reproduction_catalog()}
        assert catalog["driven_double_pendulum"].verdict != PromotionVerdict.PROMOTED
        assert catalog["driven_double_pendulum"].verdict == PromotionVerdict.UNMEASURED
        assert catalog["driven_triple_pendulum"].verdict == PromotionVerdict.UNMEASURED
        assert (
            catalog["constrained_upper_body_golfer"].verdict
            == PromotionVerdict.UNMEASURED
        )
        # Performance economics must be unmeasured without benchmark receipt on disk
        econ = catalog["driven_double_pendulum"].performance_economics
        assert econ["speedup_factor"] is None
        assert econ["verdict"] == "UNMEASURED"

    def test_research_models_classification(self) -> None:
        catalog = {c.model_id: c for c in build_reproduction_catalog()}
        assert (
            catalog["reconstruction_double_pendulum"].verdict
            == PromotionVerdict.RESEARCH_ONLY
        )
        assert (
            catalog["reconstruction_triple_pendulum"].verdict
            == PromotionVerdict.RESEARCH_ONLY
        )
        assert (
            catalog["reconstruction_golfer"].verdict == PromotionVerdict.RESEARCH_ONLY
        )

    def test_blocked_prerequisite_classification(self) -> None:
        catalog = {c.model_id: c for c in build_reproduction_catalog()}
        blocked_models = [
            "full_body_drake",
            "full_body_mujoco",
            "full_body_myosuite",
            "full_body_opensim",
            "full_body_pinocchio",
            "full_body_simscape",
        ]
        for mid in blocked_models:
            assert catalog[mid].verdict == PromotionVerdict.BLOCKED_PREREQUISITE
            assert any(
                "PREREQUISITE BLOCKED" in limit
                for limit in catalog[mid].limits_and_licensing
            )

    def test_reference_only_classification(self) -> None:
        catalog = {c.model_id: c for c in build_reproduction_catalog()}
        ref_models = [
            "myosuite_body",
            "opensim_golfer",
            "reference_drake_urdf",
            "reference_human_subject",
            "reference_mujoco_humanoid",
            "reference_pinocchio_urdf",
            "reference_pinocchio_urdf_ik",
            "reference_simple_humanoid",
        ]
        for mid in ref_models:
            assert catalog[mid].verdict == PromotionVerdict.REFERENCE_ONLY

    def test_save_and_load_catalog(self, tmp_path: Path) -> None:
        catalog = build_reproduction_catalog()
        saved = save_reproduction_catalog(catalog, tmp_path)
        # 20 cards + 1 manifest = 21 files
        assert len(saved) == 21

        manifest_path = tmp_path / "catalog_manifest.json"
        assert manifest_path.exists()
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["card_count"] == 20
        assert len(manifest_data["cards"]) == 20

        # Load one card
        card_file = tmp_path / "driven_double_pendulum_reproduction_card.json"
        loaded_card = load_reproduction_card(card_file)
        assert loaded_card.model_id == "driven_double_pendulum"
        assert loaded_card.verdict == PromotionVerdict.UNMEASURED


class TestEndToEndFlowVerification:
    """Validate 5-step end-to-end user flow execution."""

    def test_unpromoted_model_flow_fails_without_receipt(self) -> None:
        report = verify_end_to_end_flow("driven_double_pendulum")
        assert isinstance(report, EndToEndFlowReport)
        assert report.model_id == "driven_double_pendulum"
        assert report.overall_success is False
        assert report.verdict != PromotionVerdict.PROMOTED
        assert report.verdict == PromotionVerdict.UNMEASURED
        assert len(report.steps) == 5

        step_names = [s.step_name for s in report.steps]
        assert step_names == [
            "DATASET_REGISTRATION",
            "TRAINING_RUN",
            "CHECKPOINT_SELECTION",
            "OBSERVED_MOTION_MATCHING",
            "PHYSICAL_REPLAY",
        ]
        # Step 4 (OBSERVED_MOTION_MATCHING) must not pass without real VerifiedInferenceOrchestrator run
        step4 = report.steps[3]
        assert step4.step_name == "OBSERVED_MOTION_MATCHING"
        assert step4.status in (FlowStepStatus.SKIPPED, FlowStepStatus.FAILED)

        # Step 5 (PHYSICAL_REPLAY) fails closed without trained checkpoint rollout
        step5 = report.steps[4]
        assert step5.step_name == "PHYSICAL_REPLAY"
        assert step5.status == FlowStepStatus.FAILED

    def test_blocked_prerequisite_flow(self) -> None:
        report = verify_end_to_end_flow("full_body_simscape")
        assert report.model_id == "full_body_simscape"
        assert report.overall_success is False
        assert report.verdict == PromotionVerdict.BLOCKED_PREREQUISITE

        # Steps 4 and 5 should be blocked
        step4 = report.steps[3]
        step5 = report.steps[4]
        assert step4.step_name == "OBSERVED_MOTION_MATCHING"
        assert step4.status == FlowStepStatus.BLOCKED
        assert step5.step_name == "PHYSICAL_REPLAY"
        assert step5.status == FlowStepStatus.BLOCKED

    def test_unknown_model_flow_fails(self) -> None:
        report = verify_end_to_end_flow("unknown_nonexistent_model")
        assert report.overall_success is False
        assert report.steps[0].status == FlowStepStatus.FAILED
        assert "not found" in report.steps[0].message

    def test_observed_motion_matching_requires_verified_orchestrator(self) -> None:
        from unittest.mock import MagicMock
        from src.shared.python.neural_motion.inference import (
            InferenceStatus,
            VerifiedInferenceOrchestrator,
            VerifiedInferenceReport,
        )
        from src.shared.python.neural_motion.turnover.reproduce import (
            _verify_inference_step,
        )

        # Without orchestrator -> SKIPPED with reason
        outcome_skipped = _verify_inference_step("driven_double_pendulum")
        assert outcome_skipped.status == FlowStepStatus.SKIPPED
        assert "not executed" in outcome_skipped.message

        # With orchestrator returning NEURAL_ACCEPTED -> PASSED
        mock_orch = MagicMock(spec=VerifiedInferenceOrchestrator)
        mock_report = MagicMock(spec=VerifiedInferenceReport)
        mock_report.status = InferenceStatus.NEURAL_ACCEPTED
        mock_orch.orchestrate.return_value = mock_report

        outcome_passed = _verify_inference_step(
            "driven_double_pendulum", orchestrator=mock_orch, target={}
        )
        assert outcome_passed.status == FlowStepStatus.PASSED

        # With orchestrator returning REJECTED -> FAILED
        mock_report_rejected = MagicMock(spec=VerifiedInferenceReport)
        mock_report_rejected.status = InferenceStatus.REJECTED
        mock_orch.orchestrate.return_value = mock_report_rejected

        outcome_failed = _verify_inference_step(
            "driven_double_pendulum", orchestrator=mock_orch, target={}
        )
        assert outcome_failed.status == FlowStepStatus.FAILED
