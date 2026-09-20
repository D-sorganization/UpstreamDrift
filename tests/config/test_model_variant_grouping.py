"""Regression tests for model variant grouping, engine dashboards, and shortcuts (ORG-07, #10514).

Acceptance Cases:
- RED: seven exercises across four providers appear as seven logical choices with 28 retained variants.
- RED: partial/absent providers do not remove logical identity; selected engine is honored.
- RED: sit-to-stand does not fall back to gait; old dashboard/model shortcut IDs resolve.
- GREEN: name collisions across different real model identities are not incorrectly merged.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.shared.python.config.model_pack_manifest import (
    CrossEngineIdentity,
    ModelPackEntry,
)
from src.shared.python.config.model_registry import ModelConfig
from src.shared.python.config.model_variant_grouping import (
    LogicalModelChoice,
    LogicalModelIdentity,
    ModelGroupingProjection,
    ModelVariant,
    resolve_shortcut,
)

pytestmark = pytest.mark.unit

SEVEN_EXERCISES = (
    "squat",
    "deadlift",
    "bench_press",
    "snatch",
    "clean_and_jerk",
    "gait",
    "sit_to_stand",
)

FOUR_PROVIDERS = (
    ("mujoco_models", "mujoco", "mjcf"),
    ("drake_models", "drake", "sdf"),
    ("pinocchio_models", "pinocchio", "urdf"),
    ("opensim_models", "opensim", "osim"),
)


def _build_28_exercise_variants() -> list[ModelPackEntry]:
    """Generate 28 model pack entries (7 exercises x 4 engine providers)."""
    entries: list[ModelPackEntry] = []
    for ex in SEVEN_EXERCISES:
        for provider, engine, fmt in FOUR_PROVIDERS:
            entry = ModelPackEntry(
                id=f"{provider}-{ex}",
                name=ex.replace("_", " ").title(),
                description=f"{engine} model for {ex}",
                type=fmt,
                path=f"src/{provider}/exercises/{ex}",
                engine_type=engine,
                provider=provider,
                capabilities=("biomechanics", engine, ex),
                identity=CrossEngineIdentity(
                    canonical_id=f"biomech.exercise.{ex}",
                    motion_family="biomechanics",
                    exercise=ex,
                    humanoid="humanoid",
                ),
            )
            entries.append(entry)
    return entries


class TestModelVariantGrouping:
    def test_seven_exercises_across_four_providers_appear_as_seven_logical_choices_with_28_variants(
        self,
    ) -> None:
        """RED acceptance case:

        7 exercises across 4 providers appear as 7 logical choices with 28 retained variants.
        """
        entries = _build_28_exercise_variants()
        assert len(entries) == 28

        projection = ModelGroupingProjection()
        choices = projection.group_models(entries)

        assert len(choices) == 7
        total_variants = sum(choice.retained_variants_count for choice in choices)
        assert total_variants == 28

        # Verify each of the 7 exercises has all 4 engine variants retained
        for choice in choices:
            assert choice.retained_variants_count == 4
            assert set(choice.variants.keys()) == {
                "mujoco",
                "drake",
                "pinocchio",
                "opensim",
            }
            # Engine assets underneath are fully reachable
            for engine in ("mujoco", "drake", "pinocchio", "opensim"):
                variant = choice.get_variant(engine)
                assert isinstance(variant, ModelVariant)
                assert variant.engine_type == engine
                assert variant.path.startswith(f"src/{engine}_models/exercises/")

    def test_partial_or_absent_providers_preserve_logical_identity_and_honor_selected_engine(
        self,
    ) -> None:
        """RED acceptance case:

        Partial/absent providers do not remove logical identity; selected engine is honored.
        """
        # Only MuJoCo and Drake provide 'squat' in this slice
        partial_entries = [
            ModelPackEntry(
                id="mujoco_models-squat",
                name="Squat",
                description="MuJoCo squat",
                type="mjcf",
                path="src/mujoco_models/exercises/squat",
                engine_type="mujoco",
                provider="mujoco_models",
                capabilities=("biomechanics", "mujoco", "squat"),
                identity=CrossEngineIdentity(
                    canonical_id="biomech.exercise.squat",
                    motion_family="biomechanics",
                    exercise="squat",
                    humanoid="humanoid",
                ),
            ),
            ModelPackEntry(
                id="drake_models-squat",
                name="Squat",
                description="Drake squat",
                type="sdf",
                path="src/drake_models/exercises/squat",
                engine_type="drake",
                provider="drake_models",
                capabilities=("biomechanics", "drake", "squat"),
                identity=CrossEngineIdentity(
                    canonical_id="biomech.exercise.squat",
                    motion_family="biomechanics",
                    exercise="squat",
                    humanoid="humanoid",
                ),
            ),
        ]

        projection = ModelGroupingProjection()
        choices = projection.group_models(partial_entries)

        assert len(choices) == 1
        squat_choice = choices[0]
        assert squat_choice.identity.canonical_id == "biomech.exercise.squat"
        assert squat_choice.retained_variants_count == 2

        # Selected engine is honored
        drake_variant = squat_choice.get_variant("drake")
        assert drake_variant.engine_type == "drake"
        assert drake_variant.variant_id == "drake_models-squat"

        # Explicit non-silent failure for absent engine
        with pytest.raises(KeyError) as exc_info:
            squat_choice.get_variant("opensim")
        assert "opensim" in str(exc_info.value)

    def test_sit_to_stand_does_not_fall_back_to_gait_and_old_shortcut_ids_resolve(
        self,
    ) -> None:
        """RED acceptance case:

        sit-to-stand does not fall back to gait; old dashboard/model shortcut IDs resolve.
        """
        # 1. Shortcut resolution resolves legacy IDs
        target, params = resolve_shortcut("biomech_sit_to_stand")
        assert target == "biomech_exercise"
        assert params.get("exercise") == "sit_to_stand"
        assert params.get("exercise") != "gait"

        target_gait, params_gait = resolve_shortcut("biomech_gait")
        assert target_gait == "biomech_exercise"
        assert params_gait.get("exercise") == "gait"

        target_mujoco_dash, params_mujoco = resolve_shortcut("mujoco_dashboard")
        assert target_mujoco_dash == "mujoco_unified"
        assert params_mujoco.get("mode") == "dashboard"

        target_drake_dash, params_drake = resolve_shortcut("drake_dashboard")
        assert target_drake_dash == "drake_golf"
        assert params_drake.get("mode") == "dashboard"

        target_pin_dash, params_pin = resolve_shortcut("pinocchio_dashboard")
        assert target_pin_dash == "pinocchio_golf"
        assert params_pin.get("mode") == "dashboard"

        # 2. Handler execution: BiomechExerciseHandler with biomech_sit_to_stand
        from src.launchers.launcher_model_handlers import BiomechExerciseHandler

        handler = BiomechExerciseHandler()
        mock_proc_manager = MagicMock()
        mock_proc_manager.get_subprocess_env.return_value = {}
        mock_proc_manager.launch_script.return_value = "process-123"

        model = ModelConfig(
            id="biomech_sit_to_stand",
            name="Sit-to-Stand Model",
            description="Exercise preset for sit-to-stand",
            type="biomech_exercise",
            path="virtual/biomech_exercise/biomech_sit_to_stand",
            exercise="sit_to_stand",
        )

        res = handler.launch(model, Path("/fake/repo"), mock_proc_manager)
        assert res is True
        mock_proc_manager.launch_script.assert_called_once()
        call_kwargs = mock_proc_manager.launch_script.call_args[1]
        env = call_kwargs.get("env", {})
        assert env.get("BIOMECH_EXERCISE") == "sit_to_stand"
        assert env.get("BIOMECH_EXERCISE") != "gait"

    def test_name_collisions_across_different_real_identities_are_not_incorrectly_merged(
        self,
    ) -> None:
        """GREEN acceptance case:

        name collisions across different real model identities are not incorrectly merged.
        """
        # Two models with identical name "Full Body" but different canonical IDs and domains
        models = [
            ModelPackEntry(
                id="golf_humanoid_full_body",
                name="Full Body",
                description="Golf swing full body model",
                type="mjcf",
                path="models/golf/full_body.xml",
                engine_type="mujoco",
                identity=CrossEngineIdentity(
                    canonical_id="golf.swing.full_body",
                    motion_family="golf-swing",
                    exercise="driver",
                    humanoid="golfer",
                ),
            ),
            ModelPackEntry(
                id="clinical_gait_full_body",
                name="Full Body",
                description="Clinical gait full body model",
                type="osim",
                path="models/clinical/full_body.osim",
                engine_type="opensim",
                identity=CrossEngineIdentity(
                    canonical_id="clinical.gait.full_body",
                    motion_family="clinical-rehab",
                    exercise="gait",
                    humanoid="patient",
                ),
            ),
        ]

        projection = ModelGroupingProjection()
        choices = projection.group_models(models)

        assert len(choices) == 2
        choice_ids = {c.identity.canonical_id for c in choices}
        assert choice_ids == {"golf.swing.full_body", "clinical.gait.full_body"}

    def test_missing_checkout_diagnostic_for_shared_model_repos(self) -> None:
        """Move *_models_shared into Models/Integrations repository access with

        an explicit missing-checkout diagnostic.
        """
        from src.launchers.launcher_model_handlers import SharedRepoHandler

        handler = SharedRepoHandler()
        model = ModelConfig(
            id="mujoco_models_shared",
            name="MuJoCo Models",
            description="MuJoCo simulation models repo",
            type="shared_repo",
            path="MuJoCo_Models",
        )
        fake_repo = Path("/fake/repo/root/UpstreamDrift")
        diagnostic = handler.get_missing_checkout_diagnostic(model, fake_repo)

        assert "MuJoCo_Models" in diagnostic
        assert (
            "not checked out" in diagnostic.lower() or "missing" in diagnostic.lower()
        )
        assert "beside" in diagnostic.lower() or str(fake_repo.parent) in diagnostic

    def test_movement_optimizer_authority_task_mapping(self) -> None:
        """Map movement_optimizer and tools_movement_optimizer to one visible task

        with explicit authority selection under #9406.
        """
        target_mo, params_mo = resolve_shortcut("movement_optimizer")
        target_tmo, params_tmo = resolve_shortcut("tools_movement_optimizer")

        assert target_mo == "movement_optimizer"
        assert target_tmo == "movement_optimizer"
        assert "authority" in params_mo
        assert "authority" in params_tmo
