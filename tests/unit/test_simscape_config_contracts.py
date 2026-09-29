"""TDD tests for 3D Simscape config/pipeline incompatibilities (issue #2489).

Three bugs:
1. createSimulationConfig.m defaults model_name to 'GolfSwing3D_Model', but the repo
   contains 'GolfSwing3D_Kinetic.slx'. The default points at a nonexistent file.
2. createSimulationConfig.m defines config.verbosity (string), but processSimulationOutput.m
   reads config.verbose (boolean). The mismatch means processSimulationOutput.m reads a
   missing field before ensureEnhancedConfig can backfill it (race condition in call order).
3. createSimulationConfig.m validation warns on invalid config and returns the broken
   struct anyway — callers receive a config that cannot run.
"""

from __future__ import annotations

from pathlib import Path

_CREATE_CONFIG = Path(
    "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/"
    "scripts/dataset_generator/createSimulationConfig.m"
)
_PROCESS_OUTPUT = Path(
    "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/"
    "functions/dataset_generator/processSimulationOutput.m"
)


class TestCreateSimulationConfigModelName:
    """createSimulationConfig.m must default to the actual model filename."""

    def _source(self) -> str:
        return _CREATE_CONFIG.read_text(encoding="utf-8")

    def test_default_model_name_matches_actual_slx(self) -> None:
        """Default model_name must be 'GolfSwing3D_Kinetic', not 'GolfSwing3D_Model'."""
        source = self._source()
        assert (
            "'GolfSwing3D_Model'" not in source and '"GolfSwing3D_Model"' not in source
        ), (
            "createSimulationConfig.m still defaults model_name to 'GolfSwing3D_Model'. "
            "The actual model file is GolfSwing3D_Kinetic.slx. "
            "Fix: config.model_name = 'GolfSwing3D_Kinetic';"
        )

    def test_default_model_name_is_kinetic(self) -> None:
        """config.model_name must be 'GolfSwing3D_Kinetic'."""
        source = self._source()
        assert "'GolfSwing3D_Kinetic'" in source or '"GolfSwing3D_Kinetic"' in source, (
            "createSimulationConfig.m does not set model_name = 'GolfSwing3D_Kinetic'. "
            "The actual Simscape model is GolfSwing3D_Kinetic.slx."
        )


class TestConfigVerboseFieldDefined:
    """createSimulationConfig.m must define config.verbose so processSimulationOutput.m works."""

    def _source(self) -> str:
        return _CREATE_CONFIG.read_text(encoding="utf-8")

    def test_config_defines_verbose_field(self) -> None:
        """createSimulationConfig.m must define config.verbose (boolean) directly."""
        source = self._source()
        lines = source.splitlines()
        # The config must set config.verbose (not just config.verbosity)
        has_verbose = any(
            "config.verbose" in line and not line.strip().startswith("%")
            for line in lines
        )
        assert has_verbose, (
            "createSimulationConfig.m does not define config.verbose. "
            "processSimulationOutput.m reads config.verbose before ensureEnhancedConfig "
            "runs, so the field must be set in createSimulationConfig.m directly. "
            "Fix: add 'config.verbose = false;' alongside config.verbosity."
        )


class TestValidationNotSilentOnBadModel:
    """createSimulationConfig.m must not silently return a config with a nonexistent model."""

    def _source(self) -> str:
        return _CREATE_CONFIG.read_text(encoding="utf-8")

    def test_no_silent_warning_return_on_validation_failure(self) -> None:
        """Validation failure must not be silently swallowed with warning+return."""
        source = self._source()
        lines = source.splitlines()
        # Comments explaining the old behavior should also be removed
        comment_explaining_old = [
            line
            for line in lines
            if "return config anyway for inspection" in line.lower()
        ]
        assert not comment_explaining_old, (
            "createSimulationConfig.m still has comment 'Return config anyway for "
            "inspection/correction'. This justifies silently returning a broken config. "
            "Remove the comment and error loudly when validation fails for model_path."
        )
