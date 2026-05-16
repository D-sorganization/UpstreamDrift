"""Tests for launcher diagnostics fixes — issues #5476 and #5474.

RED → GREEN cycle:
  - #5476: EXPECTED_TILE_IDS must match models.yaml, not hard-coded stale list
  - #5474: Diagnostic checks must emit app-state events via StateLogger
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestExpectedTileIdsMatchesModelsYaml:
    """#5476 — EXPECTED_TILE_IDS must be derived from models.yaml."""

    def test_expected_tile_ids_is_not_hard_coded_stale_list(self) -> None:
        """EXPECTED_TILE_IDS must not contain removed IDs like 'simscape_2d'."""
        from src.launchers.launcher_diagnostics import LauncherDiagnostics

        stale_ids = {"simscape_2d", "simscape_3d", "dataset_generator", "matlab_analysis"}
        present_stale = stale_ids & set(LauncherDiagnostics.EXPECTED_TILE_IDS)
        assert not present_stale, (
            f"EXPECTED_TILE_IDS still contains stale IDs: {present_stale}"
        )

    def test_expected_tile_ids_contains_current_tiles(self) -> None:
        """EXPECTED_TILE_IDS must include currently-shipped tiles."""
        from src.launchers.launcher_diagnostics import LauncherDiagnostics
        from src.shared.python.config.model_registry import ModelRegistry
        from src.shared.python.data_io.path_utils import get_repo_root

        yaml_path = get_repo_root() / "src" / "config" / "models.yaml"
        registry = ModelRegistry(yaml_path)
        yaml_ids = {m.id for m in registry.get_all_models()}

        missing_from_expected = yaml_ids - set(LauncherDiagnostics.EXPECTED_TILE_IDS)
        assert not missing_from_expected, (
            f"EXPECTED_TILE_IDS is missing current tiles: {missing_from_expected}"
        )

    def test_expected_tile_ids_equals_models_yaml_ids(self) -> None:
        """EXPECTED_TILE_IDS must exactly match the set of IDs in models.yaml."""
        from src.launchers.launcher_diagnostics import LauncherDiagnostics
        from src.shared.python.config.model_registry import ModelRegistry
        from src.shared.python.data_io.path_utils import get_repo_root

        yaml_path = get_repo_root() / "src" / "config" / "models.yaml"
        registry = ModelRegistry(yaml_path)
        yaml_ids = {m.id for m in registry.get_all_models()}

        assert set(LauncherDiagnostics.EXPECTED_TILE_IDS) == yaml_ids, (
            "EXPECTED_TILE_IDS does not match models.yaml. "
            f"Extra: {set(LauncherDiagnostics.EXPECTED_TILE_IDS) - yaml_ids}, "
            f"Missing: {yaml_ids - set(LauncherDiagnostics.EXPECTED_TILE_IDS)}"
        )


class TestCheckModelsYamlReturnsPass:
    """#5476 — _check_models_yaml_completeness must pass when YAML matches."""

    def test_check_models_yaml_returns_pass(self) -> None:
        """After fix, check_models_yaml must not always return fail."""
        from src.launchers.launcher_diagnostics import LauncherDiagnostics

        diag = LauncherDiagnostics()
        result = diag.check_models_yaml()
        # Should not fail because EXPECTED_TILE_IDS now matches the YAML
        assert result.status != "fail", (
            f"check_models_yaml still returns fail: {result.message}\n"
            f"details: {result.details}"
        )

    def test_completeness_passes_when_expected_matches_actual(self) -> None:
        """_check_models_yaml_completeness returns pass when sets are equal."""
        from src.launchers.launcher_diagnostics import LauncherDiagnostics

        diag = LauncherDiagnostics()
        # Build a fake models list that exactly matches EXPECTED_TILE_IDS
        fake_models = [{"id": tid} for tid in LauncherDiagnostics.EXPECTED_TILE_IDS]
        result = diag._check_models_yaml_completeness(fake_models, {})
        assert result.status == "pass", (
            f"Expected pass, got {result.status}: {result.message}"
        )

    def test_completeness_fails_when_expected_not_in_actual(self) -> None:
        """_check_models_yaml_completeness returns fail when IDs are missing."""
        from src.launchers.launcher_diagnostics import LauncherDiagnostics

        diag = LauncherDiagnostics()
        # Only supply a subset of the expected tiles
        partial_models = [{"id": LauncherDiagnostics.EXPECTED_TILE_IDS[0]}]
        result = diag._check_models_yaml_completeness(partial_models, {})
        assert result.status == "fail"


class TestDiagnosticEmitsAppStateEvents:
    """#5474 — Each diagnostic check must emit a StateLogger event."""

    def test_run_all_checks_emits_events(self) -> None:
        """run_all_checks must populate the StateLogger with at least one event."""
        import sys

        # Mock heavy Qt/engine imports so tests stay headless-safe
        mock_yaml_result = {
            "models": [
                {"id": "mujoco_unified"},
            ]
        }

        with (
            patch("src.launchers.launcher_diagnostics.open", create=True),
            patch.dict(
                "sys.modules",
                {
                    "yaml": MagicMock(safe_load=MagicMock(return_value=mock_yaml_result)),
                },
                clear=False,
            ),
        ):
            from src.shared.python.app_state import get_state_logger

            logger_singleton = get_state_logger()
            initial_len = len(logger_singleton.store)

            from src.launchers.launcher_diagnostics import LauncherDiagnostics

            diag = LauncherDiagnostics()
            # Run only the models-yaml check which is headless-safe
            diag.check_models_yaml()

            assert len(logger_singleton.store) > initial_len, (
                "check_models_yaml did not emit any app-state events"
            )

    def test_each_check_method_emits_diagnostic_event(self) -> None:
        """check_python_environment must emit a diagnostic event."""
        from src.shared.python.app_state import get_state_logger

        logger_singleton = get_state_logger()
        before_len = len(logger_singleton.store)

        from src.launchers.launcher_diagnostics import LauncherDiagnostics

        diag = LauncherDiagnostics()
        diag.check_python_environment()

        after_len = len(logger_singleton.store)
        assert after_len > before_len, (
            "check_python_environment did not emit a diagnostic event to StateLogger"
        )

    def test_emitted_event_has_diagnostic_type(self) -> None:
        """Emitted events must have a 'diagnostic_check' type and a 'check_name' payload."""
        from src.shared.python.app_state import get_state_logger

        logger_singleton = get_state_logger()

        from src.launchers.launcher_diagnostics import LauncherDiagnostics

        diag = LauncherDiagnostics()
        diag.check_python_environment()

        snapshot = logger_singleton.store.snapshot()
        diagnostic_events = [e for e in snapshot if e.type == "diagnostic_check"]
        assert diagnostic_events, "No 'diagnostic_check' events found in StateLogger"

        last = diagnostic_events[-1]
        assert "check_name" in last.payload, (
            f"Event payload missing 'check_name': {last.payload}"
        )
        assert "status" in last.payload, (
            f"Event payload missing 'status': {last.payload}"
        )
