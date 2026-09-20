"""TDD Acceptance Suite for Task Launch Truthfulness (ORG-03, Issue #10512).

Tests the launch truthfulness contract:
1. Problematic tiles cannot report ready/success for placeholder/example execution.
2. Missing Video Analyzer provider yields a diagnostic, not a blank working window.
3. FreeMoCap input cancellation performs no spawn; valid parameters arrive at the CLI unchanged.
4. Simulator prototype is marked inspection/demo-only and distinguished clearly from production solvers.
5. Existing working tools still launch; pending library features remain discoverable with honest explanations.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from PyQt6.QtWidgets import QMainWindow

from src.config.launcher_manifest_loader import LauncherManifest
from src.launchers.external_tools_adapter import (
    _UnavailableToolWindow,
    get_video_analyzer_dockable_ui,
)
from src.launchers.launcher_model_handlers import ModelHandlerRegistry
from src.launchers.task_launch_truthfulness import (
    AUDITED_CAPABILITY_IDS,
    LaunchDisposition,
    audit_capability_launch,
    get_launch_truthfulness_audit,
    launch_freemocap_with_validation,
)
from src.shared.python.config.model_registry import ModelRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_YAML = REPO_ROOT / "src" / "config" / "models.yaml"
MANIFEST_JSON = REPO_ROOT / "src" / "config" / "launcher_manifest.json"

pytestmark = pytest.mark.unit


# =============================================================================
# RED Acceptance Case 1: Problematic Tiles Cannot Report Ready for Placeholder
# =============================================================================


class TestProblematicTilesCannotReportReady:
    """Problematic tiles cannot report ready/success for placeholder/example execution."""

    def test_problematic_tiles_cannot_report_ready_for_placeholder_or_example_execution(
        self,
    ) -> None:
        """Audited tiles with no interactive production front-end must not claim ready/gui_ready.

        Tiles:
        - golf_simulation_suite (prototype demo only)
        - motion_capture (requires --input / --output CLI arguments)
        - swing_optimizer (library-only algorithm; no interactive GUI)
        - injury_analysis (library-only scoring; no interactive GUI)
        - pinn_pure_rigid (library-only physics mode)
        - pinn_hybrid (library-only physics mode)
        - video_analyzer (requires external tools provider)
        - canonical_core_estimation (dual-shell service preview)
        - canonical_core_comparison (dual-shell service preview)
        """
        registry = ModelRegistry(config_path=MODELS_YAML)
        manifest = LauncherManifest.load(MANIFEST_JSON)

        for cap_id in AUDITED_CAPABILITY_IDS:
            audit = get_launch_truthfulness_audit(cap_id)
            assert audit is not None, f"Audit missing for capability '{cap_id}'"

            # Check manifest tile if present
            tile = manifest.get_tile(cap_id)
            if tile is not None:
                # If library-only, it must never claim gui_ready or ready
                if audit.disposition == LaunchDisposition.LIBRARY_ONLY:
                    err_msg = f"Library-only tile '{cap_id}' claims ready status '{tile.status}'"
                    assert tile.status not in ("gui_ready", "ready"), err_msg
                elif audit.disposition == LaunchDisposition.PROTOTYPE_DEMO:
                    err_msg = f"Prototype tile '{cap_id}' claims production status '{tile.status}'"
                    assert (
                        tile.status in ("prototype", "experimental")
                        or getattr(tile, "maturity", None) == "prototype"
                    ), err_msg

            # Check models.yaml entry if present
            model = registry.get_model(cap_id)
            if model is not None and getattr(model, "launcher", None) is not None:
                launcher = model.launcher
                assert launcher is not None
                if audit.disposition == LaunchDisposition.LIBRARY_ONLY:
                    err_msg = f"Library-only model '{cap_id}' claims ready status '{launcher.status}'"
                    assert launcher.status not in ("gui_ready", "ready"), err_msg
                elif audit.disposition == LaunchDisposition.PROTOTYPE_DEMO:
                    err_msg = f"Prototype model '{cap_id}' claims production status '{launcher.status}'"
                    assert launcher.status in (
                        "prototype",
                        "experimental",
                    ), err_msg


# =============================================================================
# RED Acceptance Case 2: Video Analyzer Missing Provider Yields Diagnostic
# =============================================================================


class TestVideoAnalyzerTruthfulness:
    """Missing Video Analyzer provider yields a diagnostic, not a blank working window."""

    def test_missing_video_analyzer_provider_yields_diagnostic_not_blank_working_window(
        self, qapp: object
    ) -> None:
        """When the external Tools repository cannot be imported, get_video_analyzer_dockable_ui

        must return an explicit _UnavailableToolWindow with actionable diagnostic and remediation,
        and NEVER fall back to a blank placeholder VideoAnalyzerWindow.
        """
        # Simulate missing external tools repo / import failure
        with patch(
            "src.launchers.external_tools_adapter._ensure_tools_on_path",
            return_value=False,
        ):
            window = get_video_analyzer_dockable_ui()
            assert isinstance(window, _UnavailableToolWindow)
            assert getattr(window, "is_tool_available", True) is False
            assert "Unavailable" in window.windowTitle()

        # Simulate import failure even if path was added
        with (
            patch(
                "src.launchers.external_tools_adapter._ensure_tools_on_path",
                return_value=True,
            ),
            patch(
                "src.launchers.external_tools_adapter._import_video_analyzer",
                side_effect=ImportError(
                    "No module named 'video_analyzer.launch_pyqt6'"
                ),
            ),
        ):
            window = get_video_analyzer_dockable_ui()
            assert isinstance(window, _UnavailableToolWindow)
            assert getattr(window, "is_tool_available", True) is False


# =============================================================================
# RED Acceptance Case 3: FreeMoCap Parameter Validation & Cancellation
# =============================================================================


class TestFreeMoCapLaunchTruthfulness:
    """FreeMoCap input cancellation performs no spawn; valid parameters arrive at CLI unchanged."""

    def test_freemocap_zero_args_refuses_to_spawn(self) -> None:
        """Launching FreeMoCap with no arguments must refuse to spawn and return failure."""
        mock_proc_manager = MagicMock()
        success = launch_freemocap_with_validation(
            input_path=None,
            output_path=None,
            repo_path=REPO_ROOT,
            process_manager=mock_proc_manager,
        )
        assert success is False
        mock_proc_manager.launch_script.assert_not_called()
        mock_proc_manager.launch_module.assert_not_called()

    def test_freemocap_input_cancellation_performs_no_spawn(self) -> None:
        """When user cancels input selection dialog (input_path=None), no process is spawned."""
        mock_proc_manager = MagicMock()
        success = launch_freemocap_with_validation(
            input_path="",
            output_path="/tmp/out",
            repo_path=REPO_ROOT,
            process_manager=mock_proc_manager,
        )
        assert success is False
        mock_proc_manager.launch_script.assert_not_called()

    def test_freemocap_valid_parameters_arrive_at_cli_unchanged(self) -> None:
        """Valid input and output paths are forwarded to the subprocess call unchanged."""
        mock_proc_manager = MagicMock()
        mock_proc_manager.launch_script_with_args.return_value = MagicMock()

        success = launch_freemocap_with_validation(
            input_path="data/raw_session_videos",
            output_path="data/output_landmarks",
            repo_path=REPO_ROOT,
            process_manager=mock_proc_manager,
        )
        assert success is True
        mock_proc_manager.launch_script_with_args.assert_called_once()
        _, kwargs = mock_proc_manager.launch_script_with_args.call_args
        args = (
            kwargs.get("args")
            or mock_proc_manager.launch_script_with_args.call_args[0][2]
        )
        assert "--input" in args
        assert "data/raw_session_videos" in args
        assert "--output" in args
        assert "data/output_landmarks" in args


# =============================================================================
# RED Acceptance Case 4: Simulator Prototype Inspection / Demo-Only
# =============================================================================


class TestSimulatorPrototypeHonesty:
    """Simulator prototype is marked inspection/demo-only; clearly distinguished from production solvers."""

    def test_simulator_prototype_marked_inspection_demo_only(self) -> None:
        """golf_simulation_suite must be classified as PROTOTYPE_DEMO, must not claim

        to be a production physics engine (no engine_type='golf_simulation'), and its
        description must clearly state it is a prototype inspection/demo view.
        """
        registry = ModelRegistry(config_path=MODELS_YAML)
        manifest = LauncherManifest.load(MANIFEST_JSON)

        audit = audit_capability_launch("golf_simulation_suite")
        assert audit.disposition == LaunchDisposition.PROTOTYPE_DEMO
        assert "prototype" in audit.label.lower() or "demo" in audit.label.lower()

        model = registry.get_model("golf_simulation_suite")
        assert model is not None
        assert model.launcher is not None
        assert model.launcher.status in ("prototype", "experimental")
        # Engine type must not claim to be a qualified production physics engine
        assert getattr(model, "engine_type", None) != "golf_simulation"

        tile = manifest.get_tile("golf_simulation_suite")
        assert tile is not None
        assert (
            "prototype" in tile.description.lower()
            or "demo" in tile.description.lower()
        )


# =============================================================================
# GREEN Acceptance Case 5: Existing Tools Launch & Library Features Honest
# =============================================================================


class TestExistingToolsAndLibraryHonesty:
    """Existing working tools still launch; pending library features remain discoverable with honest explanations."""

    def test_existing_working_tools_still_launch(self) -> None:
        """Working tools like Putting Green and Biomech Exercise continue to launch normally."""
        handler_registry = ModelHandlerRegistry()

        putting_green_handler = handler_registry.get_handler("putting_green")
        assert putting_green_handler is not None

        mock_proc_manager = MagicMock()
        mock_proc_manager.launch_script.return_value = MagicMock()

        class DummyModel:
            id = "putting_green"
            name = "Putting Green"
            path = "src/tools/putting_green_gui/gui.py"

        with patch.object(Path, "exists", return_value=True):
            res = putting_green_handler.launch(
                DummyModel(), REPO_ROOT, mock_proc_manager
            )
            assert res is True
            mock_proc_manager.launch_script.assert_called_once()

    def test_pending_library_features_remain_discoverable_with_honest_explanations(
        self,
    ) -> None:
        """Library-only features (swing_optimizer, injury_analysis, pinn_*) provide

        honest status messages explaining their library status and tracking issues.
        """
        for cap_id in (
            "swing_optimizer",
            "injury_analysis",
            "pinn_pure_rigid",
            "pinn_hybrid",
        ):
            audit = audit_capability_launch(cap_id)
            assert audit.disposition == LaunchDisposition.LIBRARY_ONLY
            assert audit.next_action and len(audit.next_action) > 10
            assert "library" in audit.explanation.lower()
