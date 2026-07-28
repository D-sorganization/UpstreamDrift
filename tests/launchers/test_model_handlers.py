"""TDD Tests for Model Launch Handlers.

Tests the SpecialAppHandler and PuttingGreenHandler to ensure
model_explorer and putting_green tiles can be launched correctly.
"""

from __future__ import annotations  # noqa: E402

from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402
from src.launchers.launcher_model_handlers import (  # noqa: E402
    ModelHandlerRegistry,
    PuttingGreenHandler,
    SpecialAppHandler,
    _package_main_module_name,
)
from src.launchers.launcher_process_manager import ProcessManager  # noqa: E402
from src.shared.python.launcher_embed import (  # noqa: E402
    EmbedCapabilities,
    register_embeddable_tool,
    unregister_embeddable_tool,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockModel:
    """Minimal mock model matching the ModelConfig protocol."""

    id: str
    name: str
    path: str
    type: str
    source_root: str | None = None
    working_dir: str | None = None
    python_paths: tuple[str, ...] = ()


@pytest.fixture
def repo_path(tmp_path: Path) -> Path:
    """Create a temporary repo root with mock scripts."""
    # Create mock script files
    scripts = [
        "src/tools/urdf_generator/launch_urdf_generator.py",
        "src/tools/putting_green_gui/gui.py",
    ]
    for script in scripts:
        script_path = tmp_path / script
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("# mock script\n")
    return tmp_path


@pytest.fixture
def process_manager() -> MagicMock:
    """Mock ProcessManager that returns a mock process."""
    pm = MagicMock()
    pm.launch_script.return_value = MagicMock()
    pm.launch_module.return_value = MagicMock()
    return pm


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_path", "expected_module"),
    [
        ("src/tools/canonical_core/__main__.py", "src.tools.canonical_core"),
        (
            "src/tools/golf_simulation_suite/__main__.py",
            "src.tools.golf_simulation_suite",
        ),
        ("src/tools/pose_studio/__main__.py", "src.tools.pose_studio"),
        (
            "src/tools/pose_subscriber_demo/__main__.py",
            "src.tools.pose_subscriber_demo",
        ),
        ("src/tools/sg_optimizer/__main__.py", "src.tools.sg_optimizer"),
        (
            "src/tools/simulation_backends_launcher/__main__.py",
            "src.tools.simulation_backends_launcher",
        ),
        (
            "src/tools/starting_pose_matcher/__main__.py",
            "src.tools.starting_pose_matcher",
        ),
        (
            "src/tools/training_controller/__main__.py",
            "src.tools.training_controller",
        ),
    ],
)
def test_package_main_preserves_local_source_namespace(
    model_path: str, expected_module: str
) -> None:
    """Local package tiles must remain under ``src`` on the launch path."""
    assert (
        _package_main_module_name(
            model_path,
            source_root=REPO_ROOT,
            launcher_root=REPO_ROOT,
        )
        == expected_module
    )


@pytest.mark.unit
def test_local_tool_package_mains_resolve_with_launcher_pythonpath() -> None:
    """The launcher PYTHONPATH resolves local tool entry points, not root ``tools``.

    ``ProcessManager`` deliberately puts the repository root before ``src``.
    The repository also has a root-level ``tools`` package, so local package
    tiles have to use their fully qualified ``src.tools`` module names.
    """
    model_paths = sorted(REPO_ROOT.glob("src/tools/**/__main__.py"))
    module_names = [
        _package_main_module_name(
            path.relative_to(REPO_ROOT).as_posix(),
            source_root=REPO_ROOT,
            launcher_root=REPO_ROOT,
        )
        for path in model_paths
    ]
    assert all(module_names)

    environment = ProcessManager(REPO_ROOT).get_subprocess_env()
    probe = (
        "import importlib.util, sys; "
        f"modules = {module_names!r}; "
        "missing = [name for name in modules "
        "if importlib.util.find_spec(name) is None]; "
        "sys.exit(f'Unresolvable launcher modules: {missing}' if missing else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


# =============================================================================
# SpecialAppHandler Tests
# =============================================================================


class TestSpecialAppHandler:
    """Test SpecialAppHandler for special_app model types."""

    def test_can_handle_special_app(self) -> None:
        """Handler accepts 'special_app' type."""
        handler = SpecialAppHandler()
        assert handler.can_handle("special_app")

    def test_cannot_handle_physics_engine(self) -> None:
        """Handler rejects physics engine types."""
        handler = SpecialAppHandler()
        assert not handler.can_handle("mujoco")
        assert not handler.can_handle("drake")
        assert not handler.can_handle("pinocchio")

    def test_launch_model_explorer(
        self, repo_path: Path, process_manager: MagicMock
    ) -> None:
        """Model explorer tile launches successfully."""
        model = MockModel(
            id="model_explorer",
            name="Model Explorer",
            path="src/tools/urdf_generator/launch_urdf_generator.py",
            type="special_app",
        )
        handler = SpecialAppHandler()
        result = handler.launch(model, repo_path, process_manager)

        assert result is True
        process_manager.launch_script.assert_called_once()

    def test_launch_fails_for_missing_path(
        self, repo_path: Path, process_manager: MagicMock
    ) -> None:
        """DBC: Launch fails when model has no path."""
        model = MockModel(id="bad", name="Bad", path="", type="special_app")
        handler = SpecialAppHandler()
        result = handler.launch(model, repo_path, process_manager)

        assert result is False
        process_manager.launch_script.assert_not_called()

    def test_launch_fails_for_missing_script(
        self, repo_path: Path, process_manager: MagicMock
    ) -> None:
        """DBC: Launch fails when script file doesn't exist."""
        model = MockModel(
            id="missing",
            name="Missing",
            path="src/does_not_exist.py",
            type="special_app",
        )
        handler = SpecialAppHandler()
        result = handler.launch(model, repo_path, process_manager)

        assert result is False
        process_manager.launch_script.assert_not_called()

    def test_launches_package_main_module_for_sibling_tool(
        self, tmp_path: Path, process_manager: MagicMock
    ) -> None:
        """Package entry points must use ``python -m`` so relative imports work."""
        repo_path = tmp_path / "UpstreamDrift"
        repo_path.mkdir()
        optimizer_root = tmp_path / "Movement_Optimizer"
        script_path = optimizer_root / "src" / "movement_optimizer" / "__main__.py"
        script_path.parent.mkdir(parents=True)
        script_path.write_text("pass\n", encoding="utf-8")
        model = MockModel(
            id="movement_optimizer",
            name="Movement Optimizer",
            path="src/movement_optimizer/__main__.py",
            type="special_app",
            source_root="../Movement_Optimizer",
            python_paths=("src",),
        )

        assert SpecialAppHandler().launch(model, repo_path, process_manager) is True

        process_manager.launch_module.assert_called_once_with(
            name="Movement Optimizer",
            module_name="movement_optimizer",
            cwd=optimizer_root.resolve(),
            extra_python_paths=((optimizer_root / "src").resolve(),),
            confirm_startup=True,
        )
        process_manager.launch_script.assert_not_called()

    def test_dockable_probe_does_not_import_package_main_entrypoint(
        self, tmp_path: Path
    ) -> None:
        """A package ``__main__`` must not be executed in the launcher process."""
        repo_path = tmp_path / "UpstreamDrift"
        repo_path.mkdir()
        optimizer_root = tmp_path / "Movement_Optimizer"
        entrypoint = optimizer_root / "src" / "movement_optimizer" / "__main__.py"
        entrypoint.parent.mkdir(parents=True)
        marker = tmp_path / "entrypoint-imported"
        entrypoint.write_text(
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).touch()\n"
            "raise SystemExit(1)\n",
            encoding="utf-8",
        )
        model = MockModel(
            id="movement_optimizer",
            name="Movement Optimizer",
            path="src/movement_optimizer/__main__.py",
            type="special_app",
            source_root="../Movement_Optimizer",
            python_paths=("src",),
        )

        assert SpecialAppHandler().get_dockable_ui(model, repo_path) is None
        assert not marker.exists()

    def test_dockable_probe_uses_registered_embeddable_tool(
        self, tmp_path: Path
    ) -> None:
        """A registered tool opens in the launcher instead of running its adapter."""
        model = MockModel(
            id="config_setup_wizard",
            name="Setup Wizard",
            path="src/tools/config_setup_wizard/_embed_adapter.py",
            type="special_app",
        )
        widget = object()
        tool = MagicMock()
        tool.create_main_widget.return_value = widget

        with patch(
            "src.shared.python.launcher_embed.registry.get_embeddable_tool",
            return_value=tool,
        ):
            result = SpecialAppHandler().get_dockable_ui(model, tmp_path)

        assert result is widget
        tool.create_main_widget.assert_called_once_with(None)

    def test_registered_embeddable_tool_receives_parent_argument(
        self,
        tmp_path: Path,
    ) -> None:
        """Special-app docking must honor the EmbeddableTool parent contract."""

        class ParentRecordingTool:
            tool_id = "registered_parent_tool"

            def __init__(self) -> None:
                self.parent_seen = object()
                self.widget = object()

            def embed_capabilities(self) -> EmbedCapabilities:
                return EmbedCapabilities(supports_embedded=True)

            def create_main_widget(self, parent: object | None) -> object:
                self.parent_seen = parent
                return self.widget

            def cleanup(self) -> None:
                pass

            def is_dirty(self) -> bool:
                return False

        model = MockModel(
            id="registered_parent_tool",
            name="Registered Parent Tool",
            path="src/tools/registered_parent_tool/gui.py",
            type="special_app",
        )
        tool = ParentRecordingTool()
        register_embeddable_tool(tool)
        try:
            result = SpecialAppHandler().get_dockable_ui(model, tmp_path)
        finally:
            unregister_embeddable_tool(tool.tool_id)

        assert result is tool.widget
        assert tool.parent_seen is None


# =============================================================================
# PuttingGreenHandler Tests
# =============================================================================


class TestPuttingGreenHandler:
    """Test PuttingGreenHandler for putting_green model type."""

    def test_can_handle_putting_green(self) -> None:
        """Handler accepts 'putting_green' type."""
        handler = PuttingGreenHandler()
        assert handler.can_handle("putting_green")

    def test_cannot_handle_other_types(self) -> None:
        """Handler rejects non-putting-green types."""
        handler = PuttingGreenHandler()
        assert not handler.can_handle("special_app")
        assert not handler.can_handle("mujoco")

    def test_launch_putting_green(
        self, repo_path: Path, process_manager: MagicMock
    ) -> None:
        """Putting green simulator launches successfully."""
        model = MockModel(
            id="putting_green",
            name="Putting Green",
            path="src/tools/putting_green_gui/gui.py",
            type="putting_green",
        )
        handler = PuttingGreenHandler()
        result = handler.launch(model, repo_path, process_manager)

        assert result is True
        process_manager.launch_script.assert_called_once()
        call_kwargs = process_manager.launch_script.call_args
        assert call_kwargs.kwargs["name"] == "Putting Green Simulator"

    def test_launch_fails_no_path(
        self, repo_path: Path, process_manager: MagicMock
    ) -> None:
        """DBC: Launch fails when model has no path."""
        model = MockModel(id="pg", name="PG", path="", type="putting_green")
        handler = PuttingGreenHandler()
        result = handler.launch(model, repo_path, process_manager)

        assert result is False


# =============================================================================
# Registry Integration Tests
# =============================================================================


class TestRegistryIntegration:
    """Test that the ModelHandlerRegistry includes all handlers."""

    def test_registry_handles_special_app(self) -> None:
        """Registry finds handler for special_app type."""
        registry = ModelHandlerRegistry()
        handler = registry.get_handler("special_app")
        assert handler is not None
        assert isinstance(handler, SpecialAppHandler)

    def test_registry_handles_putting_green(self) -> None:
        """Registry finds handler for putting_green type."""
        registry = ModelHandlerRegistry()
        handler = registry.get_handler("putting_green")
        assert handler is not None
        assert isinstance(handler, PuttingGreenHandler)

    def test_registry_still_handles_mujoco(self) -> None:
        """Regression: existing handlers still work."""
        registry = ModelHandlerRegistry()
        assert registry.get_handler("custom_humanoid") is not None

    def test_registry_still_handles_drake(self) -> None:
        """Regression: Drake handler still works."""
        registry = ModelHandlerRegistry()
        assert registry.get_handler("drake") is not None

    def test_registry_still_handles_pinocchio(self) -> None:
        """Regression: Pinocchio handler still works."""
        registry = ModelHandlerRegistry()
        assert registry.get_handler("pinocchio") is not None

    def test_registry_still_handles_opensim(self) -> None:
        """Regression: OpenSim handler still works."""
        registry = ModelHandlerRegistry()
        assert registry.get_handler("opensim") is not None

    def test_registry_still_handles_myosim(self) -> None:
        """Regression: MyoSim handler still works."""
        registry = ModelHandlerRegistry()
        assert registry.get_handler("myosim") is not None

    def test_registry_returns_none_for_unknown(self) -> None:
        """Registry returns None for unknown types."""
        registry = ModelHandlerRegistry()
        assert registry.get_handler("nonexistent") is None
