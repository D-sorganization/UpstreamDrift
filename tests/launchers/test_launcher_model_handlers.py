import os  # noqa: E402

if not hasattr(os, "startfile"):
    os.startfile = lambda x: None  # type: ignore

"""Tests for launcher_model_handlers."""

from pathlib import Path  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402

from src.launchers.launcher_model_handlers import (  # noqa: E402
    BiomechExerciseHandler,
    DocumentHandler,
    GolfSimulationSuiteHandler,
    MatlabFileHandler,
    ModelHandler,
    ModelHandlerRegistry,
    ModuleHandler,
    PuttingGreenHandler,
    ScriptHandler,
    SpecialAppHandler,
    _open_with_system_app,
)
from src.config.launcher_manifest_loader import LauncherManifest
from src.launchers.model_card import MODEL_IMAGES


def test_module_handler() -> None:
    handler = ModuleHandler({"type1", "type2"}, "my_module", "My Module")
    assert handler.can_handle("type1") is True
    assert handler.can_handle("type3") is False

    mock_manager = MagicMock()
    mock_manager.launch_module.return_value = "process"

    res = handler.launch("model", Path("/repo"), mock_manager)
    assert res is True
    mock_manager.launch_module.assert_called_once_with(
        name="My Module",
        module_name="my_module",
        cwd=Path("/repo").resolve(),
        extra_python_paths=(),
    )


def test_module_handler_fail() -> None:
    handler = ModuleHandler({"type1"}, "my_module", "My Module")
    mock_manager = MagicMock()
    mock_manager.launch_module.return_value = None

    assert handler.launch("model", Path("/repo"), mock_manager) is False


def test_script_handler() -> None:
    handler = ScriptHandler({"drake"}, "script.py", "Drake", cwd_path="dir")
    assert handler.can_handle("drake") is True
    assert handler.can_handle("other") is False

    mock_manager = MagicMock()
    mock_manager.launch_script.return_value = "process"

    res = handler.launch("model", Path("/repo"), mock_manager)
    assert res is True
    mock_manager.launch_script.assert_called_once_with(
        name="Drake",
        script_path=(Path("/repo") / "script.py").resolve(),
        cwd=(Path("/repo") / "dir").resolve(),
        extra_python_paths=(),
    )


def test_script_handler_fail() -> None:
    handler = ScriptHandler({"drake"}, "script.py", "Drake")
    mock_manager = MagicMock()
    mock_manager.launch_script.return_value = None

    assert handler.launch("model", Path("/repo"), mock_manager) is False


def test_special_app_handler() -> None:
    handler = SpecialAppHandler()
    assert handler.can_handle("special_app") is True
    assert handler.can_handle("random") is False

    class DummyModel:
        path = "app.py"
        name = "App"
        id = "app_1"

    mock_manager = MagicMock()
    mock_manager.launch_script.return_value = "proc"

    with patch.object(Path, "exists", return_value=True):
        res = handler.launch(DummyModel(), Path("/repo"), mock_manager)
        assert res is True
        mock_manager.launch_script.assert_called_once()

    # Missing path
    class NoPathModel:
        id = "app_1"

    assert handler.launch(NoPathModel(), Path("/repo"), mock_manager) is False

    # Missing script file
    with patch.object(Path, "exists", return_value=False):
        assert handler.launch(DummyModel(), Path("/repo"), mock_manager) is False


def test_putting_green_handler() -> None:
    handler = PuttingGreenHandler()
    assert handler.can_handle("putting_green") is True

    class DummyModel:
        path = "simulation/green.py"
        id = "green"

    mock_manager = MagicMock()
    mock_manager.launch_script.return_value = "proc"

    with patch.object(Path, "exists", return_value=True):
        res = handler.launch(DummyModel(), Path("/repo"), mock_manager)
        assert res is True
        mock_manager.launch_script.assert_called_once()

    class NoPathModel:
        id = "g_1"

    assert handler.launch(NoPathModel(), Path("/repo"), mock_manager) is False

    with patch.object(Path, "exists", return_value=False):
        assert handler.launch(DummyModel(), Path("/repo"), mock_manager) is False


@patch("platform.system", return_value="Windows")
@patch("os.startfile")
def test_open_with_system_app_win(mock_start, mock_sys) -> None:
    assert _open_with_system_app(Path("test.txt"), "Test") is True
    mock_start.assert_called_once()


@patch("platform.system", return_value="Darwin")
@patch("subprocess.Popen")
def test_open_with_system_app_mac(mock_popen, mock_sys) -> None:
    assert _open_with_system_app(Path("test.txt"), "Test") is True
    mock_popen.assert_called_once_with(["open", "test.txt"])


@patch("platform.system", return_value="Linux")
@patch("subprocess.Popen")
def test_open_with_system_app_linux(mock_popen, mock_sys) -> None:
    assert _open_with_system_app(Path("test.txt"), "Test") is True
    mock_popen.assert_called_once_with(["xdg-open", "test.txt"])


@patch("platform.system", return_value="Linux")
@patch("subprocess.Popen", side_effect=OSError("Boom"))
def test_open_with_system_app_fail(mock_popen, mock_sys) -> None:
    assert _open_with_system_app(Path("test.txt"), "Test") is False


class DummyMatlabModel:
    path = "file.slx"
    id = "slx1"


def test_matlab_handler() -> None:
    handler = MatlabFileHandler()
    assert handler.can_handle("matlab_file") is True

    with (
        patch.object(Path, "exists", return_value=True),
        patch(
            "src.launchers.launcher_model_handlers._open_with_system_app",
            return_value=True,
        ),
    ):
        res = handler.launch(DummyMatlabModel(), Path("/repo"), MagicMock())
        assert res is True

    with patch.object(Path, "exists", return_value=False):
        assert handler.launch(DummyMatlabModel(), Path("/repo"), MagicMock()) is False

    class NoPath:
        id = "none"

    assert handler.launch(NoPath(), Path("/repo"), MagicMock()) is False


class DummyDocModel:
    path = "doc.pdf"
    id = "doc1"


def test_document_handler() -> None:
    handler = DocumentHandler()
    assert handler.can_handle("document") is True

    with (
        patch.object(Path, "exists", return_value=True),
        patch(
            "src.launchers.launcher_model_handlers._open_with_system_app",
            return_value=True,
        ),
    ):
        assert handler.launch(DummyDocModel(), Path("/repo"), MagicMock()) is True


def test_protocol_methods() -> None:
    # Only for line coverage on ... in ModelHandler
    class Concrete(ModelHandler):
        pass

    c = Concrete()  # type: ignore
    assert c.can_handle("x") is None
    assert c.launch(None, Path(""), MagicMock()) is None


def test_registry() -> None:
    registry = ModelHandlerRegistry()

    mock_handler = MagicMock()
    mock_handler.can_handle.side_effect = lambda t: t == "custom"

    registry.register_handler(mock_handler)

    assert registry.get_handler("custom") == mock_handler
    assert registry.get_handler("unknown") is None

    mock_handler.launch.return_value = True
    assert registry.launch_model("custom", "model", Path("/repo"), MagicMock()) is True

    assert (
        registry.launch_model("unknown", "model", Path("/repo"), MagicMock()) is False
    )


# ===========================================================================
# New handler registration tests
# ===========================================================================


class TestBiomechExerciseHandler:
    """Tests for BiomechExerciseHandler."""

    def test_can_handle_biomech_exercise(self) -> None:
        handler = BiomechExerciseHandler()
        assert handler.can_handle("biomech_exercise") is True

    def test_cannot_handle_other_types(self) -> None:
        handler = BiomechExerciseHandler()
        assert handler.can_handle("special_app") is False
        assert handler.can_handle("putting_green") is False
        assert handler.can_handle("random") is False

    def test_launch_success(self) -> None:
        handler = BiomechExerciseHandler()

        class DummyModel:
            path = "src/launchers/exercise_dashboard.py"
            name = "Gait Analysis"
            id = "biomech_gait"

        mock_manager = MagicMock()
        mock_manager.launch_script.return_value = "proc"
        mock_manager.get_subprocess_env.return_value = {}

        res = handler.launch(DummyModel(), Path("/repo"), mock_manager)
        assert res is True
        mock_manager.launch_script.assert_called_once()

    def test_launch_uses_hardcoded_script_path(self) -> None:
        """BiomechExerciseHandler constructs its own script path from repo_path,
        not from model.path. It always tries to launch exercise_dashboard.py."""
        handler = BiomechExerciseHandler()

        class MinimalModel:
            id = "biomech_1"

        mock_manager = MagicMock()
        mock_manager.launch_script.return_value = "proc"
        mock_manager.get_subprocess_env.return_value = {}

        res = handler.launch(MinimalModel(), Path("/repo"), mock_manager)
        assert res is True
        mock_manager.launch_script.assert_called_once()

    def test_launch_returns_false_on_process_failure(self) -> None:
        """If launch_script returns None, launch returns False."""
        handler = BiomechExerciseHandler()

        class MinimalModel:
            id = "biomech_1"

        mock_manager = MagicMock()
        mock_manager.launch_script.return_value = None
        mock_manager.get_subprocess_env.return_value = {}

        res = handler.launch(MinimalModel(), Path("/repo"), mock_manager)
        assert res is False


class TestGolfSimulationSuiteHandler:
    """Tests for GolfSimulationSuiteHandler."""

    def test_can_handle_golf_simulation(self) -> None:
        handler = GolfSimulationSuiteHandler()
        assert handler.can_handle("golf_simulation") is True

    def test_cannot_handle_other_types(self) -> None:
        handler = GolfSimulationSuiteHandler()
        assert handler.can_handle("special_app") is False
        assert handler.can_handle("biomech_exercise") is False
        assert handler.can_handle("random") is False

    def test_launch_success(self) -> None:
        handler = GolfSimulationSuiteHandler()

        class DummyModel:
            path = "launch_golf_suite.py"
            name = "Golf Sim Suite"
            id = "golf_sim"

        mock_manager = MagicMock()
        mock_manager.launch_script.return_value = "proc"

        with patch.object(Path, "exists", return_value=True):
            res = handler.launch(DummyModel(), Path("/repo"), mock_manager)
            assert res is True
            mock_manager.launch_script.assert_called_once()

    def test_launch_no_path(self) -> None:
        handler = GolfSimulationSuiteHandler()

        class NoPathModel:
            id = "gs_1"

        assert handler.launch(NoPathModel(), Path("/repo"), MagicMock()) is False

    def test_launch_missing_script(self) -> None:
        handler = GolfSimulationSuiteHandler()

        class DummyModel:
            path = "launch_golf_suite.py"
            name = "Golf Sim Suite"
            id = "golf_sim"

        with patch.object(Path, "exists", return_value=False):
            assert handler.launch(DummyModel(), Path("/repo"), MagicMock()) is False


class TestManifestTileHandlerRegistration:
    """Verify every manifest tile type has a handler in ModelHandlerRegistry."""

    @pytest.fixture
    def manifest(self) -> LauncherManifest:
        return LauncherManifest.load()

    @pytest.fixture
    def registry(self) -> ModelHandlerRegistry:
        return ModelHandlerRegistry()

    def test_all_manifest_tile_types_have_handlers(
        self, manifest: LauncherManifest, registry: ModelHandlerRegistry
    ) -> None:
        """Every tile in the manifest must have a handler that can_handle() it."""
        missing: list[str] = []
        for tile in manifest.tiles:
            handler = registry.get_handler(tile.type)
            if handler is None:
                missing.append(f"{tile.id!r} (type={tile.type!r})")
        assert not missing, (
            f"No handler registered for tiles: {', '.join(missing)}"
        )

    def test_handler_can_handle_returns_true_for_manifest_types(
        self, manifest: LauncherManifest, registry: ModelHandlerRegistry
    ) -> None:
        """Each handler's can_handle() must return True for the declared type."""
        for tile in manifest.tiles:
            handler = registry.get_handler(tile.type)
            assert handler is not None, f"No handler for tile {tile.id!r}"
            assert handler.can_handle(tile.type) is True, (
                f"Handler {type(handler).__name__}.can_handle({tile.type!r}) "
                f"returned False for tile {tile.id!r}"
            )

    def test_manifest_static_tile_names_have_model_images(self) -> None:
        """Every tile from the static manifest must have an entry in MODEL_IMAGES.
        Provider-backed tiles may not have entries yet."""
        manifest = LauncherManifest.load()
        # Static tiles that must always have MODEL_IMAGES entries
        static_ids = {
            "model_explorer", "mujoco_unified", "drake_golf", "pinocchio_golf",
            "opensim_golf", "myosim_suite", "putting_green", "matlab_unified",
            "motion_target_preview", "motion_capture", "video_analyzer",
            "video_processor", "data_explorer", "data_processor",
            "project_map", "starting_pose_matcher",
        }
        missing: list[str] = []
        for tile in manifest.tiles:
            if tile.id in static_ids and tile.name not in MODEL_IMAGES:
                missing.append(f"{tile.id!r} (name={tile.name!r})")
        assert not missing, (
            f"MODEL_IMAGES missing entries for: {', '.join(missing)}"
        )
