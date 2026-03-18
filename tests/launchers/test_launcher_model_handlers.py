"""Tests for launcher_model_handlers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.launchers.launcher_model_handlers import (
    DocumentHandler,
    MatlabFileHandler,
    ModelHandler,
    ModelHandlerRegistry,
    ModuleHandler,
    PuttingGreenHandler,
    ScriptHandler,
    SpecialAppHandler,
    _open_with_system_app,
)


def test_module_handler():
    handler = ModuleHandler({"type1", "type2"}, "my_module", "My Module")
    assert handler.can_handle("type1") is True
    assert handler.can_handle("type3") is False

    mock_manager = MagicMock()
    mock_manager.launch_module.return_value = "process"

    res = handler.launch("model", Path("/repo"), mock_manager)
    assert res is True
    mock_manager.launch_module.assert_called_once_with(
        name="My Module", module_name="my_module", cwd=Path("/repo")
    )


def test_module_handler_fail():
    handler = ModuleHandler({"type1"}, "my_module", "My Module")
    mock_manager = MagicMock()
    mock_manager.launch_module.return_value = None

    assert handler.launch("model", Path("/repo"), mock_manager) is False


def test_script_handler():
    handler = ScriptHandler({"drake"}, "script.py", "Drake", cwd_path="dir")
    assert handler.can_handle("drake") is True
    assert handler.can_handle("other") is False

    mock_manager = MagicMock()
    mock_manager.launch_script.return_value = "process"

    res = handler.launch("model", Path("/repo"), mock_manager)
    assert res is True
    mock_manager.launch_script.assert_called_once_with(
        name="Drake",
        script_path=Path("/repo/script.py"),
        cwd=Path("/repo/dir"),
    )


def test_script_handler_fail():
    handler = ScriptHandler({"drake"}, "script.py", "Drake")
    mock_manager = MagicMock()
    mock_manager.launch_script.return_value = None

    assert handler.launch("model", Path("/repo"), mock_manager) is False


def test_special_app_handler():
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


def test_putting_green_handler():
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
def test_open_with_system_app_win(mock_start, mock_sys):
    assert _open_with_system_app(Path("test.txt"), "Test") is True
    mock_start.assert_called_once()


@patch("platform.system", return_value="Darwin")
@patch("subprocess.Popen")
def test_open_with_system_app_mac(mock_popen, mock_sys):
    assert _open_with_system_app(Path("test.txt"), "Test") is True
    mock_popen.assert_called_once_with(["open", "test.txt"])


@patch("platform.system", return_value="Linux")
@patch("subprocess.Popen")
def test_open_with_system_app_linux(mock_popen, mock_sys):
    assert _open_with_system_app(Path("test.txt"), "Test") is True
    mock_popen.assert_called_once_with(["xdg-open", "test.txt"])


@patch("platform.system", return_value="Linux")
@patch("subprocess.Popen", side_effect=OSError("Boom"))
def test_open_with_system_app_fail(mock_popen, mock_sys):
    assert _open_with_system_app(Path("test.txt"), "Test") is False


class DummyMatlabModel:
    path = "file.slx"
    id = "slx1"


def test_matlab_handler():
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


def test_document_handler():
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


def test_protocol_methods():
    # Only for line coverage on ... in ModelHandler
    class Concrete(ModelHandler):
        pass

    c = Concrete()
    assert c.can_handle("x") is None
    assert c.launch(None, Path(""), MagicMock()) is None


def test_registry():
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
