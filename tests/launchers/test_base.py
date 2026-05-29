import os  # noqa: E402

if not hasattr(os, "startfile"):
    os.startfile = lambda x: None  # type: ignore

"""Tests for base launcher."""

from pathlib import Path  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402
from src.launchers.base import BaseLauncher, LaunchItem, run_launcher  # noqa: E402


class DummyLauncher(BaseLauncher):
    def get_items(self) -> list[LaunchItem]:
        return [
            LaunchItem(name="Item1", description="Desc1", path="path1"),
            LaunchItem(
                name="Item2", description="Desc2", action=lambda: print("Action")
            ),
            LaunchItem(name="Item3", description="Desc3"),
        ]


def test_launch_item_init() -> None:
    item = LaunchItem(
        name="Test",
        description="Desc",
        path="tests/launchers",
        item_type="app",
        icon="icon.png",
    )
    assert item.name == "Test"
    assert item.description == "Desc"
    assert item.path == "tests/launchers"
    assert item.item_type == "app"
    assert item.icon == "icon.png"


def test_launch_item_get_full_path() -> None:
    item = LaunchItem(name="Test", description="Desc", path="tests")
    assert item.get_full_path() is not None
    assert item.get_full_path().name == "tests"  # type: ignore[union-attr]

    item_no_path = LaunchItem(name="NoPath", description="Desc")
    assert item_no_path.get_full_path() is None


@pytest.fixture
def launcher(qapp) -> DummyLauncher:
    with patch("src.launchers.base.BaseLauncher.center_window"):
        return DummyLauncher()


def test_base_launcher_init(launcher) -> None:
    assert launcher.windowTitle() == DummyLauncher.WINDOW_TITLE
    assert launcher.width() == DummyLauncher.WINDOW_WIDTH
    assert launcher.height() == DummyLauncher.WINDOW_HEIGHT


def test_base_launcher_center_window(qapp) -> None:
    launcher = DummyLauncher()
    # It should not crash, it relies on QApplication.primaryScreen()
    launcher.center_window()

    # Test case where screen is None
    with patch("src.launchers.base.QApplication.primaryScreen", return_value=None):
        launcher.center_window()


@patch("src.launchers.base.QMessageBox.critical")
def test_base_launcher_show_error(mock_msg, launcher) -> None:
    launcher.show_error("Title", "Message")
    mock_msg.assert_called_once_with(launcher, "Title", "Message")


@patch("src.launchers.base.QMessageBox.warning")
def test_base_launcher_show_warning(mock_msg, launcher) -> None:
    launcher.show_warning("Title", "Message")
    mock_msg.assert_called_once_with(launcher, "Title", "Message")


@patch("src.launchers.base.QMessageBox.information")
def test_base_launcher_show_info(mock_msg, launcher) -> None:
    launcher.show_info("Title", "Message")
    mock_msg.assert_called_once_with(launcher, "Title", "Message")


@patch("src.launchers.base.Path.exists", return_value=False)
def test_base_launcher_launch_file_not_found(mock_exists, launcher) -> None:
    with patch.object(launcher, "show_error") as mock_err:
        assert launcher.launch_file("missing.txt") is False
        mock_err.assert_called_once()


@patch("src.launchers.base.Path.exists", return_value=True)
def test_base_launcher_launch_file_success_win(mock_exists, launcher) -> None:
    # Pass str
    with patch("sys.platform", "win32"), patch("os.startfile") as mock_start:
        assert launcher.launch_file("C:/absolute/path.txt") is True
        mock_start.assert_called_once()

    # Pass Path
    with patch("sys.platform", "win32"), patch("os.startfile") as mock_start:
        assert launcher.launch_file(Path("C:/absolute/path.txt")) is True
        mock_start.assert_called_once()


@patch("src.launchers.base.Path.exists", return_value=True)
def test_base_launcher_launch_file_success_mac(mock_exists, launcher) -> None:
    with patch("sys.platform", "darwin"), patch("subprocess.run") as mock_run:
        assert launcher.launch_file("path.txt") is True
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0][0] == "open"


@patch("src.launchers.base.Path.exists", return_value=True)
def test_base_launcher_launch_file_success_linux(mock_exists, launcher) -> None:
    with patch("sys.platform", "linux"), patch("subprocess.run") as mock_run:
        assert launcher.launch_file("path.txt") is True
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0][0] == "xdg-open"


@patch("src.launchers.base.Path.exists", return_value=True)
@patch("subprocess.run", side_effect=OSError("Boom"))
def test_base_launcher_launch_file_failure(mock_run, mock_exists, launcher) -> None:
    with (
        patch("sys.platform", "linux"),
        patch.object(launcher, "show_error") as mock_err,
    ):
        assert launcher.launch_file("path.txt") is False
        mock_err.assert_called_once()


def test_base_launcher_create_card_widget(launcher) -> None:
    item = LaunchItem(name="Test", description="Desc", path="path")
    card = launcher.create_card_widget(item)
    assert card is not None


def test_base_launcher_on_item_launch_action(launcher) -> None:
    mock_action = MagicMock()
    item = LaunchItem(name="Test", description="Desc", action=mock_action)
    launcher._on_item_launch(item)
    mock_action.assert_called_once()


def test_base_launcher_on_item_launch_action_fails(launcher) -> None:
    mock_action = MagicMock(side_effect=RuntimeError("Boom"))
    item = LaunchItem(name="Test", description="Desc", action=mock_action)
    with patch.object(launcher, "show_error") as mock_err:
        launcher._on_item_launch(item)
        mock_err.assert_called_once()


def test_base_launcher_on_item_launch_path(launcher) -> None:
    item = LaunchItem(name="Test", description="Desc", path="path")
    with patch.object(launcher, "launch_file") as mock_launch:
        launcher._on_item_launch(item)
        mock_launch.assert_called_once_with("path")


def test_base_launcher_on_item_launch_none(launcher) -> None:
    item = LaunchItem(name="Test", description="Desc")
    with patch.object(launcher, "show_warning") as mock_warn:
        launcher._on_item_launch(item)
        mock_warn.assert_called_once()


def test_base_launcher_build_grid_layout(launcher) -> None:
    items = [
        LaunchItem(name="A", description="A"),
        LaunchItem(name="B", description="B"),
    ]
    grid = launcher.build_grid_layout(items, columns=2)
    assert grid.count() == 2


def test_base_launcher_create_header(launcher) -> None:
    layout = launcher.create_header("Title", "Subtitle")
    assert layout.count() == 2

    layout_no_sub = launcher.create_header("Title")
    assert layout_no_sub.count() == 1


def test_base_launcher_create_separator(launcher) -> None:
    sep = launcher.create_separator()
    assert sep is not None


def test_base_launcher_init_ui(launcher) -> None:
    launcher.init_ui()
    assert len(launcher._items) == 3


def test_run_launcher() -> None:
    app_instance = MagicMock()
    app_instance.exec.return_value = 0

    with (
        patch("src.launchers.base.QApplication") as mock_qapp1,
        patch("launchers.base.QApplication", create=True) as mock_qapp2,
        patch("PyQt6.QtWidgets.QApplication.exec", return_value=0),
        patch.object(DummyLauncher, "init_ui"),
        patch.object(DummyLauncher, "center_window"),
        patch.object(DummyLauncher, "show"),
    ):
        mock_qapp1.return_value = app_instance
        if mock_qapp2 is not None:
            mock_qapp2.return_value = app_instance

        res = run_launcher(DummyLauncher)
        assert res == 0


def test_base_launcher_abstract_method(launcher) -> None:
    # Test the abstractmethod just for coverage of the '...' body
    assert BaseLauncher.get_items(launcher) is None
