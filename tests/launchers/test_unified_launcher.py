"""Tests for unified_launcher.py."""

import sys  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402

from src.launchers.unified_launcher import (  # noqa: E402
    UnifiedLauncher,
    _get_golf_main,
    _is_pyqt6_available,
    launch,
    show_status,
)


@pytest.fixture
def clean_sys_modules():
    """Remove specific modules from sys.modules."""
    modules_to_remove = ["launchers.unified_launcher", "launchers.golf_launcher"]
    for mod in modules_to_remove:
        if mod in sys.modules:
            del sys.modules[mod]
    yield
    for mod in modules_to_remove:
        if mod in sys.modules:
            del sys.modules[mod]


def test_is_pyqt6_available_legacy_override():
    # Set up legacy module mock
    legacy_mock = MagicMock()
    legacy_mock.PYQT6_AVAILABLE = False

    with patch.dict("sys.modules", {"launchers.unified_launcher": legacy_mock}):
        assert _is_pyqt6_available() is False


def test_is_pyqt6_available_using_constant(clean_sys_modules):
    # Ensure no legacy module
    if "launchers.unified_launcher" in sys.modules:
        del sys.modules["launchers.unified_launcher"]

    with patch("src.launchers.unified_launcher.PYQT6_AVAILABLE", True):
        assert _is_pyqt6_available() is True


def test_get_golf_main_prefer_legacy(clean_sys_modules):
    legacy_mock = MagicMock()

    def fake_main():
        pass

    legacy_mock.main = fake_main
    sys.modules["launchers.golf_launcher"] = legacy_mock

    main_func = _get_golf_main(prefer_legacy=True)
    assert main_func is fake_main


def test_get_golf_main_absolute_import():
    import builtins

    orig_import = builtins.__import__

    mock_module = MagicMock()

    def fake_main():
        pass

    mock_module.main = fake_main

    def mock_import(name, *args, **kwargs):
        if "golf_launcher" in name and not name.startswith("launchers"):
            raise ImportError("fake")
        return orig_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=mock_import),
        patch(
            "src.launchers.unified_launcher.importlib.import_module",
            return_value=mock_module,
        ),
    ):
        main_func = _get_golf_main()
        assert main_func is fake_main


def test_get_golf_main_relative_success(clean_sys_modules):
    with (
        patch("src.launchers.unified_launcher.golf_main", create=True) as mock_main,
        patch.dict(
            "sys.modules", {"src.launchers.golf_launcher": MagicMock(main=mock_main)}
        ),
    ):
        main_func = _get_golf_main()
        assert main_func == mock_main


def test_get_golf_main_all_imports_fail(clean_sys_modules):
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if "golf_launcher" in name:
            raise ImportError("fake")
        return orig_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=mock_import),
        patch(
            "src.launchers.unified_launcher.importlib.import_module",
            side_effect=ImportError("Boom"),
        ),
        pytest.raises(ImportError),
    ):
        _get_golf_main()


def test_get_golf_main_legacy_no_main():
    legacy_mock = MagicMock()
    del legacy_mock.main

    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if "golf_launcher" in name and not name.startswith("launchers"):
            raise ImportError("fake")
        return orig_import(name, *args, **kwargs)

    mock_module = MagicMock()
    mock_module.main = "fallback"

    with (
        patch("builtins.__import__", side_effect=mock_import),
        patch(
            "src.launchers.unified_launcher.importlib.import_module",
            return_value=mock_module,
        ),
        patch.dict("sys.modules", {"launchers.golf_launcher": legacy_mock}),
    ):
        main_func = _get_golf_main(prefer_legacy=True)
        assert main_func == "fallback"


def test_get_golf_main_module_no_main():
    import builtins

    orig_import = builtins.__import__

    # We make the *relative* import fail, jumping to the absolute import
    def mock_import(name, *args, **kwargs):
        if "golf_launcher" in name and not name.startswith("launchers"):
            raise ImportError("fake")
        return orig_import(name, *args, **kwargs)

    # The absolute import succeeds but the module has no `main`
    mock_mod = MagicMock()
    del mock_mod.main
    mock_mod.__name__ = "launchers.golf_launcher"

    with (
        patch("builtins.__import__", side_effect=mock_import),
        patch(
            "src.launchers.unified_launcher.importlib.import_module",
            return_value=mock_mod,
        ),
        pytest.raises(
            ImportError
        ),  # The third fallback will fail with ImportError again
    ):
        _get_golf_main()


def test_unified_launcher_init_fails_if_no_pyqt6():
    with (
        patch("src.launchers.unified_launcher._is_pyqt6_available", return_value=False),
        pytest.raises(ImportError, match="PyQt6 is required"),
    ):
        UnifiedLauncher()


def test_pyqt6_unavailable_reload(clean_sys_modules):
    import src.launchers.unified_launcher

    path = src.launchers.unified_launcher.__file__
    with open(path, encoding="utf-8") as f:
        code = f.read()

    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "PyQt6.QtWidgets":
            raise ImportError("fake")
        return orig_import(name, *args, **kwargs)

    import contextlib

    namespace = {"__name__": "src.launchers.unified_launcher", "__file__": path}
    with (
        patch("builtins.__import__", side_effect=mock_import),
        contextlib.suppress(Exception),
    ):
        exec(compile(code, path, "exec"), namespace)

    assert namespace.get("QApplication") is None


def test_unified_launcher_mainloop():
    with (
        patch("src.launchers.unified_launcher._is_pyqt6_available", return_value=True),
        patch("src.launchers.unified_launcher._get_golf_main") as mock_get_main,
    ):
        mock_main = MagicMock()
        mock_get_main.return_value = mock_main

        launcher = UnifiedLauncher()
        launcher.mainloop()

        mock_get_main.assert_called_once_with(prefer_legacy=True)
        mock_main.assert_called_once()


def test_unified_launcher_show_status(caplog, capsys):
    with (
        patch("src.launchers.unified_launcher._is_pyqt6_available", return_value=True),
        patch(
            "src.shared.python.engine_core.engine_manager.EngineManager.get_available_engines",
            return_value=["engine_a", "engine_b"],
        ),
        patch("src.launchers.unified_launcher.Path") as mock_path,
    ):
        # Setup mock paths for launchers and engines
        mock_launcher_dir = MagicMock()
        mock_launcher_file = MagicMock()
        mock_launcher_file.name = "fake_launcher.py"
        mock_launcher_dir.glob.return_value = [mock_launcher_file]

        mock_engine_dir = MagicMock()
        mock_engine_file = MagicMock()
        mock_engine_file.is_dir.return_value = True
        mock_engine_file.name = "engine1"
        mock_engine_dir.exists.return_value = True
        mock_engine_dir.iterdir.return_value = [mock_engine_file]

        # When creating Path(__file__).parent
        mock_path_instance = MagicMock()
        mock_path_instance.parent = mock_launcher_dir
        mock_path.return_value = mock_path_instance

        # Mock SUITE_ROOT
        with patch("src.shared.python.SUITE_ROOT", mock_engine_dir, create=True):
            launcher = UnifiedLauncher()
            launcher.show_status()

            captured = capsys.readouterr()
            assert (
                "ENGINE_A" in captured.out
                or "engine_a" in captured.out
                or "ENGINE_A" in captured.out.upper()
            )
            assert (
                "ENGINE_B" in captured.out
                or "engine_b" in captured.out
                or "ENGINE_B" in captured.out.upper()
            )


def test_unified_launcher_show_status_no_engines_and_hidden_dirs(caplog, capsys):
    with (
        patch("src.launchers.unified_launcher._is_pyqt6_available", return_value=True),
        patch(
            "src.shared.python.engine_core.engine_manager.EngineManager.get_available_engines",
            return_value=[],
        ),
        patch("src.launchers.unified_launcher.Path") as mock_path,
    ):
        mock_launcher_dir = MagicMock()
        mock_launcher_file = MagicMock()
        mock_launcher_file.name = "unified_launcher.py"
        mock_launcher_dir.glob.return_value = [mock_launcher_file]

        mock_engine_dir = MagicMock()
        mock_engines = MagicMock()
        mock_engine_dir.__truediv__.return_value = mock_engines
        mock_engines.exists.return_value = True

        hidden_dir = MagicMock()
        hidden_dir.is_dir.return_value = True
        hidden_dir.name = ".hidden"

        file_not_dir = MagicMock()
        file_not_dir.is_dir.return_value = False
        file_not_dir.name = "file.txt"

        valid_dir = MagicMock()
        valid_dir.is_dir.return_value = True
        valid_dir.name = "MyEngine"

        mock_engines.iterdir.return_value = [hidden_dir, file_not_dir, valid_dir]

        mock_path_instance = MagicMock()
        mock_path_instance.parent = mock_launcher_dir
        mock_path.return_value = mock_path_instance

        with patch("src.shared.python.SUITE_ROOT", mock_engine_dir, create=True):
            launcher = UnifiedLauncher()
            launcher.show_status()

            captured = capsys.readouterr()
            assert "NO ENGINES AVAILABLE" in captured.out
            # verify it went over valid_dir
            valid_dir.is_dir.assert_called_once()
            file_not_dir.is_dir.assert_called_once()


def test_unified_launcher_get_version_metadata():
    with (
        patch("src.launchers.unified_launcher._is_pyqt6_available", return_value=True),
        patch("importlib.metadata.version", side_effect=["1.2.3"]),
    ):
        launcher = UnifiedLauncher()
        assert launcher.get_version() == "1.2.3"


def test_unified_launcher_get_version_metadata_golf_modeling_suite():
    from importlib.metadata import PackageNotFoundError

    def mock_version(pkg):
        if pkg == "upstream-drift":
            raise PackageNotFoundError
        return "1.2.4"

    with (
        patch("src.launchers.unified_launcher._is_pyqt6_available", return_value=True),
        patch("importlib.metadata.version", side_effect=mock_version),
    ):
        launcher = UnifiedLauncher()
        assert launcher.get_version() == "1.2.4"


def test_unified_launcher_get_version_fallback():
    with (
        patch("src.launchers.unified_launcher._is_pyqt6_available", return_value=True),
        patch("importlib.metadata.version", side_effect=ImportError("broken")),
    ):
        launcher = UnifiedLauncher()
        version = launcher.get_version()
        assert isinstance(version, str)
        assert len(version) > 0


def test_unified_launcher_get_version_shared_python():
    from importlib.metadata import PackageNotFoundError

    def mock_version(pkg):
        raise PackageNotFoundError

    try:
        import shared.python as _shared
    except ImportError:
        _shared = None

    with (
        patch("src.launchers.unified_launcher._is_pyqt6_available", return_value=True),
        patch("importlib.metadata.version", side_effect=mock_version),
    ):
        if _shared is not None:
            with patch("shared.python.__version__", "1.2.5", create=True):
                launcher = UnifiedLauncher()
                assert launcher.get_version() == "1.2.5"
        else:
            # If not importable, we just mock sys.modules without previous state
            mock_mod = MagicMock()
            mock_mod.__version__ = "1.2.5"
            with patch.dict(
                "sys.modules", {"shared": MagicMock(), "shared.python": mock_mod}
            ):
                launcher = UnifiedLauncher()
                assert launcher.get_version() == "1.2.5"


def test_unified_launcher_get_version_pyproject_toml():
    from importlib.metadata import PackageNotFoundError

    def mock_version(pkg):
        raise PackageNotFoundError

    with (
        patch("src.launchers.unified_launcher._is_pyqt6_available", return_value=True),
        patch("importlib.metadata.version", side_effect=mock_version),
    ):
        # Prevent finding shared.python
        if "shared.python" in sys.modules:
            del sys.modules["shared.python"]

        mock_tomllib = MagicMock()
        mock_tomllib.load.return_value = {"project": {"version": "1.2.6"}}

        with (
            patch.dict("sys.modules", {"tomllib": mock_tomllib, "shared.python": None}),
            patch("src.launchers.unified_launcher.Path") as mock_path,
        ):
            mock_file = MagicMock()
            mock_path.return_value.parent.parent.parent.__truediv__.return_value = (
                mock_file
            )
            mock_file.open.return_value.__enter__.return_value = MagicMock()

            launcher = UnifiedLauncher()
            assert launcher.get_version() == "1.2.6"


def test_unified_launcher_pyproject_toml_error():
    from importlib.metadata import PackageNotFoundError

    def mock_version(pkg):
        raise PackageNotFoundError

    with (
        patch("src.launchers.unified_launcher._is_pyqt6_available", return_value=True),
        patch("importlib.metadata.version", side_effect=mock_version),
    ):
        if "shared.python" in sys.modules:
            del sys.modules["shared.python"]

        mock_tomllib = MagicMock()
        mock_tomllib.load.side_effect = OSError("Boom")

        with (
            patch.dict("sys.modules", {"tomllib": mock_tomllib, "shared.python": None}),
            patch("src.launchers.unified_launcher.Path") as mock_path,
        ):
            mock_file = MagicMock()
            mock_path.return_value.parent.parent.parent.__truediv__.return_value = (
                mock_file
            )
            mock_file.open.return_value.__enter__.return_value = MagicMock()
            launcher = UnifiedLauncher()
            assert launcher.get_version() == "1.0.0-beta"


def test_unified_launcher_get_version_hardcoded():
    from importlib.metadata import PackageNotFoundError

    def mock_version(pkg):
        raise PackageNotFoundError

    with (
        patch("src.launchers.unified_launcher._is_pyqt6_available", return_value=True),
        patch("importlib.metadata.version", side_effect=mock_version),
    ):
        if "shared.python" in sys.modules:
            del sys.modules["shared.python"]

        with patch.dict(
            "sys.modules", {"tomllib": None, "tomli": None, "shared.python": None}
        ):
            launcher = UnifiedLauncher()
            assert launcher.get_version() == "1.0.0-beta"


def test_launch():
    with (
        patch("src.launchers.unified_launcher._is_pyqt6_available", return_value=True),
        patch("src.launchers.unified_launcher._get_golf_main") as mock_get_main,
    ):
        mock_main = MagicMock()
        mock_get_main.return_value = mock_main

        launch()
        mock_get_main.assert_called_once_with(prefer_legacy=False)
        mock_main.assert_called_once()


def test_launch_no_pyqt6():
    with (
        patch("src.launchers.unified_launcher._is_pyqt6_available", return_value=False),
        patch("src.launchers.unified_launcher._get_golf_main") as mock_get_main,
    ):
        launch()
        mock_get_main.assert_not_called()


def test_show_status_fn():
    with patch("src.launchers.unified_launcher.UnifiedLauncher.show_status") as mock_ss:
        show_status()
        mock_ss.assert_called_once()
