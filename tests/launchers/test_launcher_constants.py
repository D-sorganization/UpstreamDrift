import importlib  # noqa: E402
from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402

from src.launchers.launcher_constants import (  # noqa: E402
    _lazy_imports,
    _lazy_load_engine_manager,
    _lazy_load_model_registry,
    validate_docker_stage,
)


def test_validate_docker_stage():
    """Test Docker stage validation."""
    assert validate_docker_stage("all") == "all"
    assert validate_docker_stage("mujoco") == "mujoco"

    with pytest.raises(ValueError, match="Invalid Docker stage 'invalid'"):
        validate_docker_stage("invalid")


def test_lazy_load_engine_manager():
    """Test lazy loading of engine manager."""
    # Reset lazy imports for testing
    _lazy_imports["EngineManager"] = None
    _lazy_imports["EngineType"] = None

    em, et = _lazy_load_engine_manager()
    assert em is not None
    assert et is not None

    # Second load should use cached
    with patch("importlib.import_module") as mock_import:
        em2, et2 = _lazy_load_engine_manager()
        assert em2 is em
        assert et2 is et
        mock_import.assert_not_called()


def test_lazy_load_model_registry():
    """Test lazy loading of model registry."""
    # Reset lazy imports for testing
    _lazy_imports["ModelRegistry"] = None

    mr = _lazy_load_model_registry()
    assert mr is not None

    # Second load should use cached
    with patch("importlib.import_module") as mock_import:
        mr2 = _lazy_load_model_registry()
        assert mr2 is mr
        mock_import.assert_not_called()


def test_platform_constants_non_win32():
    """Test platform constants on non-windows."""
    with patch("sys.platform", "linux"):
        import src.launchers.launcher_constants as lc

        importlib.reload(lc)
        assert lc.CREATE_NO_WINDOW == 0
        assert lc.CREATE_NEW_CONSOLE == 0


def test_platform_constants_win32_no_subprocess_attrs():
    """Test platform constants on win32 when subprocess lacks attributes."""
    from unittest.mock import MagicMock

    import src.launchers.launcher_constants as lc

    # Create a mock subprocess module that lacks the CREATE_NO_WINDOW attribute
    mock_subprocess = MagicMock()
    del mock_subprocess.CREATE_NO_WINDOW

    with (
        patch("sys.platform", "win32"),
        patch.dict("sys.modules", {"subprocess": mock_subprocess}),
    ):
        importlib.reload(lc)
        assert lc.CREATE_NO_WINDOW == 0x08000000
        assert lc.CREATE_NEW_CONSOLE == 0x00000010

    # Restore normal state
    importlib.reload(lc)


def test_optional_imports_failure():
    """Test when optional features fail to import."""
    import src.launchers.launcher_constants as lc

    # We mock builtins.__import__ so that it raises ImportError for specific modules
    original_import = __builtins__["__import__"]

    def mock_import(name, *args, **kwargs):
        if name in (
            "src.shared.python.ai.gui",
            "src.shared.python.help_system",
            "src.shared.python.ui",
        ):
            raise ImportError(f"Mocked missing module: {name}")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        importlib.reload(lc)
        assert lc.AI_AVAILABLE is False
        assert lc.HELP_SYSTEM_AVAILABLE is False
        assert lc.UI_COMPONENTS_AVAILABLE is False

    # Reload again to restore true state for other tests just in case
    importlib.reload(lc)


def test_optional_imports_success():
    """Test when optional features succeed to import."""
    from unittest.mock import MagicMock

    import src.launchers.launcher_constants as lc

    with (
        patch.dict(
            "sys.modules",
            {
                "src.shared.python.theme": MagicMock(),
                "src.shared.python.ai.gui": MagicMock(),
                "src.shared.python.help_system": MagicMock(),
                "src.shared.python.ui": MagicMock(),
            },
        ),
        patch("importlib.util.find_spec", return_value=True),
    ):
        importlib.reload(lc)
        assert lc.AI_AVAILABLE is True
        assert lc.HELP_SYSTEM_AVAILABLE is True
        assert lc.UI_COMPONENTS_AVAILABLE is True

    # Reload again to restore
    importlib.reload(lc)
