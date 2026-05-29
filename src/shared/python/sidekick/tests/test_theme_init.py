import importlib
import sys
from unittest.mock import patch


def test_theme_init_imports() -> None:
    # We will reload the module to ensure the initialization runs during test
    import sidekick.theme as module_under_test

    importlib.reload(module_under_test)

    assert module_under_test._THEME_AVAILABLE is True
    assert "BUILTIN_THEMES" in module_under_test.__all__


def test_theme_init_no_theme() -> None:
    import sidekick.theme as module_under_test

    # Simulate an environment where theme import fails
    with patch.dict("sys.modules", {"theme": None}):
        importlib.reload(module_under_test)

    assert module_under_test._THEME_AVAILABLE is False
    assert module_under_test._PYQT6_AVAILABLE is False

    # Reload again and restore
    importlib.reload(module_under_test)
    assert module_under_test._THEME_AVAILABLE is True


def test_theme_init_sys_path() -> None:
    import sidekick.theme as module_under_test

    # Remove from sys.path
    path = str(module_under_test._shared_python_dir)
    if path in sys.path:
        sys.path.remove(path)

    # Reload should add it back
    importlib.reload(module_under_test)
    assert path in sys.path
