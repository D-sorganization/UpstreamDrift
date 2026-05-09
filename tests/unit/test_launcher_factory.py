"""Tests for src.shared.python.launcher_factory (Issues #1949, #1744)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from src.shared.python.launcher_factory import (
    ENGINE_MODULES,
    get_engine_module,
    launch_engine_directly,
)

# ---------------------------------------------------------------------------
# ENGINE_MODULES dict
# ---------------------------------------------------------------------------


class TestEngineModules:
    def test_launcher_factory_is_dict(self) -> None:
        assert isinstance(ENGINE_MODULES, dict)

    def test_launcher_factory_non_empty(self) -> None:
        assert len(ENGINE_MODULES) > 0

    def test_mujoco_key_exists(self) -> None:
        assert "mujoco" in ENGINE_MODULES

    def test_launcher_factory_values_are_strings(self) -> None:
        assert all(isinstance(v, str) for v in ENGINE_MODULES.values())

    def test_module_paths_have_dots(self) -> None:
        # Module paths like "src.engines...." should have dots
        assert all("." in v for v in ENGINE_MODULES.values())


# ---------------------------------------------------------------------------
# get_engine_module
# ---------------------------------------------------------------------------


class TestGetEngineModule:
    def test_known_engine_returns_path(self) -> None:
        result = get_engine_module("mujoco")
        assert result is not None
        assert isinstance(result, str)

    def test_unknown_engine_returns_none(self) -> None:
        result = get_engine_module("nonexistent_engine_xyz")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        result = get_engine_module("")
        assert result is None

    def test_case_sensitive(self) -> None:
        # Keys are lowercase
        assert get_engine_module("MUJOCO") is None

    def test_pinocchio_returns_path(self) -> None:
        result = get_engine_module("pinocchio")
        assert result is not None


# ---------------------------------------------------------------------------
# launch_engine_directly
# ---------------------------------------------------------------------------


class TestLaunchEngineDirectly:
    def test_unknown_engine_calls_exit(self) -> None:
        with pytest.raises(SystemExit):
            launch_engine_directly("unknown_engine_xyz")

    def test_import_error_calls_exit(self) -> None:
        # Patch get_engine_module to return a path, but importlib.import_module fails
        with (
            patch(
                "src.shared.python.launcher_factory.get_engine_module",
                return_value="fake.module.path",
            ),
            patch("importlib.import_module", side_effect=ImportError("not installed")),
            pytest.raises(SystemExit),
        ):
            launch_engine_directly("mujoco")

    def test_module_without_main_calls_exit(self) -> None:
        fake_module = MagicMock(spec=[])  # No 'main' attribute
        with (
            patch(
                "src.shared.python.launcher_factory.get_engine_module",
                return_value="fake.module",
            ),
            patch("importlib.import_module", return_value=fake_module),
            pytest.raises(SystemExit),
        ):
            launch_engine_directly("mujoco")

    def test_module_with_main_calls_main(self) -> None:
        fake_module = MagicMock()
        fake_module.main = MagicMock()
        with (
            patch(
                "src.shared.python.launcher_factory.get_engine_module",
                return_value="fake.module",
            ),
            patch("importlib.import_module", return_value=fake_module),
        ):
            launch_engine_directly("mujoco")
        fake_module.main.assert_called_once()
