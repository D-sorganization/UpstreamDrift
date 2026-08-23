"""Regression tests for the launch-mode breaks surfaced by the QA gate.

* #8972: ``--classic`` crashed because the theme package exported neither
  ``get_current_colors`` nor ``DARK_THEME`` while the launcher imported both.
* #8967: ``--engine mujoco`` failed (no module-level ``main()``; dataclass
  probed like a dict) and ``--engine pendulum`` pointed at a missing module.

Part of EPIC #8965 (WS1).
"""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

# ── #8972: theme package exports ─────────────────────────────────────

#: Keys the launcher call sites consume from ``get_current_colors()``
#: (src/launchers/launcher_ui_setup.py splitter styling and
#: src/launchers/custom_title_bar.py fallbacks).
_LAUNCHER_CONSUMED_KEYS = ("border", "accent")


def test_theme_package_exports_get_current_colors() -> None:
    """The UD-owned palette module exports a callable ``get_current_colors``."""
    from src.shared.python.theme.palette import get_current_colors

    assert callable(get_current_colors)


def test_get_current_colors_returns_complete_mapping() -> None:
    """Postcondition: mapping covers every canonical theme color key."""
    from src.shared.python.theme import THEME_COLOR_KEYS
    from src.shared.python.theme.palette import get_current_colors

    colors = get_current_colors()
    assert isinstance(colors, dict)
    missing = [k for k in THEME_COLOR_KEYS if k not in colors]
    assert not missing, f"get_current_colors() missing keys: {missing}"
    for key in _LAUNCHER_CONSUMED_KEYS:
        value = colors[key]
        assert isinstance(value, str) and value.startswith("#"), (
            f"{key!r} must be a hex color, got {value!r}"
        )


def test_dark_theme_fallback_constant_exported() -> None:
    """``DARK_THEME`` is exported and derives from the built-in Dark palette."""
    from src.shared.python.theme import BUILTIN_THEMES
    from src.shared.python.theme.palette import DARK_THEME

    assert dict(DARK_THEME) == dict(BUILTIN_THEMES["Dark"])


def test_palette_resolves_semantic_attribute_aliases() -> None:
    """Sidebar-consumed semantic names resolve via attribute access."""
    from src.shared.python.theme.palette import get_current_colors

    colors = get_current_colors()
    # Names consumed attribute-style by _launcher_navigation_ui.py.
    for name in (
        "bg",
        "bg_elevated",
        "bg_highlight",
        "border_default",
        "primary",
        "text_primary",
        "text_secondary",
    ):
        value = getattr(colors, name)
        assert isinstance(value, str) and value.startswith("#"), (name, value)
    with pytest.raises(AttributeError):
        _ = colors.not_a_color


def test_startup_get_theme_colors_returns_mapping() -> None:
    """``startup._get_theme_colors`` no longer raises ImportError (#8972)."""
    from src.launchers.startup import _get_theme_colors

    colors = _get_theme_colors()
    for key in _LAUNCHER_CONSUMED_KEYS:
        assert key in colors


@pytest.mark.ui
@pytest.mark.timeout(120)
def test_classic_launcher_constructs_offscreen(qapp: Any) -> None:
    """--classic constructs its main window offscreen (same path as QA gate)."""
    from src.launchers.launcher_constants import (
        REPOS_ROOT,
        _lazy_load_model_registry,
    )
    from src.launchers.launcher_sidekick_sidebar import SidekickSidebarManager
    from src.launchers.startup import StartupResults
    from src.launchers.upstream_drift_launcher import UpstreamDriftLauncher

    registry_cls = _lazy_load_model_registry()
    registry = registry_cls(REPOS_ROOT / "src/config/models.yaml")

    results = StartupResults()
    results.registry = registry
    results.engine_manager = MagicMock(name="engine_manager_stub")
    results.docker_available = False
    results.startup_time_ms = 1

    with (
        patch("src.launchers.upstream_drift_launcher.DockerCheckThread"),
        patch.object(
            SidekickSidebarManager,
            "_install_sidekick_import_paths",
            lambda self: None,
        ),
    ):
        launcher = UpstreamDriftLauncher(startup_results=results)
    try:
        assert launcher.registry is registry
    finally:
        launcher.close()
        launcher.deleteLater()


# ── #8967: --engine direct launch ────────────────────────────────────


def _runtime_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def test_simulation_config_declares_root_fields() -> None:
    """SimulationConfig declares engine_root/model_root (regression #7941)."""
    from src.shared.python.config.configuration_manager import SimulationConfig

    config = SimulationConfig()
    assert hasattr(config, "engine_root")
    assert hasattr(config, "model_root")


def test_mujoco_launcher_module_exposes_main() -> None:
    """--engine mujoco: the routed module must expose a callable main()."""
    if not _runtime_available("mujoco"):
        pytest.skip("engine not installed: mujoco")
    from src.shared.python.launcher_factory import ENGINE_MODULES

    module = importlib.import_module(ENGINE_MODULES["mujoco"])
    assert callable(getattr(module, "main", None))


@pytest.mark.ui
@pytest.mark.timeout(120)
def test_mujoco_launcher_constructs_offscreen(qapp: Any) -> None:
    """--engine mujoco: HumanoidLauncher constructs without dict-probing crash."""
    if not _runtime_available("mujoco"):
        pytest.skip("engine not installed: mujoco")
    from src.shared.python.launcher_factory import ENGINE_MODULES

    module = importlib.import_module(ENGINE_MODULES["mujoco"])
    window = module.HumanoidLauncher()
    try:
        assert window.config.engine_root, "engine_root default must be populated"
        assert window.config.model_root, "model_root default must be populated"
        assert window.centralWidget() is not None
    finally:
        window.close()
        window.deleteLater()


def test_pendulum_launcher_module_exposes_main() -> None:
    """--engine pendulum: the routed module must import and expose main()."""
    from src.shared.python.launcher_factory import ENGINE_MODULES

    module = importlib.import_module(ENGINE_MODULES["pendulum"])
    assert callable(getattr(module, "main", None))
