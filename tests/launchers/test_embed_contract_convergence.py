"""Convergence tests for the single ADR-0013 embedding contract.

Covers issues #8856 and #8857:

* Every bootstrap adapter module path is spelled as a valid dotted
  import path (the old Simscape entry contained ``3D_Golf_Model``,
  which can never be imported).
* Bootstrap failures are loud: a bad module logs a WARNING naming the
  module and the exception, and appears in ``get_bootstrap_failures()``.
* The five physics-engine ``EmbeddableTool`` adapters, the Simscape C3D
  viewer, and the swing->flight pipeline are reachable from bootstrap.
* Handlers resolve registered tools through the registry, and the
  legacy import-and-probe fallback emits a ``DeprecationWarning`` whose
  users are ratcheted down via an explicit allowlist.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from src.launchers import embedded_tool_bootstrap as bootstrap
from src.launchers.launcher_model_handlers import ScriptHandler, SpecialAppHandler
from src.shared.python.launcher_embed import (
    EMBEDDABLE_TOOL_REGISTRY,
    get_embeddable_tool,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]

# Tool ids that were unreachable before #8857/#8856 and must now be
# registered by bootstrap.
PREVIOUSLY_UNREACHABLE_IDS = frozenset(
    {
        "mujoco_unified",
        "drake_golf",
        "pinocchio_golf",
        "opensim_golf",
        "myosim_suite",
        "c3d_viewer",
        "swing_flight_pipeline",
    }
)


@pytest.fixture(autouse=True)
def _reset_bootstrap_state():
    bootstrap.reset_bootstrap_state()
    yield
    bootstrap.reset_bootstrap_state()


def _run_bootstrap() -> list[str]:
    return bootstrap.bootstrap_embeddable_tools()


def test_all_bootstrap_module_paths_are_valid_dotted_imports() -> None:
    """Guard against unimportable entries like ``...3D_Golf_Model...`` (#8856)."""
    for module_path in bootstrap.FALLBACK_ADAPTER_MODULES:
        segments = module_path.split(".")
        assert segments, module_path
        for segment in segments:
            assert segment.isidentifier(), (
                f"Bootstrap adapter module {module_path!r} contains segment "
                f"{segment!r}, which is not a valid Python identifier and "
                "can never be imported. Use an importable shim instead "
                "(see src/launchers/adapters/)."
            )


def test_bad_adapter_module_is_logged_and_recorded(monkeypatch, caplog) -> None:
    """A failing adapter import must warn AND be queryable, not silent (#8856)."""
    bad_module = "definitely_not_a_real_module._embed_adapter"
    monkeypatch.setattr(bootstrap, "FALLBACK_ADAPTER_MODULES", (bad_module,))
    with patch.object(bootstrap, "_iter_entry_point_adapter_modules", return_value=[]):
        with caplog.at_level("WARNING"):
            _run_bootstrap()
    messages = [rec.getMessage() for rec in caplog.records]
    assert any(bad_module in msg for msg in messages), messages
    failures = bootstrap.get_bootstrap_failures()
    assert any(module == bad_module for module, _ in failures), failures
    # The recorded failure carries the exception text.
    assert all(error for _, error in failures)


def test_simscape_shim_is_importable_and_registers_c3d_viewer() -> None:
    """The shim replaces the unimportable 3D_Golf_Model dotted path (#8856)."""
    import src.launchers.adapters.simscape_embed as shim

    assert shim.APPS_PACKAGE_NAME in sys.modules
    tool = get_embeddable_tool("c3d_viewer")
    assert tool is not None
    assert tool.embed_capabilities().supports_embedded


def test_bootstrap_registers_previously_unreachable_adapters() -> None:
    """Engine + Simscape + swing_flight adapters are reachable (#8857)."""
    _run_bootstrap()
    registered = set(EMBEDDABLE_TOOL_REGISTRY)
    missing = PREVIOUSLY_UNREACHABLE_IDS - registered
    assert not missing, (
        f"Bootstrap did not register {sorted(missing)}; "
        f"failures: {bootstrap.get_bootstrap_failures()}"
    )


def test_swing_flight_pipeline_resolves_via_registry_not_fallback() -> None:
    """The handler must consult the registry before the legacy probe (#8857)."""
    _run_bootstrap()
    tool = get_embeddable_tool("swing_flight_pipeline")
    assert tool is not None

    sentinel = object()
    handler = SpecialAppHandler()
    model = SimpleNamespace(
        id="swing_flight_pipeline",
        path="src/tools/swing_flight_pipeline/gui.py",
        embed_adapter=None,
    )
    with (
        patch.object(type(tool), "create_main_widget", return_value=sentinel),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error", DeprecationWarning)
        ui = handler.get_dockable_ui(model, REPO_ROOT)
    assert ui is sentinel


def test_legacy_fallback_emits_deprecation_warning(tmp_path) -> None:
    """Tiles resolved via import-and-probe must warn with the tile id."""
    script = tmp_path / "legacy_tool.py"
    script.write_text("def get_dockable_ui():\n    return object()\n", encoding="utf-8")
    handler = ScriptHandler(
        model_types={"legacy_demo"},
        script_path="legacy_tool.py",
        display_name="Legacy Demo",
    )
    model = SimpleNamespace(id="legacy_demo_tile", path="legacy_tool.py")
    with pytest.warns(DeprecationWarning, match="legacy_demo_tile"):
        ui = handler.get_dockable_ui(model, tmp_path)
    assert ui is not None


# --- Ratchet: legacy-fallback users may only shrink -------------------------
#
# Tiles listed here are known to still rely on the deprecated legacy
# embedding path (an ``embed_adapter`` "mod::func" string in models.yaml
# or a module-level ``get_dockable_ui`` probe) instead of registering an
# ADR-0013 ``EmbeddableTool``. Migrate a tile, then REMOVE it from this
# list. Never add to it.
LEGACY_EMBED_ADAPTER_ALLOWLIST = frozenset(
    {
        "data_explorer",
        "data_processor",
        "pendulum_simulator",
        "video_analyzer",
    }
)


def _models_yaml_tiles() -> list[dict]:
    models_path = REPO_ROOT / "src" / "config" / "models.yaml"
    data = yaml.safe_load(models_path.read_text(encoding="utf-8"))
    return list(data.get("models", []))


def test_legacy_embed_adapter_strings_only_shrink() -> None:
    """Ratchet: tiles declaring legacy ``embed_adapter`` strings (#8857)."""
    tiles_with_legacy_strings = {
        tile["id"] for tile in _models_yaml_tiles() if tile.get("embed_adapter")
    }
    unexpected = tiles_with_legacy_strings - LEGACY_EMBED_ADAPTER_ALLOWLIST
    assert not unexpected, (
        f"New tiles use the deprecated embed_adapter string: {sorted(unexpected)}. "
        "Register an EmbeddableTool adapter (ADR-0013) and add it to "
        "embedded_tool_bootstrap.FALLBACK_ADAPTER_MODULES instead."
    )
    stale = LEGACY_EMBED_ADAPTER_ALLOWLIST - tiles_with_legacy_strings
    assert not stale, (
        f"Tiles migrated off the legacy embed_adapter string: {sorted(stale)}. "
        "Remove them from LEGACY_EMBED_ADAPTER_ALLOWLIST so the ratchet "
        "only goes down."
    )
