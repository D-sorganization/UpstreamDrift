"""Tests for src.launchers.embedded_tool_bootstrap.

The bootstrap module imports a hard-coded list of adapter modules at
runtime to trigger their self-registration with the embeddable tool
registry. These tests verify:

* Idempotency: second call returns cached results without re-importing.
* Path injection: vendor/ud-tools/src is added to ``sys.path``.
* Graceful degradation: failing modules log a warning but do not abort.
* State reset for tests.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from src.launchers import embedded_tool_bootstrap as bootstrap


@pytest.fixture(autouse=True)
def _reset_bootstrap_state():
    """Reset module-level state before and after each test."""
    bootstrap.reset_bootstrap_state()
    yield
    bootstrap.reset_bootstrap_state()


def test_reset_clears_state() -> None:
    bootstrap._registered_tools = ["foo", "bar"]
    bootstrap._bootstrap_complete = True
    bootstrap.reset_bootstrap_state()
    assert bootstrap._registered_tools == []
    assert bootstrap._bootstrap_complete is False


def test_get_bootstrapped_tools_returns_copy() -> None:
    bootstrap._registered_tools = ["a", "b"]
    result = bootstrap.get_bootstrapped_tools()
    assert result == ["a", "b"]
    # Mutating the returned list should not change internal state
    result.append("c")
    assert bootstrap._registered_tools == ["a", "b"]


import builtins as _builtins  # noqa: E402

_ADAPTER_TOKEN = "_embed_adapter"
_REAL_IMPORT = _builtins.__import__


class _FakeEntryPoint:
    def __init__(self, value: str) -> None:
        self.value = value


def _make_filtered_import(decide):
    """Build an __import__ replacement that delegates to the real one
    for everything except adapter module names, which go through *decide*.
    """

    def _filtered(name, globals=None, locals=None, fromlist=(), level=0):
        if _ADAPTER_TOKEN in name or "pose_studio.gui" in name:
            return decide(name)
        return _REAL_IMPORT(name, globals, locals, fromlist, level)

    return _filtered


def test_bootstrap_handles_import_errors() -> None:
    """When all adapter modules fail to import, return an empty list."""

    def _fail(name):
        raise ImportError(f"forced for {name}")

    with patch.object(_builtins, "__import__", _make_filtered_import(_fail)):
        result = bootstrap.bootstrap_embeddable_tools()
    assert result == []
    assert bootstrap._bootstrap_complete is True


def test_bootstrap_handles_generic_errors() -> None:
    """Non-ImportError exceptions during import are also swallowed."""

    def _boom(name):
        raise RuntimeError(f"forced runtime for {name}")

    with patch.object(_builtins, "__import__", _make_filtered_import(_boom)):
        result = bootstrap.bootstrap_embeddable_tools()
    assert result == []
    assert bootstrap._bootstrap_complete is True


def test_bootstrap_is_idempotent() -> None:
    """Calling bootstrap twice should not re-run adapter imports."""
    calls = {"n": 0}

    def _counting(name):
        calls["n"] += 1
        raise ImportError("forced")

    with patch.object(_builtins, "__import__", _make_filtered_import(_counting)):
        first = bootstrap.bootstrap_embeddable_tools()
        first_count = calls["n"]
        second = bootstrap.bootstrap_embeddable_tools()
    assert second == first
    assert calls["n"] == first_count


def test_bootstrap_adds_vendor_path_to_sys_path() -> None:
    """sys.path should gain the vendor/ud-tools/src directory."""
    from pathlib import Path

    vendor_src = str(
        Path(bootstrap.__file__).resolve().parent.parent.parent
        / "vendor"
        / "ud-tools"
        / "src"
    )
    original_path = list(sys.path)
    try:
        sys.path[:] = [p for p in sys.path if p != vendor_src]

        def _noop(name):
            raise ImportError("noop")

        with (
            patch.object(Path, "is_dir", return_value=False),
            patch.object(_builtins, "__import__", _make_filtered_import(_noop)),
        ):
            bootstrap.bootstrap_embeddable_tools()
        assert vendor_src in sys.path
    finally:
        sys.path[:] = original_path


def test_bootstrap_records_successfully_imported_tools() -> None:
    """Tool ids actually registered during an adapter import are recorded.

    Recording diffs the registry around each import, so the recorded ids
    are the ones the adapter registered — not ids guessed from the
    module name.
    """
    from src.shared.python.launcher_embed import EMBEDDABLE_TOOL_REGISTRY
    from src.shared.python.launcher_embed.contract import EmbedCapabilities

    class _FakeTool:
        tool_id = "model_explorer"

        def embed_capabilities(self) -> EmbedCapabilities:
            return EmbedCapabilities(supports_embedded=True)

        def create_main_widget(self, parent: object) -> object:
            return object()

        def cleanup(self) -> None:
            pass

        def is_dirty(self) -> bool:
            return False

    def _selective(name):
        if name == "src.tools.model_explorer._embed_adapter":
            # Simulate the adapter's import-time self-registration.
            EMBEDDABLE_TOOL_REGISTRY.setdefault("model_explorer", _FakeTool())
            return object()
        raise ImportError(f"forced for {name}")

    try:
        with patch.object(_builtins, "__import__", _make_filtered_import(_selective)):
            result = bootstrap.bootstrap_embeddable_tools()
        assert "model_explorer" in result
    finally:
        EMBEDDABLE_TOOL_REGISTRY.pop("model_explorer", None)


def test_bootstrap_imports_entry_point_adapters(monkeypatch) -> None:
    """Entry-point adapters are imported before the hard-coded fallback."""
    imported: list[str] = []

    def _entry_points(*, group: str):
        assert group == bootstrap.EMBEDDABLE_TOOL_ENTRY_POINT_GROUP
        return [_FakeEntryPoint("external_package._embed_adapter")]

    def _selective(name):
        imported.append(name)
        raise ImportError(f"forced for {name}")

    monkeypatch.setattr(bootstrap.importlib.metadata, "entry_points", _entry_points)
    monkeypatch.setattr(bootstrap, "_warn_on_manifest_gaps", lambda: None)
    with patch.object(_builtins, "__import__", _make_filtered_import(_selective)):
        result = bootstrap.bootstrap_embeddable_tools()

    assert result == []
    assert imported[0] == "external_package._embed_adapter"


def test_bootstrap_de_dupes_entry_points_and_fallback(monkeypatch) -> None:
    """Adapters named by metadata and fallback are imported only once."""
    fallback_module = "src.tools.model_explorer._embed_adapter"
    imported: list[str] = []

    def _entry_points(*, group: str):
        assert group == bootstrap.EMBEDDABLE_TOOL_ENTRY_POINT_GROUP
        return [
            _FakeEntryPoint(fallback_module),
            _FakeEntryPoint("external_package._embed_adapter"),
            _FakeEntryPoint("external_package._embed_adapter"),
        ]

    def _selective(name):
        imported.append(name)
        raise ImportError(f"forced for {name}")

    monkeypatch.setattr(bootstrap.importlib.metadata, "entry_points", _entry_points)
    monkeypatch.setattr(bootstrap, "_warn_on_manifest_gaps", lambda: None)
    with patch.object(_builtins, "__import__", _make_filtered_import(_selective)):
        bootstrap.bootstrap_embeddable_tools()

    assert imported.count(fallback_module) == 1
    assert imported.count("external_package._embed_adapter") == 1


def test_entry_point_bootstrap_records_registry_diff(monkeypatch) -> None:
    """Entry-point adapters use the same registry-diff recording as fallback."""
    from src.shared.python.launcher_embed import EMBEDDABLE_TOOL_REGISTRY
    from src.shared.python.launcher_embed.contract import EmbedCapabilities

    class _FakeTool:
        tool_id = "external_tool"

        def embed_capabilities(self) -> EmbedCapabilities:
            return EmbedCapabilities(supports_embedded=True)

        def create_main_widget(self, parent: object) -> object:
            return object()

        def cleanup(self) -> None:
            pass

        def is_dirty(self) -> bool:
            return False

    def _entry_points(*, group: str):
        assert group == bootstrap.EMBEDDABLE_TOOL_ENTRY_POINT_GROUP
        return [_FakeEntryPoint("external_package._embed_adapter")]

    def _selective(name):
        if name == "external_package._embed_adapter":
            EMBEDDABLE_TOOL_REGISTRY.setdefault("external_tool", _FakeTool())
            return object()
        raise ImportError(f"forced for {name}")

    monkeypatch.setattr(bootstrap.importlib.metadata, "entry_points", _entry_points)
    monkeypatch.setattr(bootstrap, "_warn_on_manifest_gaps", lambda: None)
    try:
        with patch.object(_builtins, "__import__", _make_filtered_import(_selective)):
            result = bootstrap.bootstrap_embeddable_tools()
        assert result == ["external_tool"]
    finally:
        EMBEDDABLE_TOOL_REGISTRY.pop("external_tool", None)


def test_bootstrap_warns_on_manifest_gaps_after_entry_point_discovery(
    monkeypatch,
) -> None:
    """Manifest coverage warnings still run after discovery-based bootstrap."""
    warnings = {"count": 0}

    def _entry_points(*, group: str):
        assert group == bootstrap.EMBEDDABLE_TOOL_ENTRY_POINT_GROUP
        return [_FakeEntryPoint("external_package._embed_adapter")]

    def _selective(name):
        raise ImportError(f"forced for {name}")

    def _warn() -> None:
        warnings["count"] += 1

    monkeypatch.setattr(bootstrap.importlib.metadata, "entry_points", _entry_points)
    monkeypatch.setattr(bootstrap, "_warn_on_manifest_gaps", _warn)
    with patch.object(_builtins, "__import__", _make_filtered_import(_selective)):
        bootstrap.bootstrap_embeddable_tools()

    assert warnings["count"] == 1


# Tools whose adapters live in this repo (not the vendored Tools submodule) and
# have no heavy/optional runtime dependencies, so they must register on a bare
# bootstrap run. Regression guard for #6560 (training_controller import bug) and
# #6561 (six adapters that existed but were never added to adapter_modules).
_FIRST_PARTY_TOOL_IDS = {
    "ball_flight_gui",
    "bunker_shot_gui",
    "canonical_core_comparison",
    "canonical_core_estimation",
    "model_explorer",
    "putting_green_gui",
    "golf_environment",
    "terrain_engine",
    "golf_simulation_suite",
    "training_controller",
    "video_analyzer",
}


def test_first_party_tools_are_listed_in_adapter_modules() -> None:
    """Static guard: every first-party tool has an adapter_modules entry."""
    for tool_id in _FIRST_PARTY_TOOL_IDS:
        if tool_id.startswith("canonical_core_"):
            expected = "src.tools.canonical_core._embed_adapter"
        else:
            expected = f"src.tools.{tool_id}."
        assert any(
            expected in module_path
            for module_path in bootstrap.FALLBACK_ADAPTER_MODULES
        ), f"{tool_id} missing from adapter_modules"


def test_bootstrap_registers_all_first_party_tools(monkeypatch) -> None:
    """Functional guard: a real bootstrap run registers every first-party tool.

    Runs offscreen so PyQt6 widgets can be constructed headlessly.
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    registered = set(bootstrap.bootstrap_embeddable_tools())
    missing = _FIRST_PARTY_TOOL_IDS - registered
    assert not missing, f"first-party tools failed to bootstrap: {sorted(missing)}"
