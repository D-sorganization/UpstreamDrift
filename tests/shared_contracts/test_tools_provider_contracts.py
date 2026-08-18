from __future__ import annotations

import contextlib
import importlib
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np


def _normalized_path(value: str) -> str:
    return value.replace("\\", "/").lower()


@contextlib.contextmanager
def _fresh_provider_import(name: str) -> Iterator[None]:
    """Import ``name`` through the promoted Tools paths, then restore the cache.

    These contracts must observe where a *fresh* import resolves. By the time a
    test runs, the local copies are already cached: conftest and plugin imports
    execute before this package's ``pytest_configure`` promotes the Tools paths,
    and ``shared.python.import_aliases`` canonicalises every provider spelling
    to whichever copy loaded first. Asserting on the cached module therefore
    tested import history, not resolution order - the suite failed even when
    the vendored tree was correctly provisioned and first on ``sys.path``.

    The cache surgery is scoped to this context manager and fully restored, so
    the surrounding session (which legitimately tests the local copies in the
    ``tests (3.x)`` lanes) keeps its module identities untouched.
    """
    prefixes = [
        name,
        f"shared.python.{name}",
        f"src.shared.python.{name}",
        f"src.{name}",
        "shared",
        "src.shared",
    ]
    if name == "sidekick":
        # The alias finder canonicalises sidekick against the deprecated
        # upstream_drift_tools spellings too; a cached local copy of ANY of
        # them re-binds sidekick to the local tree.
        prefixes += [
            "upstream_drift_tools",
            "shared.python.upstream_drift_tools",
            "src.shared.python.upstream_drift_tools",
        ]
    prefixes = tuple(prefixes)

    def _affected(module_name: str) -> bool:
        return any(
            module_name == prefix or module_name.startswith(prefix + ".")
            for prefix in prefixes
        )

    saved = {key: mod for key, mod in sys.modules.items() if _affected(key)}
    for key in saved:
        del sys.modules[key]
    try:
        yield
    finally:
        for key in [key for key in sys.modules if _affected(key)]:
            del sys.modules[key]
        sys.modules.update(saved)


def _assert_from_tools(path: Path) -> None:
    normalized = _normalized_path(str(path))
    assert any(
        marker in normalized
        for marker in (
            "/_tools_dep/",
            "/vendor/ud-tools/",
            "/repositories/tools/",
            "/tools/",
        )
    ), f"Expected Tools-backed provider path, got: {path}"


def test_signal_toolkit_imports_resolve_from_tools_provider() -> None:
    with _fresh_provider_import("signal_toolkit"):
        module = importlib.import_module("signal_toolkit")
        _assert_from_tools(Path(module.__file__).resolve())

        signal = module.SignalGenerator.sinusoid(
            np.linspace(0.0, 1.0, 16), amplitude=1.0, frequency=2.0
        )
        assert len(signal.values) == 16


def test_humanoid_character_builder_imports_resolve_from_tools_provider() -> None:
    with _fresh_provider_import("humanoid_character_builder"):
        module = importlib.import_module("humanoid_character_builder")
        _assert_from_tools(Path(module.__file__).resolve())

        params = module.BodyParameters(height_m=1.75, mass_kg=72.0)
        assert params.height_m == 1.75
        assert params.mass_kg == 72.0


def test_model_generation_imports_resolve_from_tools_provider() -> None:
    with _fresh_provider_import("model_generation"):
        module = importlib.import_module("model_generation")
        _assert_from_tools(Path(module.__file__).resolve())

        # Execute actual contract behavior instead of just checking callability
        urdf = module.quick_urdf(height_m=1.8, mass_kg=80.0, robot_name="test_robot")
        assert "test_robot" in urdf
        assert "<link" in urdf
        assert module.DEFAULT_HEIGHT_M > 0


def test_sidekick_imports_resolve_from_tools_provider() -> None:
    with _fresh_provider_import("sidekick"):
        module = importlib.import_module("sidekick")
        _assert_from_tools(Path(module.__file__).resolve())

        state_manager = importlib.import_module("sidekick.utils.state_manager")
        _assert_from_tools(Path(state_manager.__file__).resolve())

        # Execute actual contract behavior
        manager = state_manager.StateManager()
        manager.save_state("test_key", {"test_key": "test_value"})
        assert manager.load_state("test_key") == {"test_key": "test_value"}
