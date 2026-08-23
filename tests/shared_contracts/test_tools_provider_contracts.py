from __future__ import annotations

import contextlib
import importlib
import json
import sys
from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest


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


@pytest.mark.integration
def test_rotating_base_provider_retains_complete_qualified_authority() -> None:
    """The pinned Tools provider must retain every scientific boundary."""
    with _fresh_provider_import("swing_sim"):
        module = importlib.import_module("shared.python.swing_sim.rotating_base")
        module_path = Path(module.__file__).resolve()
        _assert_from_tools(module_path)

        assert module.EXPECTED_UPSTREAM_SOURCE_REVISION == (
            "967c40f54cc03f8cae89cde09268d62771d220fe"
        )
        assert module.EXPECTED_STUDY_SHA256 == (
            "e6a55e6cf91e51f21fe3eb8bcb07b990a7798f18abcaf5ca73f5214cb6c5f9ec"
        )
        assert module.EXPECTED_RUN_CATALOG_SHA256 == (
            "66493b833955c6492a00eae4a600df795df60a6f473f9a11c403084b58e51678"
        )
        assert module.MODEL_TIER == ("planar_rotating_base_two_hand_compliant_club")

        study = module.load_embedded_qualified_study().study
        assert study.attempted_case_count == 18
        assert study.valid_case_count == 13
        assert [case.case_index for case in study.cases if not case.valid] == [
            6,
            7,
            8,
            15,
            16,
        ]
        assert study.human_coaching_supported is False

        catalog_path = (
            module_path.parent / "resources" / "rotating_base_registered_runs_v1.json"
        )
        catalog_text = catalog_path.read_text(encoding="utf-8").rstrip("\n")
        assert sha256(catalog_text.encode("utf-8")).hexdigest() == (
            module.EXPECTED_RUN_CATALOG_SHA256
        )
        catalog = json.loads(catalog_text)
        assert catalog["attempted_run_count"] == 18
        assert catalog["source_revision"] == module.EXPECTED_UPSTREAM_SOURCE_REVISION
        assert catalog["study_sha256"] == module.EXPECTED_STUDY_SHA256
        assert [run["request"]["case_index"] for run in catalog["runs"]] == list(
            range(18)
        )
        assert [
            run["case"]["case_index"]
            for run in catalog["runs"]
            if not run["case"]["valid"]
        ] == [6, 7, 8, 15, 16]
        assert all(
            run["boundaries"]
            == {
                "coaching_recommendation": "unsupported",
                "coordinate_semantics": "nonanatomical_model_coordinate",
                "human_validation": "unavailable",
            }
            for run in catalog["runs"]
        )
