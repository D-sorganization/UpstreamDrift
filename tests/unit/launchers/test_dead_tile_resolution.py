"""Regression tests for tiles that could never launch (#8030, #7984).

* ``_SCRIPT_HANDLERS`` hard-coded script paths that do not exist and silently
  overrode the ``path`` declared in ``models.yaml``, so the Pinocchio, OpenSim
  and MyoSuite tiles pointed at missing files.
* ``project_map`` pointed at ``docs/PROJECT_MAP.md`` (the real files live under
  ``docs/architecture/`` and ``docs/governance/``).
* ``movement_optimizer`` declared ``source_root: Movement_Optimizer``, a
  *sibling checkout*, but the policy only ever joined it to the UpstreamDrift
  root.
* ``type: physics_informed`` had no handler at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.launchers.launcher_model_handlers import (
    ModelHandlerRegistry,
    PhysicsInformedHandler,
    ScriptHandler,
    _SCRIPT_HANDLERS,
)
from src.launchers.launcher_model_sources import resolve_model_artifact_path
from src.shared.python.config.model_registry import ModelRegistry
from src.shared.python.config.model_source_providers import ModelSourcePathPolicy

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    return ModelRegistry()


@pytest.fixture(scope="module")
def handlers() -> ModelHandlerRegistry:
    return ModelHandlerRegistry()


class TestScriptHandlerPaths:
    """#8030 — the script handlers must point at files that exist."""

    @pytest.mark.parametrize(
        "handler",
        _SCRIPT_HANDLERS,
        ids=[h.display_name for h in _SCRIPT_HANDLERS],
    )
    def test_fallback_script_path_exists(self, handler: ScriptHandler) -> None:
        """The table's fallback path must name a real file in this repo."""
        candidate = REPO_ROOT / handler._script_path
        assert candidate.is_file(), (
            f"{handler.display_name} fallback script {handler._script_path} "
            "does not exist"
        )

    @pytest.mark.parametrize(
        "model_id", ["pinocchio_golf", "opensim_golf", "myosim_suite"]
    )
    def test_models_yaml_path_wins_and_resolves(
        self, model_id: str, registry: ModelRegistry, handlers: ModelHandlerRegistry
    ) -> None:
        """models.yaml is the source of truth and resolves to a real file."""
        model = registry.get_model(model_id)
        assert model is not None, f"{model_id} missing from the registry"
        handler = handlers.get_handler(model.type)
        assert isinstance(handler, ScriptHandler)

        resolved = handler.resolve_script(model, REPO_ROOT)
        assert resolved == resolve_model_artifact_path(
            model, REPO_ROOT
        ), "ScriptHandler must not override the models.yaml path"
        assert resolved.is_file(), f"{model_id} resolves to a missing file: {resolved}"

    def test_fallback_used_when_model_declares_no_path(self) -> None:
        """A model without a declared path still uses the handler's fallback."""
        handler = ScriptHandler(
            model_types={"demo"},
            script_path="src/launchers/document_proxy.py",
            display_name="Demo",
        )

        class _NoPath:
            id = "demo"
            type = "demo"
            path = None

        assert handler.resolve_script(_NoPath(), REPO_ROOT) == (
            REPO_ROOT / "src/launchers/document_proxy.py"
        )


class TestProjectMapTile:
    """#7984 — the document tile must point at a real Project Map."""

    def test_project_map_resolves(self, registry: ModelRegistry) -> None:
        model = registry.get_model("project_map")
        assert model is not None
        resolved = resolve_model_artifact_path(model, REPO_ROOT)
        assert resolved.is_file(), f"project_map points at missing {resolved}"


class TestSiblingSourceRoot:
    """#7984 — a sibling checkout name must resolve outside the repo."""

    def test_sibling_source_root_prefers_existing_sibling(self, tmp_path) -> None:
        repo = tmp_path / "UpstreamDrift"
        repo.mkdir()
        sibling = tmp_path / "Sibling_Repo"
        (sibling / "src").mkdir(parents=True)

        policy = ModelSourcePathPolicy(repo)
        assert policy.resolve_source_root("Sibling_Repo") == sibling.resolve()

    def test_in_repo_directory_still_wins(self, tmp_path) -> None:
        repo = tmp_path / "UpstreamDrift"
        (repo / "Both").mkdir(parents=True)
        (tmp_path / "Both").mkdir()

        policy = ModelSourcePathPolicy(repo)
        assert policy.resolve_source_root("Both") == (repo / "Both").resolve()

    def test_movement_optimizer_tile_resolves_when_sibling_present(
        self, registry: ModelRegistry
    ) -> None:
        model = registry.get_model("movement_optimizer")
        assert model is not None
        sibling = REPO_ROOT.parent / "Movement_Optimizer"
        if not sibling.is_dir():
            pytest.skip("Movement_Optimizer sibling checkout is not present")
        resolved = resolve_model_artifact_path(model, REPO_ROOT)
        assert resolved == (sibling / "src/movement_optimizer/__main__.py").resolve()


class TestPhysicsInformedTiles:
    """#7984 — physics_informed tiles must have a handler that explains itself."""

    @pytest.mark.parametrize("model_id", ["pinn_pure_rigid", "pinn_hybrid"])
    def test_handler_is_registered(
        self, model_id: str, registry: ModelRegistry, handlers: ModelHandlerRegistry
    ) -> None:
        model = registry.get_model(model_id)
        assert model is not None
        handler = handlers.get_handler(model.type)
        assert isinstance(handler, PhysicsInformedHandler)

    def test_launch_reports_the_real_reason(self, registry: ModelRegistry) -> None:
        model = registry.get_model("pinn_hybrid")
        assert model is not None
        handler = PhysicsInformedHandler()

        message = handler.status_message(model)
        assert "no interactive UI" in message
        assert "physics_informed" in message
        assert handler.launch(model, REPO_ROOT, None) is False
