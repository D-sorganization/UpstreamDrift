"""Launch-mode functional QA gate (issue #8966, EPIC #8965 WS1).

Matrix of functional readiness checks for the four launch modes routed
by ``launch_upstream_drift.py``:

* web (default): FastAPI app constructed within budget, health endpoint
  responds, launcher manifest serves a non-empty tile set.
* ``--classic``: PyQt6 launcher main window constructed offscreen with a
  real model registry; tile grid populated; no unhandled exception.
* ``--api-only``: same app object serves the interactive API docs.
* ``--engine <id>``: for every engine declared in the manifest/registry,
  the direct-launch module resolves (skip-with-reason when the runtime
  is not installed — never fake success). MuJoCo (installed in CI) gets
  a real construct-without-exec window test.
* Registry parity and ready-tile target resolution are real regression
  gates for #8853 / #8854, asserting against the settled contracts
  (``models.yaml`` ``web_catalog`` entries, shared ``resolve_tile_target``).

Run with: ``pytest tests/launch_modes -m launch_qa``.
See docs/testing/launch_qa.md.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.launch_qa, pytest.mark.integration]

from tests.launch_modes.conftest import WEB_APP_CONSTRUCTION_BUDGET_S  # noqa: E402

#: Runtime import that must succeed for each directly-launchable engine.
#: ``None`` means the engine has no optional heavy runtime (pure PyQt6).
_ENGINE_RUNTIMES: dict[str, str | None] = {
    "mujoco": "mujoco",
    "drake": "pydrake",
    "pinocchio": "pinocchio",
    "opensim": "opensim",
    "myosuite": "myosuite",
    "myosim": "myosuite",
    "pendulum": None,
}

#: Engines routed to the web UI instead of a direct GUI launch.
_WEB_ONLY_ENGINES = {"matlab_2d", "matlab_3d"}

#: Known-broken direct-launch paths -> tracking issue. When the issue is
#: fixed the imperative xfail below is no longer reached and the test
#: flips to pass (equivalent to strict=False).
_KNOWN_BROKEN_DIRECT_LAUNCH: dict[str, str] = {
    "mujoco": "#8967",
    "pendulum": "#8967",
}


def _declared_engine_ids() -> list[str]:
    """Engine ids the launch surface declares (CLI choices + manifest).

    Mirrors the ``--engine`` choices built by
    ``launch_upstream_drift.parse_arguments`` and augments them with any
    ``engine_type`` declared by manifest tiles.
    """
    ids: set[str] = set()
    try:
        from src.shared.python.engine_core.engine_manager import EngineType

        ids.update(e.value for e in EngineType)
    except ImportError:
        ids.update(_ENGINE_RUNTIMES)
    try:
        from src.config.launcher_manifest_loader import LauncherManifest

        ids.update(
            t.engine_type for t in LauncherManifest.load().tiles if t.engine_type
        )
    except (
        Exception  # noqa: BLE001
    ):  # pragma: no cover - manifest failures surface elsewhere
        pass
    return sorted(ids)


def _runtime_available(module_name: str) -> bool:
    """Check importability of an engine runtime without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


# ── 1. Web mode ──────────────────────────────────────────────────────


@pytest.mark.timeout(90)
def test_web_app_constructs_within_budget(
    local_app_bundle: tuple[Any, float],
) -> None:
    """Precondition: the web app must build within the cold-start budget."""
    _, elapsed = local_app_bundle
    assert elapsed <= WEB_APP_CONSTRUCTION_BUDGET_S, (
        f"create_local_app() took {elapsed:.1f}s "
        f"(budget {WEB_APP_CONSTRUCTION_BUDGET_S:.0f}s; see #8934/#8938)"
    )


def test_web_readiness_endpoint(web_client: Any) -> None:
    """Web mode readiness: the health endpoint answers 200 with a status."""
    response = web_client.get("/api/health")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, dict) and payload, "health payload must be non-empty"


def test_web_manifest_serves_nonempty_tile_set(web_client: Any) -> None:
    """The launcher manifest endpoint must return a non-empty tile set."""
    response = web_client.get("/api/launcher/manifest")
    assert response.status_code == 200, response.text
    manifest = response.json()
    tiles = manifest.get("tiles", [])
    assert isinstance(tiles, list) and len(tiles) > 0, (
        "manifest endpoint returned an empty tile set"
    )
    tile_ids = [t.get("id") for t in tiles]
    assert all(tile_ids), "every manifest tile must carry an id"
    assert len(set(tile_ids)) == len(tile_ids), "manifest tile ids must be unique"


# ── 2. Classic (PyQt6) mode ──────────────────────────────────────────


@pytest.mark.ui
@pytest.mark.timeout(120)
def test_classic_launcher_populates_tile_grid(qapp: Any) -> None:
    """--classic reaches ready: window built offscreen, tile grid populated.

    Uses a real ``ModelRegistry`` (the same source the classic launcher
    loads) via pre-built ``StartupResults`` so the tile grid population
    is genuine. Docker probing and background timers are stubbed through
    their existing seams (``DockerCheckThread`` patch + pre-supplied
    ``docker_available``) because CI runners have no Docker daemon; the
    engine manager is a stub because engine probing is covered by the
    engine-mode tests below.
    """
    from src.launchers.launcher_constants import (
        REPOS_ROOT,
        _lazy_load_model_registry,
    )
    from src.launchers.startup import StartupResults
    from src.launchers.upstream_drift_launcher import UpstreamDriftLauncher

    registry_cls = _lazy_load_model_registry()
    registry = registry_cls(REPOS_ROOT / "src/config/models.yaml")

    results = StartupResults()
    results.registry = registry
    results.engine_manager = MagicMock(name="engine_manager_stub")
    results.docker_available = False
    results.startup_time_ms = 1

    from src.launchers.launcher_sidekick_sidebar import SidekickSidebarManager

    with (
        patch("src.launchers.upstream_drift_launcher.DockerCheckThread"),
        # Existing seam: the Sidekick sidebar validates the vendored Tools
        # runtime, which is orthogonal to tile-grid readiness and not
        # guaranteed complete on every checkout (see the overlay's
        # IncompleteParentSidekickRuntimeError).
        patch.object(
            SidekickSidebarManager, "_install_sidekick_import_paths", lambda self: None
        ),
    ):
        try:
            launcher = UpstreamDriftLauncher(startup_results=results)
        except ImportError as exc:
            if "src.shared.python.theme" in str(exc) or "shared.python.theme" in str(
                exc
            ):
                # init_ui imports get_current_colors/DARK_THEME, neither of
                # which the theme package exports.
                pytest.xfail(f"#8972: classic launcher init_ui crashes ({exc})")
            raise
    try:
        assert launcher.registry is registry
        assert len(launcher.available_models) > 0, (
            "classic launcher built no available models from the registry"
        )
        assert len(launcher.model_cards) > 0, (
            "classic launcher tile grid is empty after construction"
        )
    finally:
        launcher.close()
        launcher.deleteLater()


# ── 3. API-only mode ─────────────────────────────────────────────────


def test_api_only_serves_docs(web_client: Any) -> None:
    """--api-only serves the same app object; its Swagger docs must load."""
    response = web_client.get("/api/docs")
    assert response.status_code == 200, response.text
    assert "swagger" in response.text.lower()


# ── 4. Engine mode ───────────────────────────────────────────────────


@pytest.mark.parametrize("engine_id", _declared_engine_ids())
def test_engine_direct_launch_path_resolves(engine_id: str) -> None:
    """--engine <id> resolves the module ``launch_engine_directly`` runs.

    Imports the exact module ``launch_engine_directly`` would import and
    asserts it exposes ``main``. Engines without a direct-launch module
    route to the web UI (covered by the web tests) and engines whose
    runtime is not installed skip with a reason — never fake success.
    """
    from src.shared.python.launcher_factory import ENGINE_MODULES

    if engine_id in _WEB_ONLY_ENGINES:
        pytest.skip(f"engine '{engine_id}' is web-only; routed to web UI")
    module_path = ENGINE_MODULES.get(engine_id)
    if module_path is None:
        pytest.skip(f"engine '{engine_id}' has no direct-launch module")
    runtime = _ENGINE_RUNTIMES.get(engine_id)
    if runtime is not None and not _runtime_available(runtime):
        pytest.skip(f"engine not installed: {engine_id}")

    known_broken = _KNOWN_BROKEN_DIRECT_LAUNCH.get(engine_id)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        if known_broken:
            pytest.xfail(f"{known_broken}: direct-launch module missing ({exc})")
        raise
    if not callable(getattr(module, "main", None)):
        if known_broken:
            pytest.xfail(f"{known_broken}: {module_path} lacks a callable main()")
        pytest.fail(
            f"{module_path} lacks a callable main(); --engine {engine_id} would exit(1)"
        )


@pytest.mark.ui
@pytest.mark.requires_mujoco
@pytest.mark.timeout(120)
def test_engine_mujoco_constructs_headless(qapp: Any) -> None:
    """--engine mujoco reaches its constructed state without exec().

    Drives the same code path ``route_launch`` uses (the module named in
    ``ENGINE_MODULES['mujoco']``) but constructs the launcher window
    directly instead of entering the Qt event loop.
    """
    if not _runtime_available("mujoco"):
        pytest.skip("engine not installed: mujoco")

    from src.shared.python.launcher_factory import ENGINE_MODULES

    module = importlib.import_module(ENGINE_MODULES["mujoco"])
    try:
        window = module.HumanoidLauncher()
    except TypeError as exc:
        # ConfigurationManager.load() returns a SimulationConfig dataclass
        # but __init__ probes it like a dict ("engine_root" not in config).
        pytest.xfail(f"#8967: HumanoidLauncher.__init__ crashes ({exc})")
    try:
        assert window is not None
        assert window.centralWidget() is not None, (
            "MuJoCo launcher constructed without a central widget"
        )
    finally:
        window.close()
        window.deleteLater()


# ── 5/6. Cross-mode contracts (regression gates) ─────────────────────


def test_web_and_pyqt_tile_registries_agree(web_client: Any) -> None:
    """Regression gate for #8853: both launch surfaces serve one tile set.

    The settled contract (tests/config/test_tile_registry.py, #9412):
    every visible desktop tile reaches the web surface, and the only web
    extras are the justified ``web_catalog`` entries of models.yaml (#9412). Hidden desktop
    alias tiles are deliberately absent from the web manifest (#8863).
    """
    from src.config.launcher_manifest_loader import LauncherManifest
    from src.launchers.launcher_constants import (
        REPOS_ROOT,
        _lazy_load_model_registry,
    )
    from src.config.tile_registry import load_tile_registry

    response = web_client.get("/api/launcher/manifest")
    assert response.status_code == 200
    web_ids = {t["id"] for t in response.json().get("tiles", [])}

    registry_cls = _lazy_load_model_registry()
    registry = registry_cls(REPOS_ROOT / "src/config/models.yaml")
    pyqt_ids = {model.id for model in registry.get_all_models()}

    hidden_ids = {tile.id for tile in LauncherManifest.load().tiles if tile.hidden}
    structurally_excluded = pyqt_ids - web_ids - hidden_ids
    assert not structurally_excluded, (
        f"desktop tiles structurally excluded from the web surface: "
        f"{sorted(structurally_excluded)}"
    )
    web_only = {t.id for t in load_tile_registry().web_catalog_tiles()}
    undeclared_extras = web_ids - pyqt_ids - web_only
    assert not undeclared_extras, (
        f"web-only tiles missing a models.yaml web_catalog entry: "
        f"{sorted(undeclared_extras)}"
    )


def test_ready_tiles_resolve_launch_targets() -> None:
    """Postcondition (#8966 DbC): every ready tile resolves a real target.

    "Ready" implies launchable — resolution goes through the shared
    ``resolve_tile_target`` policy (#8854), so virtual targets, provider
    gitlinks, and sibling checkouts are judged exactly as the launch
    handlers and the desktop status chip judge them. External targets
    absent on this machine are an environment gap, not a registry bug.
    """
    from src.config.launcher_manifest_loader import LauncherManifest
    from src.launchers.launcher_constants import REPOS_ROOT
    from src.shared.python.config.tile_target_resolution import (
        EXTERNAL_KINDS,
        resolve_tile_target,
    )

    manifest = LauncherManifest.load()
    ready_statuses = {"ready", "gui_ready", "engine_ready"}
    broken: list[str] = []
    for tile in manifest.tiles:
        if tile.status not in ready_statuses or not tile.path:
            continue
        if tile.path.startswith(("http://", "https://")):
            continue
        resolution = resolve_tile_target(tile, REPOS_ROOT)
        if resolution.resolvable or resolution.kind in EXTERNAL_KINDS:
            continue
        broken.append(f"{tile.id} -> {tile.path} ({resolution.reason})")
    assert not broken, f"ready tiles with unresolvable targets: {broken}"
