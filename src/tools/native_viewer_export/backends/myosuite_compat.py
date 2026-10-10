"""MyoSuite version compatibility for the native export worker.

``myosuite`` 2.x ships ``myosuite.envs.env_base.MujocoEnv`` and
``myosuite.renderer.mj_renderer.MJRenderer``. ``myosuite`` 3.x removed
``env_base`` (environments became Gymnasium task environments) and moved the
renderer to ``myosuite.viz.mj_renderer``. The export only needs the renderer,
which takes a raw ``mujoco`` model and data in both lines, so the worker builds
the model itself and imports ``MJRenderer`` through :func:`import_mj_renderer`.

This module imports only the standard library so the availability probe and
the worker share one import path (issue #11997).
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any

# Newest layout first; the 2.x location is the fallback.
MJ_RENDERER_MODULES: tuple[str, ...] = (
    "myosuite.viz.mj_renderer",
    "myosuite.renderer.mj_renderer",
)

PROBE_CODE = (
    "import mujoco; "
    "from src.tools.native_viewer_export.backends.myosuite_compat "
    "import import_mj_renderer; import_mj_renderer()"
)


def import_mj_renderer() -> Any:
    """Return MyoSuite's ``MJRenderer`` class for the installed major version.

    Postcondition: the returned class accepts ``(mj_model, mj_data)``.

    Raises:
        ImportError: no known renderer module could be imported; the message
            lists every module tried and why it failed.
    """
    failures: list[str] = []
    for name in MJ_RENDERER_MODULES:
        try:
            return import_module(name).MJRenderer
        except (ImportError, AttributeError) as exc:
            failures.append(f"{name}: {exc}")
    raise ImportError("no MyoSuite MJRenderer found (" + "; ".join(failures) + ")")


SCENE_NAME = "myosuite_quad.xml"


def scene_candidates() -> list[Path]:
    """Arena scene locations, newest layout first.

    myosuite 3.x ships its assets in the separate ``myo_sim`` package
    (``myo_sim/models/scene``); 2.x bundles ``myosuite/simhive/myo_sim/scene``.
    """
    paths: list[Path] = []
    spec = find_spec("myo_sim")
    for root in (spec.submodule_search_locations or []) if spec else []:
        paths.append(Path(root) / "models" / "scene" / SCENE_NAME)
    myosuite_spec = find_spec("myosuite")
    if myosuite_spec is not None and myosuite_spec.origin:
        base = Path(myosuite_spec.origin).parent
        paths.append(base / "simhive" / "myo_sim" / "scene" / SCENE_NAME)
    return paths


def find_scene() -> Path:
    """First existing arena scene file.

    Raises:
        FileNotFoundError: no candidate exists; the message lists them all.
    """
    candidates = scene_candidates()
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"MyoSuite arena scene {SCENE_NAME} not found; tried "
        + ", ".join(str(p) for p in candidates)
    )
