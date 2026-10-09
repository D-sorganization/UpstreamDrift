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
