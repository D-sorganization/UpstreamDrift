"""``bunkershot3d`` stays import-safe on GL-less machines (issue #9145).

``backends`` imports mujoco, and mujoco touches a PyOpenGL binding at
*import time*. Before this fix, ``bunkershot3d/__init__.py`` eagerly
re-exported ``backends`` -- both via the package's own ``from . import
(...)`` block and via ``from .backends import ChronoDriver, LiggghtsDriver,
MPMDriver`` -- so *any* import of any name from ``bunkershot3d``, including a
pure-dataclass leaf like ``EnvelopeStatus``, dragged in mujoco and OpenGL.

On a machine with no GPU/display the OpenGL binding resolves to ``None`` and
mujoco's import fails with ``AttributeError: 'NoneType' object has no
attribute 'glGetError'`` -- a crash on import of a headless, numeric module.
That is what broke ``unit-test-gate`` on PR #9138: ``render3d_vtk.py``, a
deliberately headless renderer, did
``from bunkershot3d.solvers import EnvelopeStatus`` and pulled in mujoco.

This mirrors the check added for ``render3d_vtk`` in PR #9138
(``tests/unit/tools/bunker_shot_gui/test_render3d_vtk_degradation.py``), but
at the package boundary where the leak actually originates: importing
``bunkershot3d`` or any of its leaves must never import mujoco, OpenGL, vtk,
or pyvista as a side effect.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

#: Heavyweight, graphics-adjacent modules that a headless numeric import
#: must never pull in as a side effect.
_FORBIDDEN_MODULES = ("mujoco", "OpenGL", "vtk", "pyvista")

#: Repo root -- this file lives at ``tests/bunkershot3d/``.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _sys_modules_after(statement: str) -> set[str]:
    """Run ``statement`` in a fresh subprocess and report loaded modules.

    A subprocess (rather than importing in-process and inspecting
    ``sys.modules`` directly) is required: this test suite may run after
    other tests have already imported ``bunkershot3d`` -- or mujoco, or
    pyvista -- for unrelated reasons, which would make an in-process check
    meaningless. A clean interpreter is the only way to observe the true
    import graph of ``statement`` alone.

    ``bunkershot3d`` is a ``src``-layout package resolved for the normal
    test run via pytest's ``pythonpath`` ini option, which does not apply
    to a subprocess -- so ``src`` is put on ``PYTHONPATH`` explicitly here.
    """
    code = f"{statement}\nimport sys\nprint(','.join(sorted(sys.modules)))\n"
    env = dict(os.environ)
    src_dir = str(_REPO_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing}" if existing else src_dir
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
        cwd=str(_REPO_ROOT),
        env=env,
    )
    return set(result.stdout.strip().split(","))


class TestImportingThePackageStaysHeadless:
    def test_import_bunkershot3d_does_not_load_forbidden_modules(self) -> None:
        loaded = _sys_modules_after("import bunkershot3d")
        leaked = loaded & set(_FORBIDDEN_MODULES)
        assert not leaked, (
            f"`import bunkershot3d` pulled in {sorted(leaked)}; the "
            "backends subpackage (mujoco/OpenGL) must be lazy -- see #9145"
        )

    def test_leaf_dataclass_import_does_not_load_forbidden_modules(self) -> None:
        """The exact regression from #9138: a pure-dataclass leaf import."""
        loaded = _sys_modules_after("from bunkershot3d.solvers import EnvelopeStatus")
        leaked = loaded & set(_FORBIDDEN_MODULES)
        assert not leaked, (
            f"`from bunkershot3d.solvers import EnvelopeStatus` pulled in "
            f"{sorted(leaked)} -- see #9145"
        )


class TestBackendsStillResolveLazilyOnFirstUse:
    """The public API must stay identical -- only the import-time cost moves."""

    def test_driver_classes_resolve_from_the_top_level_package(self) -> None:
        import bunkershot3d
        from bunkershot3d.backends import ChronoDriver, LiggghtsDriver, MPMDriver

        assert bunkershot3d.ChronoDriver is ChronoDriver
        assert bunkershot3d.LiggghtsDriver is LiggghtsDriver
        assert bunkershot3d.MPMDriver is MPMDriver

    def test_backends_submodule_resolves_from_the_top_level_package(self) -> None:
        import bunkershot3d
        from bunkershot3d import backends

        assert bunkershot3d.backends is backends

    def test_accessing_a_driver_name_does_not_raise(self) -> None:
        import bunkershot3d

        # Merely resolving the attribute is enough to prove __getattr__
        # works; it necessarily imports backends (and therefore mujoco) as
        # a side effect, which is expected and fine at point of *use*.
        assert bunkershot3d.MPMDriver is not None

    def test_unknown_attribute_still_raises_attribute_error(self) -> None:
        import bunkershot3d

        with pytest.raises(AttributeError):
            bunkershot3d.__getattr__("definitely_not_a_real_attribute")
