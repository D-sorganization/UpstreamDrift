"""The acceptance gate of issue #8618: nothing in this package is invented.

The GUI it replaced drew ``np.random.normal`` particles and reported
``0.5 * 0.3 * v**2`` as "Est. Force". Two properties keep that from coming
back:

1. **No random number generator appears anywhere in the GUI package.** A
   procedural preview is indistinguishable from a simulation once it is on
   screen, so the source is checked directly rather than the behaviour.
2. **The model layer imports no Qt.** Checked in a subprocess, because the
   pytest session may have imported a Qt binding for some other test and an
   in-process ``sys.modules`` check would then prove nothing. It also matters
   in practice: PyQt6 fails to load on some development machines, and the
   physics must stay runnable there.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import src.tools.bunker_shot_gui as package

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = Path(package.__file__).resolve().parent


def _child_env() -> dict[str, str]:
    """Environment giving a subprocess the same import roots pytest uses."""
    environment = dict(os.environ)
    roots = os.pathsep.join((str(REPO_ROOT), str(REPO_ROOT / "src")))
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = roots if not existing else roots + os.pathsep + existing
    return environment


def _leaked_qt_modules(script: str) -> str:
    """Run ``script`` in a clean interpreter and return the Qt modules it left.

    Args:
        script: Source to execute; it must print the leaked module names.

    Returns:
        The child's stdout, stripped.

    Raises:
        AssertionError: If the child failed.
    """
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env=_child_env(),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


#: Spellings that would reintroduce a procedural stand-in.
_FORBIDDEN = (
    "np.random",
    "numpy.random",
    "random.random",
    "random.gauss",
    "random.uniform",
    "default_rng",
    "RandomState",
)


def _sources() -> list[Path]:
    """Every Python source file in the GUI package."""
    return sorted(PACKAGE_ROOT.glob("*.py"))


def test_the_package_has_sources_to_check() -> None:
    """Guard the guard: an empty glob would make the ban vacuous."""
    assert len(_sources()) >= 5


@pytest.mark.parametrize("spelling", _FORBIDDEN)
def test_no_random_number_generator_in_the_gui_package(spelling: str) -> None:
    offenders = [
        path.name for path in _sources() if spelling in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"{spelling!r} appears in {offenders}. The bunker workbench renders the "
        "F0 solver's output; a procedural stand-in is exactly the defect #8618 "
        "was filed to remove."
    )


def test_no_dummy_kinetic_energy_force_estimate() -> None:
    """The old widget reported ``0.5 * 0.3 * v**2`` newtons. It is gone."""
    offenders = [
        path.name
        for path in _sources()
        if "0.5 * 0.3" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"the dummy force estimate survives in {offenders}"


def test_the_gui_imports_the_real_solver() -> None:
    """The complaint in #8618 was that the GUI never imports bunkershot3d."""
    model_source = (PACKAGE_ROOT / "model.py").read_text(encoding="utf-8")
    assert "from bunkershot3d.solvers import" in model_source
    assert "simulate_shot" in model_source


@pytest.mark.parametrize(
    "module",
    [
        "design",
        "field",
        "model",
        "render",
        "render3d",
        "render3d_vtk",
        "render_traces",
        "report",
        "shot3d",
        "traces",
    ],
)
def test_the_headless_layer_imports_no_qt(module: str) -> None:
    leaked = _leaked_qt_modules(
        "import sys\n"
        f"import src.tools.bunker_shot_gui.{module}\n"
        "print(','.join(sorted(n for n in sys.modules "
        "if n.split('.')[0] in {'PyQt6', 'PyQt5', 'PySide6', 'PySide2'})))\n"
    )
    assert leaked == "", (
        f"importing {module} pulled in a Qt binding: {leaked}. The workbench "
        "model must stay runnable where PyQt6 does not load."
    )


def test_render3d_vtk_import_touches_no_gl_or_mujoco() -> None:
    """Issue #9138 (PR): a bare import must reach no GL-touching library.

    ``render3d_vtk``'s own docstring says PyVista is imported lazily, inside
    ``require_pyvista``, and that importing the module is always safe. On a
    CI runner with no working GL driver that used to be false: ``import
    render3d_vtk`` crashed with ``AttributeError: 'NoneType' object has no
    attribute 'glGetError'``. Not from PyVista -- from an eager sibling
    import. ``render3d_vtk`` reaches ``bunkershot3d.solvers`` (directly, and
    via ``.bridge``/``.shot3d``), and ``bunkershot3d/__init__.py`` eagerly
    imports ``backends``, whose MPM driver eagerly ``import mujoco``s;
    ``mujoco`` touches an OpenGL/OSMesa binding (PyOpenGL's
    ``OpenGL.raw.GL._errors``) at *that* import, not at first render.

    ``test_the_headless_layer_imports_no_qt`` above only proves the process
    did not crash and leaked no Qt binding; a machine with a working GPU
    would pass it even with every sibling imported eagerly. This checks
    ``sys.modules`` directly, so it fails even where the underlying GL
    binding happens to succeed.
    """
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import src.tools.bunker_shot_gui.render3d_vtk\n"
            "watched = {\n"
            "    'pyvista', 'vtk', 'vtkmodules', 'mujoco', 'bunkershot3d', 'OpenGL',\n"
            "}\n"
            "leaked = sorted(n for n in sys.modules if n.split('.')[0] in watched)\n"
            "print(','.join(leaked))\n",
        ],
        cwd=str(REPO_ROOT),
        env=_child_env(),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    leaked = completed.stdout.strip()
    assert leaked == "", (
        f"importing render3d_vtk pulled in {leaked!r}. PyVista must stay "
        "lazy (inside require_pyvista), and every sibling import that "
        "reaches bunkershot3d/mujoco (.bridge, .render, .report, .shot3d, "
        ".traces) must be deferred into the function or method that "
        "actually needs it, not left at module scope."
    )


def test_the_package_root_imports_no_qt() -> None:
    """The launcher imports this package to register the embed adapter."""
    leaked = _leaked_qt_modules(
        "import sys\n"
        "import src.tools.bunker_shot_gui\n"
        "print(','.join(sorted(n for n in sys.modules "
        "if n.split('.')[0] in {'PyQt6', 'PyQt5', 'PySide6', 'PySide2'})))\n"
    )
    assert leaked == ""
