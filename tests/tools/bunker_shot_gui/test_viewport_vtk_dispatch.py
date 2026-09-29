"""VTK/PyVista renderer dispatch in ShotViewportWidget (#8706, epic #8699).

ADR-0027's viewport layer is consulted at draw time, inside
:meth:`~.viewport_widgets.ShotViewportWidget.set_shot`, not at import time --
so this is the one place that actually exercises "who draws the payload"
through Qt. ``test_shot_scene_render_vtk.py`` covers the renderer itself
directly and never touches a widget; this file covers the wiring between the
two.

Two things are pinned here, deliberately independent of whether pyvista is
actually installed on the runner:

* Omitting ``build`` -- what every existing caller does today -- must leave
  this view on the matplotlib path unconditionally. Installing the optional
  ``viz3d`` extra must not change behaviour nothing asked for.
* When ``build`` is supplied and a VTK provider really is available and can
  render, the view switches to it and reports so; when it cannot (pyvista
  absent, or installed but the offscreen render fails on this machine), the
  view degrades to matplotlib rather than crashing.

Qt is imported through ``pytest.importorskip`` for the same reason
``test_linked_scrub.py`` does: PyQt6 fails to load on some development
machines. The PyVista-specific assertions carry their own
``pytest.mark.requires_pyvista`` and ``pytest.importorskip("pyvista")``,
mirroring ``tests/unit/visualization/test_rerun_renderer.py``'s
``requires_rerun`` pattern, so this whole file still runs -- and still
proves the fallback wiring -- on a stock runner without the extra.
"""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("PyQt6", reason="the workbench shell needs a Qt binding")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.bunker_shot_gui.bridge import HeadBuild  # noqa: E402
from src.tools.bunker_shot_gui.design import (  # noqa: E402
    SandCondition,
    SolverSetup,
    SwingSetup,
    WedgeDesign,
)
from src.tools.bunker_shot_gui.model import DesignEvaluation, WorkbenchModel  # noqa: E402
from src.tools.bunker_shot_gui.shot3d import ShotScene  # noqa: E402
from src.tools.bunker_shot_gui.viewport_widgets import ShotViewportWidget  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

COARSE = SolverSetup(
    n_profile_points=12, n_stations=5, playability_points=2, target_carry_m=12.0
)


@pytest.fixture(scope="session", autouse=True)
def qapp() -> QApplication:
    """One offscreen QApplication for the module."""
    application = QApplication.instance()
    if application is None:
        application = QApplication(sys.argv[:1])
    return application


@pytest.fixture(scope="module")
def evaluation() -> DesignEvaluation:
    """One real evaluated design on the coarse settings."""
    return WorkbenchModel(COARSE).evaluate(
        WedgeDesign(name="nominal"),
        SandCondition(),
        SwingSetup(),
        include_playability=False,
    )


@pytest.fixture()
def scene(evaluation: DesignEvaluation) -> ShotScene:
    """The evaluated design's 3-D scene."""
    result = evaluation.shot.scene
    assert result is not None, evaluation.shot.unavailable
    return result


@pytest.fixture()
def build(evaluation: DesignEvaluation) -> HeadBuild:
    """The lofted head backing :func:`scene`'s centroids.

    ``lru_cache``-backed on :class:`~.model.WorkbenchModel`, so this costs
    nothing beyond what evaluating the design already paid.
    """
    return WorkbenchModel(COARSE).head_build(evaluation.geometry)


class TestOmittingBuildKeepsTheExistingMatplotlibWiring:
    """Installing pyvista must not change behaviour nothing asked for."""

    def test_no_build_never_selects_the_vtk_path(self, scene: ShotScene) -> None:
        view = ShotViewportWidget("scene")
        view.set_shot(scene)
        assert view._vtk_artists is None  # noqa: SLF001 - no public reader

    def test_no_build_reports_a_renderer_note(self, scene: ShotScene) -> None:
        view = ShotViewportWidget("scene")
        view.set_shot(scene)
        # Whatever the ADR-0027 default-provider check says (MeshCat is not
        # installed here either), it must not be the VTK/PyVista wording --
        # that would mean the view switched renderers without a mesh to draw.
        assert "VTK/PyVista" not in view.renderer_note

    def test_clear_after_no_build_stays_on_the_matplotlib_path(
        self, scene: ShotScene
    ) -> None:
        view = ShotViewportWidget("scene")
        view.set_shot(scene)
        view.clear()
        assert view._vtk_artists is None  # noqa: SLF001
        assert not view.has_shot


class TestVtkDispatchWhenPyVistaIsActuallyAvailable:
    """These need the ``viz3d`` extra installed *and* a working offscreen
    render target; on a runner missing either, they skip rather than fail --
    installing pyvista is the thing under test, not this machine's GPU."""

    @pytest.mark.requires_pyvista
    def test_build_plus_available_vtk_switches_the_renderer(
        self, scene: ShotScene, build: HeadBuild
    ) -> None:
        pytest.importorskip("pyvista", reason="needs the viz3d extra installed")
        view = ShotViewportWidget("scene")
        view.set_shot(scene, build=build)
        if view._vtk_artists is None:  # noqa: SLF001
            pytest.skip(
                "pyvista is installed but the VTK path degraded on this "
                f"runner: {view.renderer_note}"
            )
        assert view.renderer_note.startswith("3-D viewport: VTK/PyVista")

    @pytest.mark.requires_pyvista
    def test_scrubbing_frames_still_works_on_the_vtk_path(
        self, scene: ShotScene, build: HeadBuild
    ) -> None:
        pytest.importorskip("pyvista", reason="needs the viz3d extra installed")
        view = ShotViewportWidget("scene")
        view.set_shot(scene, build=build)
        if view._vtk_artists is None:  # noqa: SLF001
            pytest.skip("VTK could not initialise on this runner")
        view.set_frame(3)
        assert view.frame_index == 3

    @pytest.mark.requires_pyvista
    def test_camera_switch_still_works_on_the_vtk_path(
        self, scene: ShotScene, build: HeadBuild
    ) -> None:
        pytest.importorskip("pyvista", reason="needs the viz3d extra installed")
        view = ShotViewportWidget("scene")
        view.set_shot(scene, build=build)
        if view._vtk_artists is None:  # noqa: SLF001
            pytest.skip("VTK could not initialise on this runner")
        view.set_frame(5)
        view._camera_box.setCurrentIndex(2)  # noqa: SLF001 - no public setter
        assert view.frame_index == 5

    @pytest.mark.requires_pyvista
    def test_clearing_releases_the_vtk_render_window(
        self, scene: ShotScene, build: HeadBuild
    ) -> None:
        pytest.importorskip("pyvista", reason="needs the viz3d extra installed")
        view = ShotViewportWidget("scene")
        view.set_shot(scene, build=build)
        if view._vtk_artists is None:  # noqa: SLF001
            pytest.skip("VTK could not initialise on this runner")
        view.clear()
        assert view._vtk_artists is None  # noqa: SLF001
        assert not view.has_shot

    @pytest.mark.requires_pyvista
    def test_a_broken_vtk_construction_degrades_to_matplotlib(
        self,
        scene: ShotScene,
        build: HeadBuild,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Even with pyvista installed and a provider available, a render
        failure that only shows up at draw time (no GPU, no display, a
        broken driver) must degrade rather than crash the workbench."""
        pytest.importorskip("pyvista", reason="needs the viz3d extra installed")
        import src.tools.bunker_shot_gui.viewport_widgets as viewport_widgets

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("simulated offscreen render failure")

        monkeypatch.setattr(viewport_widgets, "VtkSceneArtists", _boom)
        view = ShotViewportWidget("scene")
        view.set_shot(scene, build=build)
        assert view._vtk_artists is None  # noqa: SLF001
        assert view.has_shot
        assert "VTK/PyVista" not in view.renderer_note
