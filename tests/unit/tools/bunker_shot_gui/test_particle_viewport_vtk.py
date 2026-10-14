"""Drawing particles in the canonical VTK viewport (issue #9545 / #9693).

Gated on the optional ``viz3d`` extra; ``test_particle_viewport_degradation``
covers the path this file skips. Every artifact is the SYNTHETIC TEST
FIXTURE -- these tests check geometry, instancing and labels, not physics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py", reason="the trajectory artifact is HDF5")
pytest.importorskip("pyvista", reason="the VTK/PyVista viz3d extra is optional")

from bunkershot3d.fields.trajectory_reader import ParticleTrajectoryReader  # noqa: E402
from src.tools.bunker_shot_gui.particle_playback import (  # noqa: E402
    BodySource,
    BodyTrack,
    ParticlePlayback,
    measure_frame_rate,
)
from src.tools.bunker_shot_gui.particle_render_vtk import (  # noqa: E402
    ParticleViewport,
    open_particle_viewport,
)
from tests.unit.tools.bunker_shot_gui._synthetic_particle_artifact import (  # noqa: E402
    FRAME_TIMES_S,
    write_synthetic_artifact,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_pyvista]

_SIZE = (320, 240)


@pytest.fixture()
def reader(tmp_path: Path):  # type: ignore[no-untyped-def]
    path = write_synthetic_artifact(tmp_path / "synthetic.h5")
    with ParticleTrajectoryReader(path) as opened:
        yield opened


@pytest.fixture()
def viewport(reader):  # type: ignore[no-untyped-def]
    ball = BodyTrack(
        "ball",
        np.array([2.0e-4, 5.0e-4]),
        np.array([[0.0, 0.0, 0.0], [0.004, 0.0, 0.008]]),
        BodySource.SIMULATED,
    )
    opened = open_particle_viewport(
        ParticlePlayback(reader, bodies=(ball,)), window_size=_SIZE
    )
    assert isinstance(opened, ParticleViewport), opened
    yield opened
    opened.close()


def test_glyphs_are_instanced_from_one_sphere_source(viewport) -> None:  # type: ignore[no-untyped-def]
    mapper = viewport.layer.mapper
    assert mapper.IsA("vtkGlyph3DMapper")
    assert mapper.GetSourceTableTree() is None
    assert viewport.layer.glyph_count == 6


def test_glyph_points_are_the_artifact_positions_in_mm(viewport, reader) -> None:  # type: ignore[no-untyped-def]
    viewport.show_frame(4)
    frame = reader.frame(4)
    visible = viewport.layer.visible_mask()
    np.testing.assert_allclose(
        viewport.layer.points_mm()[visible], frame.positions_m[visible] * 1e3
    )
    # Grain 23 is lost at frame 4 and must be masked, not drawn at a NaN.
    assert not bool(visible[5])
    assert bool(np.all(np.isfinite(viewport.layer.points_mm())))


def test_lateral_motion_reaches_the_picture(viewport) -> None:  # type: ignore[no-untyped-def]
    viewport.show_frame(0)
    first = viewport.image_array().copy()
    lateral_first = viewport.layer.points_mm()[:, 1].copy()
    viewport.show_frame(len(FRAME_TIMES_S) - 1)
    lateral_last = viewport.layer.points_mm()[:, 1]
    assert np.ptp(lateral_last - lateral_first) > 0.0
    assert not np.array_equal(first, viewport.image_array())


def test_the_stamp_is_drawn_and_names_the_fixture(viewport) -> None:  # type: ignore[no-untyped-def]
    assert "SYNTHETIC TEST FIXTURE" in viewport.stamp_text
    assert "6 of 6" in viewport.stamp_text


def test_the_ball_is_hidden_while_it_has_no_data(viewport) -> None:  # type: ignore[no-untyped-def]
    viewport.show_frame(0)
    assert not viewport.layer.body_visible("ball")
    viewport.show_frame(4)
    assert viewport.layer.body_visible("ball")


def test_frame_rate_is_measured_on_real_renders(viewport) -> None:  # type: ignore[no-untyped-def]
    report = measure_frame_rate(viewport.show_frame, range(len(FRAME_TIMES_S)))
    assert report.n_frames == len(FRAME_TIMES_S)
    assert report.median_fps > 0.0


def test_the_layer_attaches_to_an_existing_plotter(reader) -> None:  # type: ignore[no-untyped-def]
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True, window_size=list(_SIZE))
    try:
        opened = open_particle_viewport(ParticlePlayback(reader), plotter=plotter)
        assert isinstance(opened, ParticleViewport)
        assert opened.plotter is plotter
    finally:
        plotter.close()


def test_the_layer_joins_the_canonical_shot_viewport(
    reader, nominal_scene, nominal_build
) -> None:  # type: ignore[no-untyped-def]
    """Particles draw into the shot scene's own plotter, beside the head."""
    from src.tools.bunker_shot_gui.render3d_vtk import draw_scene_frame_vtk

    artists = draw_scene_frame_vtk(nominal_scene, nominal_build, window_size=_SIZE)
    try:
        opened = open_particle_viewport(
            ParticlePlayback(reader), plotter=artists.plotter
        )
        assert isinstance(opened, ParticleViewport)
        opened.show_frame(2)
        assert "particles" in artists.plotter.actors
        assert int(artists.image_array().std()) > 0
    finally:
        artists.close()
