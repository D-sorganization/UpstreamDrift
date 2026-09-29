"""Drawing the sand volume in the VTK/PyVista scene (issue #8729).

The VTK backend is the optional upgrade: ``pip install
'upstream-drift[viz3d]'``. These tests skip without it, and
``test_render3d_vtk_degradation`` covers the without-it path deliberately
*not* skipping.

What matters here is that the upgrade and the fallback show the *same*
volume, qualified the same way. Two renderers that drew the same field
differently -- or, worse, qualified it differently -- would make the
choice of backend a claim about the physics, which it must never be.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="the VTK/PyVista viz3d extra is optional")

from bunkershot3d.fields.schema import FieldQuantity  # noqa: E402
from src.tools.bunker_shot_gui.render3d import draw_scene_frame  # noqa: E402
from src.tools.bunker_shot_gui.render3d_vtk import (  # noqa: E402
    VtkSceneArtists,
    shot_scene_still_vtk,
)
from src.tools.bunker_shot_gui.sandvolume import sand_volume_scale  # noqa: E402
from src.tools.bunker_shot_gui.shot3d import CameraPreset, ShotScene  # noqa: E402

from .test_scene_sand import f1_scene  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_pyvista]


class TestTheUpgradeDrawsTheSameVolume:
    """Choosing a backend must not be a claim about the physics."""

    def test_the_sand_mesh_is_built(self, nominal_scene, nominal_build) -> None:  # type: ignore[no-untyped-def]
        artists = shot_scene_still_vtk(f1_scene(nominal_scene), nominal_build)
        assert artists._sand_mesh is not None
        artists.close()

    def test_an_f0_scene_builds_no_sand_mesh(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:  # type: ignore[no-untyped-def]
        artists = shot_scene_still_vtk(nominal_scene, nominal_build)
        assert artists._sand_mesh is None
        artists.close()

    def test_one_quad_per_lattice_cell_per_sheet(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:  # type: ignore[no-untyped-def]
        scene = f1_scene(nominal_scene)
        artists = shot_scene_still_vtk(scene, nominal_build)
        volume = scene.sand
        assert volume is not None
        expected = volume.n_along * volume.n_up * volume.n_sheets
        assert artists._sand_mesh.n_cells == expected
        artists.close()

    def test_empty_cells_are_nan_so_nothing_is_painted_over_air(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:  # type: ignore[no-untyped-def]
        """nan_opacity draws them as nothing; a floor colour would be a lie."""
        artists = shot_scene_still_vtk(f1_scene(nominal_scene), nominal_build)
        values = np.asarray(artists._sand_mesh.cell_data["sand"])
        assert np.any(np.isnan(values))
        artists.close()

    def test_the_painted_cells_carry_real_speeds(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:  # type: ignore[no-untyped-def]
        artists = shot_scene_still_vtk(f1_scene(nominal_scene), nominal_build)
        values = np.asarray(artists._sand_mesh.cell_data["sand"])
        live = values[np.isfinite(values)]
        assert live.size > 0
        assert float(live.max()) > 0.0
        artists.close()

    def test_both_backends_paint_the_same_cell_count(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:
        """The fallback and the upgrade must agree on what is moving."""
        from matplotlib.figure import Figure

        scene = f1_scene(nominal_scene)
        volume = scene.sand
        assert volume is not None
        ramp = sand_volume_scale((volume,))
        frame = 0

        figure = Figure(figsize=(6.0, 5.0))
        mpl = draw_scene_frame(figure, scene, frame=frame, sand_scale=ramp)
        assert mpl.sand is not None

        vtk = shot_scene_still_vtk(scene, nominal_build, frame=frame, sand_scale=ramp)
        values = np.asarray(vtk._sand_mesh.cell_data["sand"])
        assert int(np.isfinite(values).sum()) == mpl.sand.n_painted
        vtk.close()


class TestTheUpgradeQualifiesItTheSameWay:
    """A fallback and an upgrade that disagreed would be worse than either."""

    def test_the_note_says_extruded(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:  # type: ignore[no-untyped-def]
        artists = shot_scene_still_vtk(f1_scene(nominal_scene), nominal_build)
        assert "extrud" in artists.note_text.lower()
        artists.close()

    def test_the_note_names_the_tier(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:  # type: ignore[no-untyped-def]
        artists = shot_scene_still_vtk(f1_scene(nominal_scene), nominal_build)
        assert "F1" in artists.note_text
        artists.close()

    def test_an_f0_note_still_denies_grains(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:  # type: ignore[no-untyped-def]
        artists = shot_scene_still_vtk(nominal_scene, nominal_build)
        assert "resolves no grains" in artists.note_text
        artists.close()

    def test_the_validity_stamp_survives(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:  # type: ignore[no-untyped-def]
        artists = shot_scene_still_vtk(f1_scene(nominal_scene), nominal_build)
        assert "not calibrated for bunker sand" in artists.stamp_text
        artists.close()

    def test_the_note_matches_the_matplotlib_backend(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:
        """Both compose it from ShotScene.sand_note, and must keep doing so.

        Compared on words rather than characters: this backend folds long
        lines to the render target's width, because PyVista draws text
        unwrapped and a sentence wider than the window is simply lost off
        the right edge. Where the fold lands is a rendering detail; what
        the two say is not.
        """
        scene = f1_scene(nominal_scene)
        artists = shot_scene_still_vtk(scene, nominal_build)
        drawn = " ".join(artists.note_text.split())
        for line in scene.sand_note():
            assert " ".join(line.split()) in drawn
        artists.close()

    def test_the_edge_on_warning_reaches_the_frame(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:
        """Down the line the sheets go edge-on and show as stripes."""
        artists = shot_scene_still_vtk(
            f1_scene(nominal_scene),
            nominal_build,
            camera=CameraPreset.DOWN_THE_LINE,
        )
        assert "edge-on" in " ".join(artists.note_text.split())
        artists.close()

    def test_a_square_on_camera_says_so_instead(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:
        artists = shot_scene_still_vtk(
            f1_scene(nominal_scene), nominal_build, camera=CameraPreset.FACE_ON
        )
        assert "square to the solved plane" in " ".join(artists.note_text.split())
        artists.close()

    def test_no_caption_line_runs_off_the_frame(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:
        """PyVista draws text unwrapped; an unfolded line is lost, not clipped."""
        artists = shot_scene_still_vtk(f1_scene(nominal_scene), nominal_build)
        assert max(len(line) for line in artists.note_text.splitlines()) <= 140
        artists.close()


class TestNothingAutoScales:
    """Issue #8728, in the upgrade."""

    def test_the_colour_limits_come_from_the_injected_ramp(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:
        scene = f1_scene(nominal_scene)
        volume = scene.sand
        assert volume is not None
        ramp = sand_volume_scale((volume,))
        artists = shot_scene_still_vtk(scene, nominal_build, sand_scale=ramp)
        assert artists._sand_scale == ramp
        artists.close()

    def test_a_merged_ramp_survives_into_the_frame(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:
        """A comparison must be painted on the covering ramp, not its own."""
        scene = f1_scene(nominal_scene)
        volume = scene.sand
        assert volume is not None
        own = sand_volume_scale((volume,))
        wider = own.merged(
            type(own)(speed_m_s=(0.0, own.speed_m_s[1] * 4.0), density_kg_m3=(0.0, 1.0))
        )
        artists = shot_scene_still_vtk(scene, nominal_build, sand_scale=wider)
        assert artists._sand_scale.speed_m_s[1] > own.speed_m_s[1]
        artists.close()

    def test_the_ramp_is_the_slice_views_own(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:
        """A speed must read the same colour in the cut and in the volume."""
        from src.tools.bunker_shot_gui.slices import SPEED_COLORMAP

        scene = f1_scene(nominal_scene)
        volume = scene.sand
        assert volume is not None
        ramp = sand_volume_scale((volume,))
        assert ramp.colormap_name(FieldQuantity.VELOCITY) == SPEED_COLORMAP


class TestEveryCameraRenders:
    """The stills a batch sweep writes out."""

    @pytest.mark.parametrize("camera", list(CameraPreset))
    def test_a_frame_comes_out(
        self, nominal_scene: ShotScene, nominal_build, camera: CameraPreset
    ) -> None:
        artists = shot_scene_still_vtk(
            f1_scene(nominal_scene), nominal_build, camera=camera
        )
        image = artists.image_array()
        assert image.ndim == 3
        assert image.shape[0] > 0
        artists.close()

    def test_updating_to_another_frame_repaints_the_sand(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:
        scene = f1_scene(nominal_scene)
        artists = VtkSceneArtists(scene, nominal_build, _scale(scene))
        artists.update(0)
        first = np.asarray(artists._sand_mesh.cell_data["sand"]).copy()
        artists.update(scene.n_frames - 1)
        last = np.asarray(artists._sand_mesh.cell_data["sand"])
        assert not np.array_equal(np.nan_to_num(first), np.nan_to_num(last))
        artists.close()


def _scale(scene: ShotScene):  # type: ignore[no-untyped-def]
    """The world box one scene needs."""
    from src.tools.bunker_shot_gui.render3d import scene_scale

    return scene_scale((scene,))
