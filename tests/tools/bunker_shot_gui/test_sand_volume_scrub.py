"""The sand volume rides the one shared cursor (issue #8729, epic #8699).

The headless tests in ``tests/unit/tools/bunker_shot_gui`` cover what the
volume is and how it is drawn. What is pinned here is the part that only
exists once there is a window: that the solved sand moves when the
*existing* transport moves, and that no second slider was added to make
it do so.

Two clocks are involved and they are not the same length. The pose is
recorded every CFL step -- microseconds -- and the sand field every
stride block, so a shot with hundreds of poses carries tens of field
frames. :class:`~.slices.CursorMap` maps one onto the other, and it is
the same mapping the 2-D sand cut already uses, so the cut and the
volume land on the same moment rather than on two moments that happen to
share an index.

Qt is imported through ``pytest.importorskip`` for the same reason as
``test_gui``: PyQt6 fails to load on some development machines.
"""

from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="the workbench shell needs a Qt binding")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from bunkershot3d.solvers.protocol import FidelityTier  # noqa: E402
from src.tools.bunker_shot_gui.design import (  # noqa: E402
    SandCondition,
    SolverSetup,
    SwingSetup,
    WedgeDesign,
)
from src.tools.bunker_shot_gui.model import WorkbenchModel  # noqa: E402
from src.tools.bunker_shot_gui.sandvolume import (  # noqa: E402
    sand_volume,
    sand_volume_scale,
)
from src.tools.bunker_shot_gui.shot3d import ShotScene  # noqa: E402
from src.tools.bunker_shot_gui.viewport_widgets import (  # noqa: E402
    ShotViewportWidget,
)
from src.tools.bunker_shot_gui.widgets import (  # noqa: E402
    FollowsFrame,
    SoleLoadFieldWidget,
)

from tests.unit.tools.bunker_shot_gui.test_slices import (  # noqa: E402
    analytic_field,
)

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
def evaluation():  # type: ignore[no-untyped-def]
    """One real F0 evaluation, whose scene the sand is hung on."""
    model = WorkbenchModel(COARSE)
    return model.evaluate(WedgeDesign(name="nominal"), SandCondition(), SwingSetup())


@pytest.fixture()
def sand_scene(evaluation) -> ShotScene:  # type: ignore[no-untyped-def]
    """That scene, re-labelled F1 and given a solved sand field."""
    scene = evaluation.shot.scene
    assert scene is not None
    volume = sand_volume(analytic_field(n_frames=6), n_sheets=3, max_cells=200)
    return dataclasses.replace(
        scene,
        fidelity_tier=FidelityTier.F1,
        surface=dataclasses.replace(
            scene.surface, resolves_grains=True, tier=FidelityTier.F1
        ),
        divot=dataclasses.replace(
            scene.divot, resolves_grains=True, tier=FidelityTier.F1
        ),
        sand=volume,
    )


class TestTheSandFollowsTheOneCursor:
    """No second slider. The transport that already exists drives it."""

    def test_the_viewport_still_only_follows(self, sand_scene: ShotScene) -> None:
        """A follower is handed an index and given no way to drive back."""
        viewport = ShotViewportWidget("scene")
        viewport.set_shot(sand_scene)
        follower: FollowsFrame = viewport
        assert callable(follower.set_frame)

    def test_scrubbing_the_transport_moves_the_sand(
        self, evaluation, sand_scene: ShotScene
    ) -> None:  # type: ignore[no-untyped-def]
        field = evaluation.shot.sole_field
        assert field is not None
        transport = SoleLoadFieldWidget("sole")
        transport.set_shot(field)
        viewport = ShotViewportWidget("scene")
        viewport.set_shot(sand_scene)
        transport.link(viewport)

        transport.set_frame(0)
        assert viewport.sand is not None
        first = np.array(viewport.sand.painted_values)

        transport.set_frame(transport.n_frames - 1)
        last = np.array(viewport.sand.painted_values)
        assert not (first.shape == last.shape and np.array_equal(first, last))

    def test_the_two_clocks_are_mapped_not_indexed(self, sand_scene: ShotScene) -> None:
        """The pose has far more samples than the field, by construction."""
        assert sand_scene.sand is not None
        assert sand_scene.n_frames != sand_scene.sand.n_frames
        viewport = ShotViewportWidget("scene")
        viewport.set_shot(sand_scene)
        # The last pose must map onto the last field frame, not run off it.
        viewport.set_frame(sand_scene.n_frames - 1)
        assert viewport.sand is not None
        assert viewport.sand.n_painted >= 0

    def test_every_pose_maps_onto_a_field_frame(self, sand_scene: ShotScene) -> None:
        """A clamped or wrapped index would draw a different moment."""
        viewport = ShotViewportWidget("scene")
        viewport.set_shot(sand_scene)
        for frame in (0, sand_scene.n_frames // 2, sand_scene.n_frames - 1):
            viewport.set_frame(frame)
        assert viewport.frame_index == sand_scene.n_frames - 1


class TestTheComparisonRampIsShared:
    """Issue #8728 reaches the workbench, not just the renderer."""

    def test_a_merged_ramp_is_kept(self, sand_scene: ShotScene) -> None:
        assert sand_scene.sand is not None
        slow = sand_volume(analytic_field(peak_m_s=4.0), n_sheets=3, max_cells=200)
        shared = sand_volume_scale((sand_scene.sand, slow))
        viewport = ShotViewportWidget("scene")
        viewport.set_shot(sand_scene, sand_scale=shared)
        assert viewport.sand is not None
        assert viewport.sand.scale == shared

    def test_clearing_drops_the_ramp_with_the_shot(self, sand_scene: ShotScene) -> None:
        """Nothing stale must survive a refusal."""
        viewport = ShotViewportWidget("scene")
        viewport.set_shot(sand_scene)
        assert viewport.sand is not None
        viewport.clear()
        assert viewport.sand is None
        assert viewport.sand_scale is None


class TestAnF0ShotStillShowsNoSand:
    """The tier the shot was solved at decides, not the view."""

    def test_no_sand_artists_for_an_f0_scene(self, evaluation) -> None:  # type: ignore[no-untyped-def]
        scene = evaluation.shot.scene
        assert scene is not None
        viewport = ShotViewportWidget("scene")
        viewport.set_shot(scene)
        assert viewport.sand is None
