"""The cross-tier view inside the Qt workbench (issue #8713, epic #8699).

The arithmetic and the drawing are covered headlessly in ``test_crosstier``
and ``test_render_crosstier``. What is pinned here is what only exists once
there is a window:

* the view **follows** the transport the sole load field already owns, and
  owns none of its own -- a second slider would let the panels drift apart,
  which is the one thing linking them exists to prevent;
* it is **empty until asked**, because a cross-tier check is minutes of F1
  marching and must not be on the path of every shot;
* clearing it stops nothing animating under a stale verdict.

Qt is imported through ``pytest.importorskip`` for the same reason as
``test_gui``: PyQt6 fails to load on some development machines.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="the workbench shell needs a Qt binding")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from bunkershot3d.solvers import EnvelopeStatus  # noqa: E402
from src.tools.bunker_shot_gui.crosstier import CrossTierComparison  # noqa: E402
from src.tools.bunker_shot_gui.traces import ValidityBand  # noqa: E402
from src.tools.bunker_shot_gui.viewport_widgets import (  # noqa: E402
    CrossTierWidget,
)
from src.tools.bunker_shot_gui.widgets import (  # noqa: E402
    FollowsFrame,
    SoleLoadFieldWidget,
)
from tests.tools.bunker_shot_gui.test_crosstier import probe  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """One offscreen QApplication for the module."""
    application = QApplication.instance()
    if application is None:
        application = QApplication(sys.argv[:1])
    return application


def comparison() -> CrossTierComparison:
    """A comparison over a synthetic 9-sample F0 record."""
    time_s = np.linspace(0.0, 8.0e-3, 9)
    force = np.zeros((9, 3))
    force[:, 0] = -np.linspace(10.0, 50.0, 9)
    force[:, 2] = np.linspace(10.0, 50.0, 9)
    return CrossTierComparison(
        shot_probes=(
            probe(2, 2.0e-3, f0_force_n=20.0, f1_force_n=22.0),
            probe(6, 6.0e-3, f0_force_n=40.0, f1_force_n=120.0),
        ),
        time_s=time_s,
        f0_force_n=force,
        f0_sole_depth_m=np.linspace(0.0, 0.012, 9),
        f0_velocity_m_s=np.stack(
            [
                np.linspace(25.0, 21.0, 9),
                np.zeros(9),
                np.zeros(9),
            ],
            axis=1,
        ),
        f0_divot_section_area_m2=np.linspace(0.0, 6.0e-4, 9),
        band=ValidityBand(
            time_s=time_s, statuses=tuple([EnvelopeStatus.BEYOND_VALIDATION] * 9)
        ),
        head_mass_kg=0.300,
        declared_width_m=0.030,
        bulk_density_kg_m3=1550.0,
        f1_cell_size_m=0.002,
    )


class TestTheViewOwnsNoTransport:
    def test_it_is_accepted_where_a_follower_is_asked_for(self, qapp) -> None:  # type: ignore[no-untyped-def]
        """Structural, since ``FollowsFrame`` is not runtime-checkable."""
        follower: FollowsFrame = CrossTierWidget("check")
        assert callable(follower.set_frame)

    def test_the_transport_actually_drives_it(self, qapp) -> None:  # type: ignore[no-untyped-def]
        """The link is wired, not merely accepted."""
        field_view = SoleLoadFieldWidget("A")
        view = CrossTierWidget("check")
        view.set_comparison(comparison())
        field_view.link(view)
        field_view.frame_changed.emit(3)
        assert view.frame_index == 3

    def test_it_owns_no_way_to_drive_the_transport_back(self, qapp) -> None:  # type: ignore[no-untyped-def]
        """Two views driving each other is what the narrow protocol prevents."""
        view = CrossTierWidget("check")
        assert not hasattr(view, "play")
        assert not hasattr(view, "advance")

    def test_a_frame_arriving_before_a_comparison_is_ignored(self, qapp) -> None:  # type: ignore[no-untyped-def]
        """The workbench clears views independently of the transport."""
        CrossTierWidget("check").set_frame(3)

    def test_a_frame_outside_the_record_is_refused(self, qapp) -> None:  # type: ignore[no-untyped-def]
        view = CrossTierWidget("check")
        view.set_comparison(comparison())
        with pytest.raises(ValueError, match="outside"):
            view.set_frame(99)


class TestTheViewIsEmptyUntilAsked:
    """A cross-tier check is minutes of F1 marching, not a per-shot view."""

    def test_it_starts_with_nothing_and_says_so(self, qapp) -> None:  # type: ignore[no-untyped-def]
        view = CrossTierWidget("check")
        assert not view.has_comparison
        assert view.n_frames == 0
        assert "not been run" in view.status_text.lower()

    def test_loading_a_comparison_opens_on_the_peak_probe(self, qapp) -> None:  # type: ignore[no-untyped-def]
        view = CrossTierWidget("check")
        view.set_comparison(comparison())
        assert view.has_comparison
        assert view.frame_index == 6

    def test_clearing_drops_it(self, qapp) -> None:  # type: ignore[no-untyped-def]
        view = CrossTierWidget("check")
        view.set_comparison(comparison())
        view.clear()
        assert not view.has_comparison
        assert view.n_frames == 0


class TestTheViewRestatesWhatItDoesNotLicense:
    def test_the_readout_carries_the_licence_not_only_the_figure(
        self,
        qapp,  # type: ignore[no-untyped-def]
    ) -> None:
        view = CrossTierWidget("check")
        view.set_comparison(comparison())
        assert "not validation" in view.status_text.lower()

    def test_following_the_cursor_keeps_the_licence_on_screen(self, qapp) -> None:  # type: ignore[no-untyped-def]
        view = CrossTierWidget("check")
        view.set_comparison(comparison())
        view.set_frame(2)
        assert "not validation" in view.status_text.lower()
        assert "2.00" in view.status_text
