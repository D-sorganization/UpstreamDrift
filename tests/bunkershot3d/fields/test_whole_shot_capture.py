"""Keeping the sand field of a whole marched shot (issue #8729), headless.

These run the real F1 solver on a deliberately coarse bed. They are not
mocks: a capture that agreed with a mock and disagreed with the solver
would be worse than no test.

The claim this file holds down is the one a picture cannot make for
itself. A field captured from a **declared constant-velocity approach**
and a field captured from a **marched shot** animate identically, and
they are different claims -- sand thrown by a decelerating head is not
sand thrown by one driven through at constant speed. So the kinematics
travel in the provenance, and recording must not perturb the march it is
watching.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bunkershot3d.fields.capture import F1_KINEMATICS_NOTE
from bunkershot3d.fields.schema import FieldLayout
from bunkershot3d.fields.shotcapture import (
    WHOLE_SHOT_KINEMATICS_NOTE,
    WholeShotRecorder,
    capture_f1_shot_field,
)
from bunkershot3d.fields.standing import RetentionPolicy
from bunkershot3d.sand import PlayingCondition, playing_condition
from bunkershot3d.solvers.elements import SurfaceElements
from bunkershot3d.solvers.envelope import RefusalPolicy
from bunkershot3d.solvers.mpm.constitutive import SandContinuum
from bunkershot3d.solvers.mpm.solver import PlaneStrainMPMSolver
from bunkershot3d.solvers.mpm.wholeshot import F1ShotSettings, simulate_f1_shot
from bunkershot3d.solvers.protocol import FidelityTier, IntrusionState

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

SHOT_SPEED_M_S = 25.0
"""A real bunker shot, and 17x the published corpus limit of 1.44 m/s."""

SETTINGS = F1ShotSettings(head_mass_kg=0.30, max_time_s=0.004)


@pytest.fixture(scope="module")
def solver() -> PlaneStrainMPMSolver:
    """A deliberately coarse F1 solver, so a whole march fits in a test."""
    return PlaneStrainMPMSolver(
        material=SandContinuum.from_sand_state(
            playing_condition(PlayingCondition.FIRM)
        ),
        cell_size_m=0.008,
        effective_width_m=0.030,
        bed_depth_m=0.03,
        refusal_policy=RefusalPolicy.REPORT,
        max_steps=4000,
    )


def delivery(speed_m_s: float = SHOT_SPEED_M_S) -> IntrusionState:
    """A 40 x 16 mm sole section delivered at 20 degrees."""
    corners = np.array(
        [
            [-0.020, 0.0, -0.008],
            [0.020, 0.0, -0.008],
            [0.020, 0.0, 0.008],
            [-0.020, 0.0, 0.008],
        ]
    )
    angle = math.radians(20.0)
    return IntrusionState(
        SurfaceElements(
            corners,
            np.tile([0.0, 0.0, -1.0], (corners.shape[0], 1)),
            np.full(corners.shape[0], 4.0e-4),
        ),
        (speed_m_s * math.cos(angle), 0.0, -speed_m_s * math.sin(angle)),
        free_surface_height_m=0.0,
    )


@pytest.fixture(scope="module")
def captured(solver: PlaneStrainMPMSolver):  # type: ignore[no-untyped-def]
    """One captured whole shot, shared by every test that only reads it."""
    return capture_f1_shot_field(
        solver,
        delivery(),
        settings=SETTINGS,
        policy=RetentionPolicy(target_frames=12),
    )


class TestRecordingDoesNotChangeTheShot:
    """A watched march must be the march."""

    def test_the_trajectory_is_the_unwatched_trajectory(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        """The recorder is shown the state, never asked about it."""
        reference = simulate_f1_shot(solver, delivery(), settings=SETTINGS)
        _, watched = capture_f1_shot_field(
            solver,
            delivery(),
            settings=SETTINGS,
            policy=RetentionPolicy(target_frames=12),
        )
        np.testing.assert_allclose(watched.shot.positions_m, reference.shot.positions_m)

    def test_the_wrench_is_the_unwatched_wrench(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        reference = simulate_f1_shot(solver, delivery(), settings=SETTINGS)
        _, watched = capture_f1_shot_field(
            solver,
            delivery(),
            settings=SETTINGS,
            policy=RetentionPolicy(target_frames=12),
        )
        np.testing.assert_allclose(watched.shot.forces_n, reference.shot.forces_n)

    def test_the_recorder_keeps_no_reference_to_the_live_bed(self, captured) -> None:  # type: ignore[no-untyped-def]
        """The march advances in place; a held reference is the last frame."""
        series, _ = captured
        first = series.velocity_m_s[0]
        last = series.velocity_m_s[-1]
        assert not np.array_equal(first, last)

    def test_the_undisturbed_bed_is_the_first_frame(self, captured) -> None:  # type: ignore[no-untyped-def]
        """The reference every later frame is read against."""
        series, _ = captured
        assert float(series.time_s[0]) == 0.0
        np.testing.assert_allclose(series.velocity_m_s[0], 0.0, atol=1e-9)


class TestTheKinematicsAreWrittenDown:
    """The difference a picture cannot show."""

    def test_the_provenance_says_whole_shot_march(self, captured) -> None:  # type: ignore[no-untyped-def]
        series, _ = captured
        assert series.provenance.kinematics == WHOLE_SHOT_KINEMATICS_NOTE

    def test_it_is_not_the_declared_approach_note(self, captured) -> None:  # type: ignore[no-untyped-def]
        """The two fields animate identically and are different claims."""
        series, _ = captured
        assert series.provenance.kinematics != F1_KINEMATICS_NOTE

    def test_the_note_names_the_marched_trajectory(self) -> None:
        assert "trajectory" in WHOLE_SHOT_KINEMATICS_NOTE

    def test_the_tier_is_f1(self, captured) -> None:  # type: ignore[no-untyped-def]
        series, _ = captured
        assert series.provenance.fidelity_tier is FidelityTier.F1

    def test_a_greenside_shot_is_outside_the_published_corpus(self, captured) -> None:  # type: ignore[no-untyped-def]
        """MAX_VALIDATED_SPEED_M_S is 1.44; this is 25."""
        series, _ = captured
        assert not series.provenance.is_within_published_speed

    def test_out_of_plane_stays_refused(self, captured) -> None:  # type: ignore[no-untyped-def]
        series, _ = captured
        assert "out_of_plane" in series.provenance.refused

    def test_the_effective_width_is_declared(self, captured) -> None:  # type: ignore[no-untyped-def]
        """The volume view needs it to know what it may extrude across."""
        series, _ = captured
        assert float(series.provenance.settings["effective_width_m"]) > 0.0


class TestTheFieldIsUsable:
    """What the 3-D view will actually read."""

    def test_the_field_is_a_lattice(self, captured) -> None:  # type: ignore[no-untyped-def]
        series, _ = captured
        assert series.layout is FieldLayout.GRID
        assert series.geometry is not None
        assert series.geometry.dimension == 2

    def test_the_sand_actually_moves(self, captured) -> None:  # type: ignore[no-untyped-def]
        """The whole point: a time-resolved field with motion in it."""
        series, _ = captured
        assert series.peak_speed_m_s() > 1.0

    def test_every_frame_carries_the_body(self, captured) -> None:  # type: ignore[no-untyped-def]
        series, _ = captured
        assert series.body_outline_m is not None
        assert series.body_outline_m.shape[0] == series.n_frames

    def test_the_record_is_time_resolved(self, captured) -> None:  # type: ignore[no-untyped-def]
        series, _ = captured
        assert series.n_frames > 2
        assert np.all(np.diff(series.time_s) > 0.0)

    def test_the_shot_comes_back_from_the_same_solve(self, captured) -> None:  # type: ignore[no-untyped-def]
        """Pairing a field from one run with a trajectory from another
        is the substitution the 3-D scene refuses at the other end."""
        series, shot = captured
        assert shot.shot.fidelity_tier is series.provenance.fidelity_tier

    def test_the_retention_record_says_what_was_dropped(self, captured) -> None:  # type: ignore[no-untyped-def]
        series, _ = captured
        assert series.retention.frames_kept == series.n_frames
        assert series.retention.time_stride >= 1


class TestRefusals:
    """The capture validates its inputs rather than producing a wrong field."""

    def test_a_non_f1_solver_is_refused(self) -> None:
        with pytest.raises(Exception, match="PlaneStrainMPMSolver"):
            capture_f1_shot_field(
                object(),  # type: ignore[arg-type]
                delivery(),
                settings=SETTINGS,
            )

    def test_a_recorder_read_before_its_march_is_refused(self) -> None:
        """A lattice read off an unstarted recorder would be nothing."""
        recorder = WholeShotRecorder(RetentionPolicy())
        with pytest.raises(Exception, match="never handed a march"):
            _ = recorder.grid
