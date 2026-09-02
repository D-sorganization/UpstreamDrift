"""Shared fixtures for the BunkerShot3D workbench model tests (issue #8618).

Every fixture here is headless: nothing in this directory imports Qt, and
``test_no_fake_physics`` proves it in a subprocess.

The settings are deliberately the coarsest the underlying geometry package
accepts. Lofting a wedge costs about a second because the camber segment is
solved by root-finding per station, so a coarse mesh and a 2x2 playability
grid keep the suite in the tens of milliseconds per test while still running
the *real* F0 solver -- these are not mocks.
"""

from __future__ import annotations

import pytest

from src.tools.bunker_shot_gui.bridge import HeadBuild
from src.tools.bunker_shot_gui.design import (
    SandCondition,
    SolverSetup,
    SwingSetup,
    WedgeDesign,
)
from src.tools.bunker_shot_gui.model import WorkbenchModel
from src.tools.bunker_shot_gui.shot3d import ShotScene


@pytest.fixture(scope="session")
def coarse_settings() -> SolverSetup:
    """The cheapest settings the geometry package accepts."""
    return SolverSetup(
        n_profile_points=12,
        n_stations=5,
        playability_points=2,
        target_carry_m=12.0,
    )


@pytest.fixture(scope="session")
def model(coarse_settings: SolverSetup) -> WorkbenchModel:
    """A workbench model on the coarse settings."""
    return WorkbenchModel(coarse_settings)


@pytest.fixture(scope="session")
def nominal_design() -> WedgeDesign:
    """The archetypal greenside wedge: a 58 deg crescent sole."""
    return WedgeDesign(name="nominal")


@pytest.fixture(scope="session")
def firm_sand() -> SandCondition:
    """A firm USGA bunker."""
    return SandCondition()


@pytest.fixture(scope="session")
def tour_swing() -> SwingSetup:
    """A tour greenside delivery: 25 m/s, -6 deg, face open, shaft neutral.

    This used to lean the shaft 6 deg forward, which issue #9247 showed
    is a full-shot delivery that buries the head once the delivery frame
    is un-mirrored. See :class:`~.design.SwingSetup` for the measurements.
    """
    return SwingSetup()


@pytest.fixture(scope="session")
def quasi_static_swing() -> SwingSetup:
    """The same delivery with the DRFT inertial term switched off.

    Above ``Fr ~ 1`` the envelope refuses to report a force for a
    quasi-static solver, so this is the fixture that exercises refusal.
    """
    return SwingSetup(dynamic_terms_active=False)


@pytest.fixture(scope="session")
def nominal_shot(model, nominal_design, firm_sand, tour_swing):  # type: ignore[no-untyped-def]
    """One real F0 shot for the nominal design."""
    return model.run_shot(nominal_design.geometry(), firm_sand.sand_state(), tour_swing)


@pytest.fixture(scope="session")
def nominal_build(model, nominal_design) -> HeadBuild:  # type: ignore[no-untyped-def]
    """The lofted head :func:`nominal_shot` was solved against.

    Free of the solver: :meth:`~.model.WorkbenchModel.head_build` is
    ``lru_cache``-backed, so this is the same object ``nominal_shot`` and
    ``nominal_evaluation`` already forced -- not a second loft. A
    :class:`~.bridge.HeadBuild` is what a real 3-D renderer needs beyond the
    scene's own centroids: the watertight mesh they came from.
    """
    return model.head_build(nominal_design.geometry())


@pytest.fixture(scope="session")
def nominal_scene(nominal_shot) -> ShotScene:  # type: ignore[no-untyped-def]
    """The 3-D scene of the nominal shot (issue #8706)."""
    scene = nominal_shot.scene
    assert scene is not None, nominal_shot.unavailable
    return scene


@pytest.fixture(scope="session")
def decelerating_swing() -> SwingSetup:
    """A delivery slow enough that the sand slows it out of one regime.

    ``MAX_VALIDATED_SPEED_M_S`` is 1.44 m/s, so a head delivered at 1.5 m/s
    starts past the published corpus and drops back inside it partway
    through the strike. It is the fixture that exercises a validity band
    which actually changes (issue #8708); a greenside 25 m/s delivery never
    returns to the corpus and so is uniform.
    """
    return SwingSetup(clubhead_speed_mps=1.5)


@pytest.fixture(scope="session")
def decelerating_shot(model, nominal_design, firm_sand, decelerating_swing):  # type: ignore[no-untyped-def]
    """One real F0 shot that changes envelope regime mid-record."""
    return model.run_shot(
        nominal_design.geometry(), firm_sand.sand_state(), decelerating_swing
    )


@pytest.fixture(scope="session")
def decelerating_traces(decelerating_shot):  # type: ignore[no-untyped-def]
    """The scalar traces of the shot that changes regime."""
    traces = decelerating_shot.traces
    assert traces is not None, decelerating_shot.unavailable
    return traces


@pytest.fixture(scope="session")
def nominal_evaluation(model, nominal_design, firm_sand, tour_swing):  # type: ignore[no-untyped-def]
    """The nominal design evaluated end to end, playability included."""
    return model.evaluate(nominal_design, firm_sand, tour_swing)
