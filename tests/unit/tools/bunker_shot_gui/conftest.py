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

from src.tools.bunker_shot_gui.design import (
    SandCondition,
    SolverSetup,
    SwingSetup,
    WedgeDesign,
)
from src.tools.bunker_shot_gui.model import WorkbenchModel


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
    """A tour greenside delivery: 25 m/s, -8 deg, face open, shaft leaning."""
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
def nominal_evaluation(model, nominal_design, firm_sand, tour_swing):  # type: ignore[no-untyped-def]
    """The nominal design evaluated end to end, playability included."""
    return model.evaluate(nominal_design, firm_sand, tour_swing)
