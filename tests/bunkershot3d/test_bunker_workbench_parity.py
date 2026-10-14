"""Parity tests between the web API and PyQt headless model (issue #12536).

Pins that the versioned web API route (/tools/bunker-workbench/v1/evaluate) and
the headless model driving the PyQt workbench produce bitwise/numerical parity
on all physical outcomes, envelope verdicts, validity stamps, and provenance.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.bunker_workbench import (
    BunkerWorkbenchRequestV1,
    SCHEMA_VERSION,
    evaluate_v1,
    record_digest,
    router,
)
from src.tools.bunker_shot_gui.design import (
    SandCondition,
    SolverSetup,
    SwingSetup,
    WedgeDesign,
)
from src.tools.bunker_shot_gui.model import WorkbenchModel
from src.tools.bunker_shot_gui.report import (
    evaluation_sources,
    status_headline,
    validity_stamp,
)

pytestmark = [pytest.mark.unit, pytest.mark.integration]


@pytest.fixture(scope="module")
def coarse_setup() -> SolverSetup:
    return SolverSetup(
        n_profile_points=12,
        n_stations=5,
        playability_points=2,
        target_carry_m=12.0,
    )


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_web_and_pyqt_data_parity() -> None:
    # 1. Evaluate through the headless model (desktop engine)
    model = WorkbenchModel()
    design = WedgeDesign(name="WedgeA", grind_preset="sm9_58_m")
    sand = SandCondition(preset="firm")
    swing = SwingSetup(clubhead_speed_mps=25.0, attack_angle_deg=-6.0)

    gui_eval = model.evaluate(design, sand, swing, include_playability=False)
    gui_shot = gui_eval.shot

    # 2. Evaluate through the web API handler
    api_request = BunkerWorkbenchRequestV1(
        handedness="right",
        design={"name": "WedgeA", "grind_preset": "sm9_58_m"},
        sand={"preset": "firm"},
        swing={"clubhead_speed_mps": 25.0, "attack_angle_deg": -6.0},
    )
    api_record = evaluate_v1(api_request)

    # 3. Assert exact parity
    assert api_record["schema_version"] == SCHEMA_VERSION
    assert api_record["tier"] == gui_shot.fidelity_tier.value
    assert api_record["verdict"]["status"] == gui_shot.status.value
    assert api_record["verdict"]["headline"] == status_headline(gui_shot.status)
    assert api_record["stamp"] == validity_stamp(
        gui_shot.status, gui_shot.fidelity_tier
    )

    # Physical outcomes
    assert api_record["shot"]["peak_force_n"] == pytest.approx(gui_shot.peak_force_n)
    assert api_record["shot"]["impulse_n_s"] == pytest.approx(gui_shot.impulse_n_s)
    assert api_record["shot"]["max_depth_m"] == pytest.approx(gui_shot.max_depth_m)
    assert api_record["shot"]["contact_duration_s"] == pytest.approx(
        gui_shot.contact_duration_s
    )

    # Delivered angles
    assert api_record["delivered"]["effective_loft_deg"] == pytest.approx(
        gui_shot.delivered.effective_loft_deg
    )
    assert api_record["delivered"]["effective_marketed_bounce_deg"] == pytest.approx(
        gui_shot.delivered.effective_bounce.angle_deg
    )
    assert api_record["delivered"]["aim_offset_deg"] == pytest.approx(
        gui_shot.delivered.aim_offset_deg
    )

    # Provenance sources
    assert api_record["sources"] == evaluation_sources(gui_eval)

    # Non-predictive contract
    assert api_record["predictive"] is False

    # Record digest
    assert api_record["digest"] == record_digest(api_record)


def test_api_route_and_direct_evaluate_parity(client: TestClient) -> None:
    payload = {
        "handedness": "right",
        "design": {"name": "WedgeA", "grind_preset": "sm9_58_m"},
        "sand": {"preset": "firm"},
        "swing": {"clubhead_speed_mps": 25.0, "attack_angle_deg": -6.0},
    }
    response = client.post("/tools/bunker-workbench/v1/evaluate", json=payload)
    assert response.status_code == 200
    route_record = response.json()

    direct_record = evaluate_v1(BunkerWorkbenchRequestV1(**payload))
    assert route_record["digest"] == direct_record["digest"]
    assert route_record["shot"]["peak_force_n"] == pytest.approx(
        direct_record["shot"]["peak_force_n"]
    )
