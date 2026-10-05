"""Contract tests for the versioned BunkerShot3D workbench route (issue #9545).

``tools.bunkershot3d_workbench`` had ``api: null`` in the feature-parity
registry. The v1 route is a thin client of the same headless
:class:`~src.tools.bunker_shot_gui.model.WorkbenchModel` the PyQt workbench
drives, so these tests pin four things:

* the route is versioned under the established ``/tools/<name>/vN`` pattern;
* a nominal solve returns the tier, the verdict, the in-frame stamp, the
  source stamps and a digest, and the stamp and digest equal what the model
  path produces for the same inputs;
* invalid input -- wrong units, out-of-range values, an unstated or
  unsupported handedness, an unconstructible design -- is a precise 4xx; and
* a playability objective can never be ranked as predictive (#9239).
"""

from __future__ import annotations

import re
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api.routes.bunker_workbench import (  # noqa: E402
    SCHEMA_VERSION,
    evaluation_record,
    record_digest,
    router,
)

pytestmark = [pytest.mark.contract, pytest.mark.integration]

EVALUATE = "/tools/bunker-workbench/v1/evaluate"

NOMINAL: dict[str, Any] = {
    "handedness": "right",
    "design": {"name": "A", "grind_preset": "sm9_58_m"},
    "sand": {"preset": "firm"},
    "swing": {"clubhead_speed_mps": 25.0, "attack_angle_deg": -6.0},
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A bare app carrying only the workbench router."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(scope="module")
def nominal(client: TestClient) -> dict[str, Any]:
    """The nominal solve, run once for the module."""
    response = client.post(EVALUATE, json=NOMINAL)
    assert response.status_code == 200, response.text
    return response.json()


def _post(client: TestClient, **overrides: Any) -> Any:
    """POST the nominal request with top-level sections replaced."""
    return client.post(EVALUATE, json={**NOMINAL, **overrides})


def _detail_code(response: Any) -> str:
    """The machine-readable error code of a 4xx response."""
    return str(response.json()["detail"]["code"])


# --------------------------------------------------------------- versioning


def test_route_is_versioned_under_the_tools_prefix() -> None:
    paths = {route.path for route in router.routes}
    assert EVALUATE in paths
    assert not router.prefix.startswith("/api")


# ---------------------------------------------------------- nominal solve


def test_nominal_solve_carries_tier_verdict_stamp_and_digest(
    nominal: dict[str, Any],
) -> None:
    assert nominal["schema_version"] == SCHEMA_VERSION
    assert nominal["tier"] == "F0"
    assert nominal["tier_model"] == "dynamic 3D-RFT"
    assert nominal["verdict"]["status"] in {
        "within",
        "extrapolated",
        "beyond_validation",
        "refused",
    }
    assert nominal["verdict"]["headline"]
    assert "not calibrated for bunker sand" in nominal["stamp"]
    assert re.fullmatch(r"[0-9a-f]{64}", nominal["digest"])


def test_nominal_solve_stamps_its_sources_and_conventions(
    nominal: dict[str, Any],
) -> None:
    sources = nominal["sources"]
    assert sources["model"] == "src.tools.bunker_shot_gui.model.WorkbenchModel"
    assert sources["sand"], "every sand property must carry its provenance"
    for entry in sources["sand"].values():
        assert entry["basis"] and entry["source"]
    conventions = nominal["conventions"]
    assert conventions["handedness"] == "right"
    assert conventions["bounce"] == "marketed"
    assert conventions["units"]["sole_width_mm"] == "mm"


def test_nominal_solve_is_never_predictive(nominal: dict[str, Any]) -> None:
    assert nominal["predictive"] is False


def test_digest_covers_the_record_it_is_returned_with(
    nominal: dict[str, Any],
) -> None:
    body = {key: value for key, value in nominal.items() if key != "digest"}
    assert record_digest(body) == nominal["digest"]


def test_the_same_inputs_give_the_same_digest(
    client: TestClient, nominal: dict[str, Any]
) -> None:
    again = client.post(EVALUATE, json=NOMINAL).json()
    assert again["digest"] == nominal["digest"]


def test_stamp_and_digest_match_the_model_path(nominal: dict[str, Any]) -> None:
    """The PyQt workbench's model call, with the same inputs, agrees exactly."""
    pytest.importorskip("matplotlib")
    from src.tools.bunker_shot_gui.design import (
        SandCondition,
        SwingSetup,
        WedgeDesign,
    )
    from src.tools.bunker_shot_gui.model import WorkbenchModel
    from src.tools.bunker_shot_gui.render import validity_stamp

    swing = SwingSetup(clubhead_speed_mps=25.0, attack_angle_deg=-6.0)
    evaluation = WorkbenchModel().evaluate(
        WedgeDesign(name="A", grind_preset="sm9_58_m"),
        SandCondition(preset="firm"),
        swing,
        include_playability=False,
    )
    shot = evaluation.shot
    assert nominal["stamp"] == validity_stamp(shot.status, shot.fidelity_tier)
    record = evaluation_record(evaluation, swing, handedness="right", objective=None)
    assert record_digest(record) == nominal["digest"]


def test_a_refused_shot_reports_no_numbers(client: TestClient) -> None:
    """Quasi-static RFT is refused at greenside speed (ADR-0032)."""
    swing = {**NOMINAL["swing"], "dynamic_terms_active": False}
    body = _post(client, swing=swing).json()
    assert body["refused"] is True
    assert body["verdict"]["status"] == "refused"
    assert body["stamp"].startswith("REFUSED")
    assert all(value is None for value in body["shot"].values())
    assert body["carry"] is None


# ------------------------------------------------------------ invalid input


def test_unstated_handedness_is_rejected(client: TestClient) -> None:
    request = {key: value for key, value in NOMINAL.items() if key != "handedness"}
    response = client.post(EVALUATE, json=request)
    assert response.status_code == 422
    assert any(error["loc"][-1] == "handedness" for error in response.json()["detail"])


def test_left_handed_request_is_refused_not_mirrored(client: TestClient) -> None:
    response = _post(client, handedness="left")
    assert response.status_code == 422
    assert _detail_code(response) == "handedness_unsupported"


def test_a_field_in_the_wrong_unit_is_rejected(client: TestClient) -> None:
    design = {**NOMINAL["design"], "sole_width_in": 0.8}
    response = _post(client, design=design)
    assert response.status_code == 422
    assert any(
        error["loc"][-1] == "sole_width_in" for error in response.json()["detail"]
    )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("swing", "attack_angle_deg", 4.0),
        ("swing", "clubhead_speed_mps", 0.0),
        ("swing", "clubhead_speed_mps", 500.0),
        ("design", "loft_deg", 0.5),
        ("design", "sole_width_mm", -20.0),
        ("sand", "firmness_kg_per_cm2", -1.0),
    ],
)
def test_out_of_range_values_are_rejected(
    client: TestClient, section: str, field: str, value: float
) -> None:
    response = _post(client, **{section: {**NOMINAL[section], field: value}})
    assert response.status_code == 422
    assert any(error["loc"][-1] == field for error in response.json()["detail"])


def test_an_unknown_preset_is_a_precise_422(client: TestClient) -> None:
    design = {**NOMINAL["design"], "grind_preset": "no_such_grind"}
    response = _post(client, design=design)
    assert response.status_code == 422
    assert _detail_code(response) == "invalid_workbench_input"
    assert "no_such_grind" in response.json()["detail"]["message"]


# --------------------------------------------- playability objective (#9239)


def test_a_predictive_objective_is_refused(client: TestClient) -> None:
    objective = {"use": "predictive", "target_carry_m": 12.0, "tolerance_fraction": 0.1}
    response = _post(client, objective=objective)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "objective_not_predictive"
    assert detail["disposition"] == "unavailable-uncalibrated"


def test_an_exploratory_objective_is_reported_but_never_ranks(
    client: TestClient,
) -> None:
    objective = {
        "use": "exploratory",
        "target_carry_m": 12.0,
        "tolerance_fraction": 0.1,
    }
    body = _post(client, objective=objective).json()
    reported = body["objective"]
    assert reported["disposition"] == "unavailable-uncalibrated"
    assert reported["ranking_permitted"] is False
    assert body["predictive"] is False


def test_a_target_the_nominal_carry_misses_is_flagged_degenerate(
    client: TestClient,
) -> None:
    """#9239: a window that excludes the nominal design is reported, not hidden."""
    objective = {
        "use": "exploratory",
        "target_carry_m": 40.0,
        "tolerance_fraction": 0.02,
    }
    reported = _post(client, objective=objective).json()["objective"]
    assert reported["degenerate"] is True
    assert reported["ranking_permitted"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [("target_carry_m", 0.0), ("tolerance_fraction", 0.0), ("tolerance_fraction", 2.0)],
)
def test_a_degenerate_objective_definition_is_rejected(
    client: TestClient, field: str, value: float
) -> None:
    objective = {
        "use": "exploratory",
        "target_carry_m": 12.0,
        "tolerance_fraction": 0.1,
        field: value,
    }
    response = _post(client, objective=objective)
    assert response.status_code == 422
    assert any(error["loc"][-1] == field for error in response.json()["detail"])
