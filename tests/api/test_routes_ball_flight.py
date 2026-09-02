"""Tests for the ball-flight REST route."""

from __future__ import annotations

import importlib
import json
import math
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.routes.ball_flight import BallFlightSimulationRequest, router
from src.shared.python.physics.flight_models import FlightModelType
from src.shared.python.physics.flight_trajectory_export import (
    APP_FRAME_ID,
    FLIGHT_FRAME_ID,
    flight_result_to_trajectory_record,
    trajectory_parameter_digest,
)

pytestmark = pytest.mark.integration

_VENDOR_INTERCHANGE = "shared.python.swing_sim.flight_interchange"


def _flight_interchange_available() -> bool:
    """Whether the pinned ``vendor/ud-tools`` checkout carries the H1 reader.

    Mirrors ``tests/unit/physics/test_flight_trajectory_export.py``'s
    ``_vendor_interchange`` skip: absent on a pin predating the Tools half
    of ADR-0047 H1 (D-sorganization/Tools#4888, landing here via #9363).
    The import endpoint's own tests below arm themselves automatically on
    the next vendor pin bump, with no edit needed here.

    Deliberately imports the actual reader symbol rather than only
    checking ``find_spec``: an empty or stale ``flight_interchange``
    directory (e.g. a leftover ``__pycache__`` from a prior checkout, with
    no ``__init__.py``) still resolves as a valid Python 3 namespace
    package, which would report "available" without the module actually
    working.
    """
    try:
        module = importlib.import_module(_VENDOR_INTERCHANGE)
    except ImportError:
        return False
    return hasattr(module, "ball_flight_trajectory_from_json")


requires_flight_interchange = pytest.mark.skipif(
    not _flight_interchange_available(),
    reason=(
        f"{_VENDOR_INTERCHANGE} is absent from the pinned vendor/ud-tools "
        "tree (pin predates Tools#4888, or the submodule is not "
        "materialised); POST /tools/ball-flight/import's tests arm on the "
        "next pin bump"
    ),
)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _driver_payload(model_name: str) -> dict[str, float | str]:
    return {
        "ball_speed_mps": 70.0,
        "launch_angle_deg": 12.0,
        "azimuth_angle_deg": 1.0,
        "spin_rate_rpm": 2600.0,
        "spin_axis_tilt_deg": -2.0,
        "wind_speed_mps": 0.0,
        "wind_direction_deg": 0.0,
        "model_name": model_name,
        "max_time_s": 1.0,
        "time_step_s": 0.05,
    }


@pytest.mark.parametrize("model_type", list(FlightModelType))
def test_simulate_ball_flight_happy_path_per_model(
    client: TestClient, model_type: FlightModelType
) -> None:
    response = client.post(
        "/tools/ball-flight/simulate", json=_driver_payload(model_type.value)
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model_name"]
    assert data["trajectory"]
    assert data["summary"]["carry_m"] > 0.0
    assert data["summary"]["apex_m"] > 0.0
    assert data["summary"]["flight_time_s"] > 0.0
    assert "lateral_deviation_m" in data["summary"]


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("ball_speed_mps", 0.0),
        ("launch_angle_deg", 95.0),
        ("spin_rate_rpm", -1.0),
        ("wind_speed_mps", 45.0),
        ("time_step_s", 0.0),
    ],
)
def test_simulate_ball_flight_rejects_invalid_ranges(
    client: TestClient, field: str, bad_value: float
) -> None:
    payload = _driver_payload(FlightModelType.WATERLOO_PENNER.value)
    payload[field] = bad_value

    response = client.post("/tools/ball-flight/simulate", json=payload)

    assert response.status_code == 422


def test_simulate_ball_flight_rejects_time_step_larger_than_max_time() -> None:
    with pytest.raises(ValidationError):
        BallFlightSimulationRequest(max_time_s=0.25, time_step_s=0.5)


def test_simulate_ball_flight_rejects_invalid_model_name(client: TestClient) -> None:
    payload = _driver_payload("not_a_model")

    response = client.post("/tools/ball-flight/simulate", json=payload)

    assert response.status_code == 422


# =============================================================================
# GET /models — shared flight-model enumeration (issue #7456)
# =============================================================================


def test_list_models_enumerates_full_registry(client: TestClient) -> None:
    response = client.get("/tools/ball-flight/models")

    assert response.status_code == 200
    models = response.json()["models"]
    keys = [m["key"] for m in models]
    assert keys == [mt.value for mt in FlightModelType], (
        "GET /models must enumerate the same registry the desktop tracer uses"
    )
    for model in models:
        assert model["name"]
        assert model["description"]
        assert model["reference"]


# =============================================================================
# Multi-model simulate (issue #7456)
# =============================================================================


def test_simulate_multi_model_returns_per_model_results(client: TestClient) -> None:
    payload = _driver_payload(FlightModelType.WATERLOO_PENNER.value)
    payload["models"] = [
        FlightModelType.WATERLOO_PENNER.value,
        FlightModelType.NATHAN.value,
    ]

    response = client.post("/tools/ball-flight/simulate", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert [r["model_key"] for r in data["results"]] == payload["models"]
    # Back-compat: top-level fields mirror the first requested model.
    assert data["model_key"] == FlightModelType.WATERLOO_PENNER.value
    assert data["trajectory"] == data["results"][0]["trajectory"]
    assert data["summary"] == data["results"][0]["summary"]


def test_simulate_single_model_response_is_backwards_compatible(
    client: TestClient,
) -> None:
    payload = _driver_payload(FlightModelType.NATHAN.value)

    response = client.post("/tools/ball-flight/simulate", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["model_key"] == FlightModelType.NATHAN.value
    assert len(data["results"]) == 1
    assert data["results"][0]["model_key"] == FlightModelType.NATHAN.value


def test_simulate_multi_model_deduplicates_preserving_order(
    client: TestClient,
) -> None:
    payload = _driver_payload(FlightModelType.WATERLOO_PENNER.value)
    payload["models"] = [
        FlightModelType.NATHAN.value,
        FlightModelType.NATHAN.value,
        FlightModelType.BALLANTYNE.value,
    ]

    response = client.post("/tools/ball-flight/simulate", json=payload)

    assert response.status_code == 200
    assert [r["model_key"] for r in response.json()["results"]] == [
        FlightModelType.NATHAN.value,
        FlightModelType.BALLANTYNE.value,
    ]


def test_simulate_rejects_empty_models_list(client: TestClient) -> None:
    payload = _driver_payload(FlightModelType.WATERLOO_PENNER.value)
    payload["models"] = []

    response = client.post("/tools/ball-flight/simulate", json=payload)

    assert response.status_code == 422


# =============================================================================
# Structural golden tests (issue #7456)
#
# The flight models have open physics-accuracy issues (#7403-#7405), so these
# assert structural correctness (monotonic time, finite values, returns to
# ground) rather than TrackMan golden numbers.
# =============================================================================


@pytest.mark.parametrize("model_type", list(FlightModelType))
def test_trajectory_is_structurally_sound(
    client: TestClient, model_type: FlightModelType
) -> None:
    payload = _driver_payload(model_type.value)
    payload["max_time_s"] = 10.0
    payload["time_step_s"] = 0.01

    response = client.post("/tools/ball-flight/simulate", json=payload)

    assert response.status_code == 200
    data = response.json()
    trajectory = data["trajectory"]
    assert len(trajectory) >= 2

    times = [sample["time_s"] for sample in trajectory]
    assert times == sorted(times), "time must be monotonically non-decreasing"
    assert all(b > a for a, b in zip(times, times[1:], strict=False)), (
        "time must be strictly increasing"
    )

    for sample in trajectory:
        values = [sample["time_s"], *sample["position_m"], *sample["velocity_mps"]]
        assert all(math.isfinite(v) for v in values), "all samples must be finite"

    heights = [sample["position_m"][2] for sample in trajectory]
    apex = max(heights)
    assert apex > 0.0, "ball must rise above launch height"
    assert heights[-1] <= 0.5, "ball must return to (near) ground level"

    summary = data["summary"]
    assert summary["carry_m"] > 0.0
    assert summary["apex_m"] > 0.0
    assert summary["flight_time_s"] > 0.0
    assert all(math.isfinite(v) for v in summary.values())


# =============================================================================
# POST /import — ADR-0047 H3 (issue #9352)
# =============================================================================


def _ud_family_record() -> dict[str, Any]:
    """A real, freshly simulated ``ud.flight_models`` export (H1, #9360)."""
    from src.shared.python.physics import flight_models as fm

    fm.FlightModelRegistry.reset()
    model = fm.FlightModelRegistry.get_model(FlightModelType.WATERLOO_PENNER)
    try:
        result = model.simulate(
            fm.UnifiedLaunchConditions(
                ball_speed=70.0, launch_angle=math.radians(12.0), spin_rate=2600.0
            ),
            max_time=1.0,
            dt=0.05,
        )
    finally:
        fm.FlightModelRegistry.reset()
    return flight_result_to_trajectory_record(
        result, model_type=FlightModelType.WATERLOO_PENNER
    )


def _tools_family_record() -> dict[str, Any]:
    """A hand-built, wire-valid ``swing_sim.flight`` record.

    Built directly against the documented contract (see
    ``src/shared/python/physics/flight_trajectory_import.py``) rather
    than by importing the vendored Tools package, so this test does not
    depend on ``swing_sim.flight``'s own physics being importable —
    only the *reader* needs to be, which is the module under test.
    """
    digest = trajectory_parameter_digest({"cd": 0.24, "cl": 0.21, "spin_decay": 0.08})
    return {
        "format": "swing_sim.ball_flight_trajectory/1",
        "source_id": "swing_sim.flight:Nathan",
        "frame_id": FLIGHT_FRAME_ID,
        "channels": ["velocity_mps"],
        "provenance": {
            "model_family": "swing_sim.flight",
            "model_name": "Nathan",
            "parameter_digest": digest,
        },
        "samples": [
            {
                "time_s": 0.0,
                "position_m": [0.0, 0.0, 0.0],
                "velocity_mps": [50.0, 0.0, 20.0],
            },
            {
                "time_s": 0.5,
                "position_m": [24.0, 0.3, 9.0],
                "velocity_mps": [47.0, 0.2, 6.0],
            },
            {
                "time_s": 1.0,
                "position_m": [45.0, 0.5, 0.0],
                "velocity_mps": [44.0, 0.1, -8.0],
            },
        ],
    }


@pytest.mark.parametrize(
    ("record_factory", "expected_family", "expected_name"),
    [
        (_ud_family_record, "ud.flight_models", "Waterloo/Penner"),
        (_tools_family_record, "swing_sim.flight", "Nathan"),
    ],
)
@requires_flight_interchange
def test_import_accepts_records_from_either_family(
    client: TestClient,
    record_factory: Any,
    expected_family: str,
    expected_name: str,
) -> None:
    response = client.post(
        "/tools/ball-flight/import", json={"record": record_factory()}
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["model_family"] == expected_family
    assert data["model_name"] == expected_name
    assert data["model_key"].startswith(f"{expected_family}:{expected_name}")
    assert len(data["parameter_digest"]) == 64
    assert len(data["trajectory"]) >= 2
    for sample in data["trajectory"]:
        values = [sample["time_s"], *sample["position_m"]]
        assert all(math.isfinite(v) for v in values)
    summary = data["summary"]
    assert all(math.isfinite(v) for v in summary.values())


@requires_flight_interchange
def test_import_rejects_unknown_top_level_field(client: TestClient) -> None:
    record = _tools_family_record()
    record["unexpected_field"] = "oops"

    response = client.post("/tools/ball-flight/import", json={"record": record})

    assert response.status_code == 400
    assert "unknown trajectory fields" in response.json()["detail"]
    assert "unexpected_field" in response.json()["detail"]


@requires_flight_interchange
def test_import_rejects_missing_provenance(client: TestClient) -> None:
    record = _tools_family_record()
    del record["provenance"]

    response = client.post("/tools/ball-flight/import", json={"record": record})

    assert response.status_code == 400
    assert "missing trajectory fields" in response.json()["detail"]
    assert "provenance" in response.json()["detail"]


@requires_flight_interchange
def test_import_rejects_malformed_provenance(client: TestClient) -> None:
    record = _tools_family_record()
    record["provenance"] = {"model_family": "swing_sim.flight"}  # missing keys

    response = client.post("/tools/ball-flight/import", json={"record": record})

    assert response.status_code == 400
    assert "provenance fields" in response.json()["detail"]


@requires_flight_interchange
def test_import_rejects_a_wire_invalid_frame(client: TestClient) -> None:
    """A ``frame_id`` outside the wire's own declared enum is the reader's refusal."""
    record = _tools_family_record()
    record["frame_id"] = "not_a_real_frame"

    response = client.post("/tools/ball-flight/import", json={"record": record})

    assert response.status_code == 400
    assert "frame_id must be one of" in response.json()["detail"]


@requires_flight_interchange
def test_import_rejects_a_wire_valid_but_unsupported_frame(client: TestClient) -> None:
    """``app_xtarget_yup_zright`` is valid on the wire but not yet plottable here."""
    record = _tools_family_record()
    record["frame_id"] = APP_FRAME_ID

    response = client.post("/tools/ball-flight/import", json={"record": record})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "unsupported frame" in detail
    assert APP_FRAME_ID in detail


@requires_flight_interchange
def test_import_rejects_non_monotone_samples(client: TestClient) -> None:
    record = _tools_family_record()
    record["samples"][2]["time_s"] = record["samples"][1]["time_s"]

    response = client.post("/tools/ball-flight/import", json={"record": record})

    assert response.status_code == 400
    assert "strictly increasing" in response.json()["detail"]


@requires_flight_interchange
def test_import_rejects_non_finite_sample_value(client: TestClient) -> None:
    """A literal ``NaN`` cannot come from ``JSON.stringify`` (it emits ``null``),
    but a non-browser HTTP client can still send one; the server must refuse
    it rather than accept it via Python's permissive ``json`` module. Sent as
    raw bytes because strict JSON — and therefore ``TestClient(json=...)`` —
    has no NaN literal to encode in the first place.
    """
    record = _tools_family_record()
    record["samples"][1]["position_m"][2] = "__NAN_PLACEHOLDER__"
    record_text = json.dumps(record).replace('"__NAN_PLACEHOLDER__"', "NaN")
    assert "NaN" in record_text
    raw_body = ('{"record": ' + record_text + "}").encode("utf-8")

    response = client.post(
        "/tools/ball-flight/import",
        content=raw_body,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400, response.text
    assert "finite" in response.json()["detail"]


@requires_flight_interchange
def test_import_rejects_ragged_channel_declaration(client: TestClient) -> None:
    """A sample missing a channel every other sample declares is refused."""
    record = _tools_family_record()
    del record["samples"][1]["velocity_mps"]

    response = client.post("/tools/ball-flight/import", json={"record": record})

    assert response.status_code == 400
    assert "sample fields must be exactly" in response.json()["detail"]


def test_import_rejects_non_object_record(client: TestClient) -> None:
    response = client.post(
        "/tools/ball-flight/import", json={"record": ["not", "an", "object"]}
    )

    assert response.status_code == 422


@requires_flight_interchange
def test_import_response_is_independent_across_requests(client: TestClient) -> None:
    """Two different families in a row must not bleed state between imports."""
    ud_response = client.post(
        "/tools/ball-flight/import", json={"record": _ud_family_record()}
    )
    tools_response = client.post(
        "/tools/ball-flight/import", json={"record": _tools_family_record()}
    )

    assert ud_response.status_code == 200
    assert tools_response.status_code == 200
    assert ud_response.json()["model_family"] != tools_response.json()["model_family"]
