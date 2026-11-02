"""Putting analytics API routes (#14177).

The routes pass request bodies straight to the shared putting service, so
these tests check the wiring: same answer as the service, service errors as
400s, the work cap holds, and route import stays free of the roll model.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from typing import Any

import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api.rate_limit import limiter  # noqa: E402
from src.api.routes.putting_analytics import router  # noqa: E402
from src.shared.python.putting_analytics import (  # noqa: E402
    service,
    service_curve,
    service_variance,
)

PREFIX = "/tools/putting-analytics"

# Test-only miss pricing, shaped like a putting curve; not reference data.
BASELINE = {
    "distances_ft": [1.0, 3.0, 6.0, 10.0, 20.0, 40.0],
    "expected_putts": [1.0, 1.1, 1.4, 1.6, 1.85, 2.1],
}


def _study(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "distance_ft": 8.0,
        "baseline": BASELINE,
        "aims_deg": [0.0, 3.0],
        "paces_ft": [0.5, 2.0],
        "line_sd_deg": 0.5,
        "speed_sd": 0.03,
        "n": 12,
        "seed": 4,
    }
    return {**args, **overrides}


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(router)
    limiter.reset()
    yield TestClient(app)
    limiter.reset()


def test_router_prefix_leaves_api_segment_to_the_server() -> None:
    assert not router.prefix.startswith("/api")


def test_optimize_matches_the_shared_service(client: TestClient) -> None:
    response = client.post(f"{PREFIX}/optimize", json=_study())
    assert response.status_code == 200, response.text
    assert response.json() == service.optimize_putt(_study())


def test_optimize_must_make_ranks_by_make_rate(client: TestClient) -> None:
    response = client.post(f"{PREFIX}/optimize", json=_study(objective="must_make"))
    assert response.status_code == 200, response.text
    rates = [entry["make_rate"] for entry in response.json()["top"]]
    assert rates == sorted(rates, reverse=True)


def test_misread_cost_matches_the_shared_service(client: TestClient) -> None:
    body = _study(grade_pct=1.0, aspect_deg=270.0, reads=[{"slope_scale": 0.5}])
    response = client.post(f"{PREFIX}/misread-cost", json=body)
    assert response.status_code == 200, response.text
    assert response.json() == service.misread_cost(body)


def test_library_study_matches_the_shared_service(client: TestClient) -> None:
    body = _study(
        classes=[
            {"distance_ft": 4.0, "count": 3},
            {"distance_ft": 12.0, "slope_pct": 1.0, "orientation": "uphill"},
        ],
        n=8,
    )
    response = client.post(f"{PREFIX}/library-study", json=body)
    assert response.status_code == 200, response.text
    assert response.json() == service.library_study(body)


VARIANCE_BODY: dict[str, Any] = {
    "baseline": BASELINE,
    "classes": [{"distance_ft": 6.0, "count": 2.0}],
    "factors": ["face", "speed"],
    "base_samples": 8,
    "n_samples": 32,
    "n_bootstrap": 10,
    "seed": 2,
}
CURVE_BODY: dict[str, Any] = {"distances_ft": [3.0, 10.0], "n": 60, "seed": 2}


def test_variance_study_matches_the_shared_service(client: TestClient) -> None:
    response = client.post(f"{PREFIX}/variance-study", json=VARIANCE_BODY)
    assert response.status_code == 200, response.text
    assert response.json() == service_variance.variance_study(VARIANCE_BODY)


def test_simulated_curve_matches_the_shared_service(client: TestClient) -> None:
    response = client.post(f"{PREFIX}/simulated-curve", json=CURVE_BODY)
    assert response.status_code == 200, response.text
    assert response.json() == service_curve.curve_study(CURVE_BODY)


@pytest.mark.parametrize(
    ("path", "body", "fragment"),
    [
        ("optimize", _study(distance_ft=-1.0), "distance_ft"),
        ("optimize", _study(objective="lag"), "objective"),
        ("optimize", _study(n=service.MAX_ROLLOUTS), "candidates x n"),
        ("misread-cost", _study(), "reads"),
        ("library-study", _study(), "classes"),
        ("variance-study", {**VARIANCE_BODY, "factors": ["strike"]}, "factors"),
        ("simulated-curve", {**CURVE_BODY, "n": 1}, "'n'"),
    ],
)
def test_service_rejections_are_400s(
    client: TestClient, path: str, body: dict[str, Any], fragment: str
) -> None:
    response = client.post(f"{PREFIX}/{path}", json=body)
    assert response.status_code == 400, response.text
    assert fragment in response.text


def test_a_non_object_body_is_rejected(client: TestClient) -> None:
    response = client.post(f"{PREFIX}/optimize", json=[1, 2, 3])
    assert response.status_code == 422


def test_limits_report_the_service_caps(client: TestClient) -> None:
    response = client.get(f"{PREFIX}/limits")
    assert response.status_code == 200, response.text
    limits = response.json()
    assert limits["max_rollouts"] == service.MAX_ROLLOUTS
    assert limits["stimp_range"] == list(service.STIMP_RANGE)
    assert "flat" in limits["orientations"]
    assert set(limits["objectives"]) == {"sg", "must_make"}
    assert limits["variance_factors"] == list(service_variance.MODELLED_FACTORS)
    assert limits["max_variance_rollouts"] == service_variance.MAX_VARIANCE_ROLLOUTS
    assert limits["max_curve_rollouts"] == service_curve.MAX_CURVE_ROLLOUTS
    assert limits["default_curve_distances_ft"] == list(
        service_curve.DEFAULT_CURVE_GRID_FT
    )


def test_route_import_does_not_load_the_roll_model() -> None:
    """API start-up imports every route module; keep this one light."""
    names = (
        "src.api.routes.putting_analytics",
        "src.shared.python.putting_analytics.service",
        "src.shared.python.putting_analytics.roll_model",
        "src.shared.python.putting_analytics.service_variance",
        "src.shared.python.putting_analytics.service_curve",
    )
    saved = {name: sys.modules.pop(name, None) for name in names}
    try:
        importlib.import_module(names[0])
        assert not [name for name in names[1:] if name in sys.modules]
    finally:
        for name, module in saved.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module
