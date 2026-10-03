"""Optional shaft records cross HTTP through the canonical typed job boundary."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from src.api.routes.necromatcher import get_refits, get_video_exports, router
from tests.unit.motion_matching.test_shaft_observations import evidence

pytestmark = pytest.mark.unit


@pytest.fixture
def shaft_client() -> Any:
    calls: list[tuple[Any, ...]] = []

    def submit(*args: Any) -> dict[str, str]:
        calls.append(args)
        return {"run_id": "owned-run"}

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_refits] = lambda: SimpleNamespace(submit=submit)
    with TestClient(app) as client:
        yield client, calls


def payload() -> dict[str, Any]:
    return {
        "new_fit_id": "shaft-variant",
        "frame_indices": [0, 1],
        "knot_count": 2,
        "coordinate_scales": [1.0],
    }


def test_optional_record_becomes_typed_evidence_before_queue(shaft_client: Any) -> None:
    client, calls = shaft_client
    request = payload() | {"shaft_evidence": evidence().to_record()}
    response = client.post("/necromatcher/fits/parent/refits", json=request)
    assert response.status_code == 202, response.text
    assert len(calls) == 1 and len(calls[0]) == 4
    assert calls[0][3] == evidence()


@pytest.mark.parametrize("included", [False, True])
def test_disabled_term_retains_legacy_submit_shape(
    shaft_client: Any, included: bool
) -> None:
    client, calls = shaft_client
    request = payload()
    if included:
        request["shaft_evidence"] = None
    assert (
        client.post("/necromatcher/fits/parent/refits", json=request).status_code == 202
    )
    assert len(calls[0]) == 3


@pytest.mark.parametrize("fault", ["schema", "boolean_index", "nan", "unknown_field"])
def test_malformed_record_never_reaches_queue(shaft_client: Any, fault: str) -> None:
    client, calls = shaft_client
    record = deepcopy(evidence().to_record())
    if fault == "schema":
        record["schema"] = "untrusted"
    elif fault == "boolean_index":
        record["frames"][0]["frame_index"] = True
    elif fault == "nan":
        record["frames"][0]["segment"]["sigma_px"] = "NaN"
    else:
        record["host_path"] = "forbidden"
    response = client.post(
        "/necromatcher/fits/parent/refits",
        json=payload() | {"shaft_evidence": record},
    )
    assert response.status_code == 422
    assert calls == []


def test_canonical_source_rejection_is_http_error_without_fallback(
    shaft_client: Any,
) -> None:
    client, calls = shaft_client

    def reject(*args: Any) -> dict[str, str]:
        calls.append(args)
        raise ValueError("Shaft original PNG hash mismatch")

    client.app.dependency_overrides[get_refits] = lambda: SimpleNamespace(submit=reject)
    response = client.post(
        "/necromatcher/fits/parent/refits",
        json=payload() | {"shaft_evidence": evidence().to_record()},
    )
    assert response.status_code == 422
    assert len(calls) == 1 and len(calls[0]) == 4


@pytest.mark.parametrize("enabled", [False, True])
def test_video_record_is_optional_and_typed_before_queue(
    shaft_client: Any, enabled: bool
) -> None:
    client, calls = shaft_client
    client.app.dependency_overrides[get_video_exports] = (
        client.app.dependency_overrides[get_refits]
    )
    request = {"shaft_evidence": evidence().to_record()} if enabled else None
    response = client.post("/necromatcher/fits/parent/video-exports", json=request)
    assert response.status_code == 202, response.text
    assert len(calls[0]) == (2 if enabled else 1)
    if enabled:
        assert calls[0][1] == evidence()
