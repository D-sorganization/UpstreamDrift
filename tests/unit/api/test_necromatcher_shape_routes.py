"""Shape proxies are explicit opt-ins at the SDK-free HTTP queue boundary."""

from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from src.api.routes.necromatcher import get_video_exports, router

pytestmark = pytest.mark.unit


@pytest.fixture
def shape_client() -> Any:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def submit(*args: Any, **kwargs: Any) -> dict[str, str]:
        calls.append((args, kwargs))
        return {"run_id": "owned"}

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_video_exports] = lambda: SimpleNamespace(submit=submit)
    with TestClient(app) as client:
        yield client, calls


def test_explicit_shape_option_is_typed_before_queue(shape_client: Any) -> None:
    client, calls = shape_client
    response = client.post(
        "/necromatcher/fits/fit/video-exports", json={"shape_overlay": {"opacity": 0.6}}
    )
    assert response.status_code == 202, response.text
    assert calls[0][0] == ("fit",)
    assert calls[0][1]["shape_overlay"].to_record() == {"opacity": 0.6}


def test_shape_combines_with_optional_reviewed_shaft(shape_client: Any) -> None:
    from tests.unit.motion_matching.test_shaft_observations import evidence

    client, calls = shape_client
    response = client.post(
        "/necromatcher/fits/fit/video-exports",
        json={
            "shaft_evidence": evidence().to_record(),
            "shape_overlay": {"opacity": 0.35},
        },
    )
    assert response.status_code == 202, response.text
    assert calls[0][0] == ("fit", evidence())
    assert calls[0][1]["shape_overlay"].opacity == 0.35


@pytest.mark.parametrize("value", [True, "0.3", -0.1, 1.1, None])
def test_invalid_opacity_never_reaches_queue(shape_client: Any, value: Any) -> None:
    client, calls = shape_client
    response = client.post(
        "/necromatcher/fits/fit/video-exports",
        json={"shape_overlay": {"opacity": value}},
    )
    assert response.status_code == 422
    assert calls == []


@pytest.mark.parametrize("body", [None, {}, {"shape_overlay": None}])
def test_disabled_shape_retains_bodyless_legacy_call(
    shape_client: Any, body: Any
) -> None:
    client, calls = shape_client
    assert (
        client.post("/necromatcher/fits/fit/video-exports", json=body).status_code
        == 202
    )
    assert calls == [(("fit",), {})]
