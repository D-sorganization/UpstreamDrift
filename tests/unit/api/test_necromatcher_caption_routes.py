"""Compact captions remain an explicit SDK-free HTTP display recipe."""

from typing import Any

import pytest

from tests.unit.api.test_necromatcher_shape_routes import shape_client as _shape_client

shape_client = _shape_client

pytestmark = pytest.mark.unit


def test_explicit_caption_is_typed_and_combines_with_shape(shape_client: Any) -> None:
    client, calls = shape_client
    response = client.post(
        "/necromatcher/fits/fit/video-exports",
        json={
            "caption_overlay": {"style": "compact_research_v1"},
            "shape_overlay": {"opacity": 0.35},
        },
    )
    assert response.status_code == 202, response.text
    assert calls[0][1]["caption_overlay"].to_record() == {
        "style": "compact_research_v1"
    }
    assert calls[0][1]["shape_overlay"].opacity == 0.35


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"style": True},
        {"style": "legacy"},
        {"style": "compact_research_v1", "opacity": 0.35},
    ],
)
def test_invalid_caption_never_reaches_queue(shape_client: Any, record: Any) -> None:
    client, calls = shape_client
    response = client.post(
        "/necromatcher/fits/fit/video-exports", json={"caption_overlay": record}
    )
    assert response.status_code == 422
    assert calls == []


@pytest.mark.parametrize("body", [None, {}, {"caption_overlay": None}])
def test_absent_caption_preserves_legacy_call(shape_client: Any, body: Any) -> None:
    client, calls = shape_client
    assert (
        client.post("/necromatcher/fits/fit/video-exports", json=body).status_code
        == 202
    )
    assert calls == [(("fit",), {})]
