"""OF-06 (#13085): the read-only open-fp LAN client, offline via MockTransport."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from src.shared.python.force_plates import PlateUncertainty
from src.shared.python.force_plates.open_fp_format import (
    HTTP_TIME_QUANTUM_S,
    OpenFpClient,
    OpenFpClientError,
    PlateDeclarations,
    UnknownSwingError,
    read_swing,
)
from tests.unit.force_plates import _open_fp_writer as w

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

BASE = "http://192.168.1.50:8000"
SWING = w.SWING_ID
SWING_PATH = f"/api/swings/{SWING}"


def declared() -> PlateDeclarations:
    return PlateDeclarations(
        uncertainty=PlateUncertainty(
            force_floor_n=2.0,
            force_fraction=0.01,
            cop_m=0.010,
            sensing_depth_sigma_m=0.002,
            source="declared hypothesis for tests",
        ),
        sensing_depth_m=0.015,
        capacity_n=3923.0,
    )


class Server:
    """A MockTransport handler recording every request it receives."""

    def __init__(self, routes: dict[str, Callable[[], httpx.Response]]) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        route = self.routes.get(request.url.path)
        return route() if route else httpx.Response(404, json={"detail": "x"})

    def client(self, *, max_bytes: int = 1 << 20) -> OpenFpClient:
        return OpenFpClient(
            BASE,
            timeout_s=2.0,
            max_bytes=max_bytes,
            transport=httpx.MockTransport(self),
        )


def body_of(payload: Any) -> bytes:
    return json.dumps(payload, allow_nan=False).encode()


def serving(payload: Any, path: str = SWING_PATH) -> Server:
    data = body_of(payload)
    return Server({path: lambda: httpx.Response(200, content=data)})


def fetch(payload: Any, *, max_bytes: int = 1 << 20):
    client = serving(payload).client(max_bytes=max_bytes)
    return client.fetch_swing(SWING, declared=declared())


def payload() -> dict[str, Any]:
    return w.http_swing_payload()


def server_for_body(body: bytes, path: str = SWING_PATH) -> Server:
    return Server({path: lambda: httpx.Response(200, content=body)})


# -------------------------------------------------------------- happy path --
def test_trial_matches_file_reader_within_declared_rounding(tmp_path: Path) -> None:
    http = fetch(payload())
    disk = read_swing(w.write_swing(tmp_path), declared=declared())
    np.testing.assert_allclose(http.times_s, disk.times_s, atol=5e-5)
    for foot in ("left", "right"):
        np.testing.assert_allclose(http.load_n[foot], disk.load_n[foot], atol=0.05)
    for key in ("left", "right", "net"):
        np.testing.assert_allclose(http.cop_m[key], disk.cop_m[key], atol=5e-5)
    assert http.handedness == disk.handedness
    assert http.body_weight_n == disk.body_weight_n
    for name, value in disk.events_s.items():
        other = http.events_s[name]
        assert (value is None) == (other is None)
        if value is not None:
            assert other == pytest.approx(value, abs=1e-3)


def test_provenance_records_format_and_body_digest() -> None:
    trial = fetch(payload())
    prov = trial.provenance
    digest = hashlib.sha256(body_of(payload())).hexdigest()
    assert prov.source_format == "open-fp-http-json"
    assert dict(prov.file_sha256) == {"http": digest}
    assert prov.qualification is None


def test_uncertainty_is_widened_by_declared_rounding() -> None:
    sigma = fetch(payload()).uncertainty
    assert sigma.cop_m == pytest.approx(math.hypot(0.010, 5e-5 / math.sqrt(3.0)))
    assert sigma.force_floor_n == pytest.approx(math.hypot(2.0, 0.05 / math.sqrt(3.0)))
    assert sigma.force_fraction == 0.01
    assert sigma.sensing_depth_sigma_m == 0.002
    assert "rounding" in sigma.source and "declared hypothesis" in sigma.source


def test_time_quantum_is_exported() -> None:
    assert HTTP_TIME_QUANTUM_S == 1e-4


def test_list_swings_returns_ids() -> None:
    other = "20260101-120500-007-2"
    summaries = [
        w.http_summary(w.http_swing_payload(swing_id=i)) for i in (SWING, other)
    ]
    server = Server({"/api/swings": lambda: httpx.Response(200, json=summaries)})
    assert server.client().list_swings() == (SWING, other)


def test_only_get_requests_to_the_two_endpoints() -> None:
    server = Server(
        {
            "/api/swings": lambda: httpx.Response(200, json=[]),
            SWING_PATH: lambda: httpx.Response(200, content=body_of(payload())),
        }
    )
    client = server.client()
    client.list_swings()
    client.fetch_swing(SWING, declared=declared())
    assert {r.method for r in server.requests} == {"GET"}
    assert [r.url.path for r in server.requests] == ["/api/swings", SWING_PATH]
    assert all(not r.url.query for r in server.requests)


# --------------------------------------------------------- constructor ----
@pytest.mark.parametrize(
    "url",
    [
        "ftp://host:8000",
        "host:8000",
        "http://",
        "http://host/api",
        "http://host/?x=1",
        "http://host/?",
        "http://host/#frag",
        "http://user:pw@host",
        "http://user@host:8000",
        "",
    ],
)
def test_bad_base_url_refused(url: str) -> None:
    with pytest.raises(ValueError):
        OpenFpClient(url, timeout_s=1.0, max_bytes=100)


@pytest.mark.parametrize(
    "url", ["http://host", "https://host:8443/", "http://10.0.0.2:80"]
)
def test_good_base_url_accepted(url: str) -> None:
    OpenFpClient(url, timeout_s=1.0, max_bytes=100).close()


@pytest.mark.parametrize("timeout", [0.0, -1.0, math.nan, math.inf])
def test_bad_timeout_refused(timeout: float) -> None:
    with pytest.raises(ValueError):
        OpenFpClient(BASE, timeout_s=timeout, max_bytes=100)


@pytest.mark.parametrize("limit", [0, -5, True, 1.5])
def test_bad_max_bytes_refused(limit: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        OpenFpClient(BASE, timeout_s=1.0, max_bytes=limit)


def test_non_string_url_and_bad_transport_refused() -> None:
    with pytest.raises(TypeError):
        OpenFpClient(5, timeout_s=1.0, max_bytes=100)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        OpenFpClient(BASE, timeout_s=1.0, max_bytes=100, transport="x")  # type: ignore[arg-type]


# ------------------------------------------------------------- swing id ----
@pytest.mark.parametrize(
    "bad", ["../x", "20260101-120000-123\n", "", "20260101-120000-123/..", "a?b=1", 7]
)
def test_invalid_id_refused_before_any_request(bad: Any) -> None:
    server = serving(payload())
    with pytest.raises((ValueError, TypeError)):
        server.client().fetch_swing(bad, declared=declared())
    assert server.requests == []


def test_declared_type_checked_before_any_request() -> None:
    server = serving(payload())
    with pytest.raises(TypeError):
        server.client().fetch_swing(SWING, declared="x")  # type: ignore[arg-type]
    assert server.requests == []


# ---------------------------------------------------------- transport -----
def test_oversize_body_with_content_length_refused() -> None:
    server = server_for_body(body_of(payload()))
    with pytest.raises(OpenFpClientError, match="exceeds"):
        server.client(max_bytes=100).fetch_swing(SWING, declared=declared())


def test_oversize_streamed_body_without_content_length_refused() -> None:
    data = body_of(payload())
    chunks = [data[i : i + 50] for i in range(0, len(data), 50)]
    server = Server({SWING_PATH: lambda: httpx.Response(200, content=iter(chunks))})
    with pytest.raises(OpenFpClientError, match="exceeds"):
        server.client(max_bytes=200).fetch_swing(SWING, declared=declared())


def test_lying_content_length_refused() -> None:
    data = body_of(payload())
    headers = {"Content-Length": "10"}
    server = Server(
        {SWING_PATH: lambda: httpx.Response(200, content=data, headers=headers)}
    )
    with pytest.raises(OpenFpClientError):
        server.client(max_bytes=200).fetch_swing(SWING, declared=declared())


def test_non_numeric_content_length_refused() -> None:
    headers = {"Content-Length": "abc"}
    server = Server(
        {SWING_PATH: lambda: httpx.Response(200, content=b"{}", headers=headers)}
    )
    with pytest.raises(OpenFpClientError, match="Content-Length"):
        server.client().fetch_swing(SWING, declared=declared())


def raising_client(error: type[httpx.TransportError]) -> OpenFpClient:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error("boom", request=request)

    return OpenFpClient(
        BASE, timeout_s=1.0, max_bytes=100, transport=httpx.MockTransport(handler)
    )


def test_timeout_is_wrapped() -> None:
    with pytest.raises(OpenFpClientError, match="timed out"):
        raising_client(httpx.ReadTimeout).list_swings()


def test_connection_error_is_wrapped() -> None:
    with pytest.raises(OpenFpClientError):
        raising_client(httpx.ConnectError).list_swings()


@pytest.mark.parametrize("status", [201, 301, 500, 503])
def test_non_200_refused(status: int) -> None:
    server = Server({SWING_PATH: lambda: httpx.Response(status, content=b"{}")})
    with pytest.raises(OpenFpClientError, match=str(status)):
        server.client().fetch_swing(SWING, declared=declared())


def test_404_is_unknown_swing() -> None:
    with pytest.raises(UnknownSwingError) as info:
        Server({}).client().fetch_swing(SWING, declared=declared())
    assert isinstance(info.value, LookupError)
    assert SWING in str(info.value)


def test_404_on_list_is_a_plain_client_error() -> None:
    with pytest.raises(OpenFpClientError) as info:
        Server({}).client().list_swings()
    assert not isinstance(info.value, LookupError)


# ------------------------------------------------------ content refusals ---
def test_duplicate_json_members_refused() -> None:
    text = (
        body_of(payload())
        .decode()
        .replace('"golfer": "RH"', '"golfer": "RH", "golfer": "LH"')
    )
    with pytest.raises(ValueError, match="duplicate"):
        server_for_body(text.encode()).client().fetch_swing(SWING, declared=declared())


def test_duplicate_members_in_list_refused() -> None:
    server = server_for_body(b'[{"id": "a", "id": "b"}]', "/api/swings")
    with pytest.raises(ValueError, match="duplicate"):
        server.client().list_swings()


def test_unknown_and_missing_top_level_keys_refused() -> None:
    extra = payload()
    extra["surprise"] = 1
    with pytest.raises(ValueError, match="unexpected keys"):
        fetch(extra)
    missing = payload()
    del missing["arrays"]
    with pytest.raises(ValueError, match="missing keys"):
        fetch(missing)


def test_unknown_array_key_refused() -> None:
    bad = payload()
    bad["arrays"]["extra"] = [0.0]
    with pytest.raises(ValueError, match="unexpected keys"):
        fetch(bad)


def test_id_mismatch_refused() -> None:
    bad = payload()
    bad["id"] = "20260101-120000-124"
    with pytest.raises(ValueError, match="'id'"):
        fetch(bad)


def test_length_mismatch_refused() -> None:
    bad = payload()
    bad["arrays"]["fz_L"] = bad["arrays"]["fz_L"][:-1]
    with pytest.raises(ValueError, match="fz_L"):
        fetch(bad)


def test_ragged_cop_refused() -> None:
    bad = payload()
    bad["arrays"]["cop_L"][0] = [1.0]
    with pytest.raises(ValueError, match="cop_L"):
        fetch(bad)


@pytest.mark.parametrize("key", ["t", "fz_L", "fz_R", "lead_pct"])
def test_null_in_non_cop_array_refused(key: str) -> None:
    bad = payload()
    bad["arrays"][key][0] = None
    with pytest.raises(ValueError, match=key):
        fetch(bad)


def test_null_in_forces_refused() -> None:
    bad = payload()
    bad["arrays"]["forces"][1][3] = None
    with pytest.raises(ValueError, match="forces"):
        fetch(bad)


def test_null_in_cop_is_nan() -> None:
    trial = fetch(payload())
    assert np.isnan(trial.cop_m["left"][-1]).all()


def test_non_number_element_refused() -> None:
    for bad_value in ("400", True):
        bad = payload()
        bad["arrays"]["fz_R"][0] = bad_value
        with pytest.raises(ValueError, match="fz_R"):
            fetch(bad)


def test_nan_literal_refused() -> None:
    text = body_of(payload()).decode().replace('"bw_n": 800.0', '"bw_n": NaN')
    with pytest.raises(ValueError, match="non-finite"):
        server_for_body(text.encode()).client().fetch_swing(SWING, declared=declared())


def test_lead_pct_inconsistent_with_handedness_refused() -> None:
    bad = w.http_swing_payload("LH")
    bad["golfer"] = "RH"
    with pytest.raises(ValueError, match="lead_pct"):
        fetch(bad)


def test_lead_pct_within_json_rounding_accepted() -> None:
    ok = payload()
    ok["arrays"]["lead_pct"][0] += 0.004
    fetch(ok)


def test_event_outside_trial_refused() -> None:
    bad = payload()
    bad["events"]["finish"] = 9.0
    with pytest.raises(ValueError, match="events"):
        fetch(bad)


def test_non_object_detail_body_refused() -> None:
    with pytest.raises(ValueError, match="object"):
        server_for_body(b"[]").client().fetch_swing(SWING, declared=declared())


@pytest.mark.parametrize("body", [b"{}", b"[1]", b'[{"id": "../x"}]'])
def test_malformed_list_bodies_refused(body: bytes) -> None:
    with pytest.raises(ValueError):
        server_for_body(body, "/api/swings").client().list_swings()


def test_list_with_bad_summary_keys_or_duplicate_ids_refused() -> None:
    entry = w.http_summary(payload())
    odd = copy.deepcopy(entry)
    odd["bonus"] = 1
    for listing in ([odd], [entry, entry]):
        server = server_for_body(body_of(listing), "/api/swings")
        with pytest.raises(ValueError):
            server.client().list_swings()


def test_context_manager_closes() -> None:
    with serving(payload()).client() as client:
        client.fetch_swing(SWING, declared=declared())


def test_content_length_shorter_than_body_refused_even_under_the_cap() -> None:
    data = body_of(payload())
    headers = {"Content-Length": "10"}
    server = Server(
        {SWING_PATH: lambda: httpx.Response(200, content=data, headers=headers)}
    )
    with pytest.raises(OpenFpClientError, match="Content-Length"):
        server.client().fetch_swing(SWING, declared=declared())


def test_client_source_issues_no_method_but_get() -> None:
    root = Path(__file__).resolve().parents[3]
    module = root / "src/shared/python/force_plates/open_fp_format/client.py"
    text = module.read_text(encoding="utf-8")
    for verb in ("post", "put", "delete", "patch", "request", "send"):
        assert f".{verb}(" not in text
    assert text.count('.stream("GET"') == 1
