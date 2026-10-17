"""Read-only client of the open-fp LAN API (OF-06, epic #13061).

Only ``GET /api/swings`` and ``GET /api/swings/{id}`` are ever requested: the
single transport call is ``stream("GET", ...)`` and no other method, endpoint
(DELETE/PUT/POST) or query string is reachable.  Responses are streamed under a
byte cap, redirects are not followed, and the JSON then goes through the same
validation and trial builder as the file reader (:mod:`.reader`, :mod:`._meta`,
:mod:`._arrays`).

The JSON form is rounded by the server (see ``HTTP_*`` in
:mod:`~src.shared.python.force_plates.open_fp_format.format`).  That rounding is
declared, not hidden: the trial's :class:`PlateUncertainty` is widened by the
uniform-quantisation sigma, and :data:`HTTP_TIME_QUANTUM_S` is exported for
callers that reason about clock skew (OF-05).

open-fp is AGPL-3.0; this code is written from the documented interface only.

Errors: ``ValueError`` / ``TypeError`` for bad arguments and for responses that
fail validation (as the file readers); :class:`OpenFpClientError` for transport
failures, timeouts, oversize bodies and non-200 statuses;
:class:`UnknownSwingError` (a ``LookupError``) for a 404 on a swing.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any
from urllib.parse import SplitResult, urlsplit

import httpx

from src.shared.python.force_plates.contracts import (
    PlateUncertainty,
    VerticalPlateTrial,
    finite_positive,
)
from src.shared.python.force_plates.open_fp_format import _files, _json_arrays, _meta
from src.shared.python.force_plates.open_fp_format import format as fmt
from src.shared.python.force_plates.open_fp_format._arrays import (
    HTTP_LEAD_CHECK,
    check_lead_pct,
)
from src.shared.python.force_plates.open_fp_format.reader import (
    PlateDeclarations,
    _build_trial,
    _events_on_time_axis,
    _Header,
    _require_declared,
    _Source,
)

__all__ = [
    "HTTP_TIME_QUANTUM_S",
    "OpenFpClient",
    "OpenFpClientError",
    "UnknownSwingError",
    "widen_for_http_rounding",
]

logger = logging.getLogger(__name__)

#: Time resolution [s] of the JSON form; re-exported for OF-05 skew budgets.
HTTP_TIME_QUANTUM_S = fmt.HTTP_TIME_QUANTUM_S

_LIST_PATH = "/api/swings"
_ALLOWED_SCHEMES = ("http", "https")
_MM_PER_M = fmt.MM_PER_M
_CHUNK_BYTES = 65536
_STATUS_OK = 200
_STATUS_NOT_FOUND = 404
#: Std. deviation of a uniform error of half-width ``h`` is ``h / sqrt(3)``.
_SQRT3 = math.sqrt(3.0)
_COP_HALF_WIDTH_M = 0.5 * fmt.HTTP_COP_QUANTUM_MM / _MM_PER_M
_FORCE_HALF_WIDTH_N = 0.5 * fmt.HTTP_FORCE_QUANTUM_N


class OpenFpClientError(RuntimeError):
    """A transport failure, timeout, oversize body or non-200 response."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class UnknownSwingError(LookupError):
    """The server has no swing with the requested id (HTTP 404)."""


def widen_for_http_rounding(uncertainty: PlateUncertainty) -> PlateUncertainty:
    """Add the JSON form's declared rounding to ``uncertainty`` in quadrature.

    CoP: ``hypot(cop_m, 5e-5 m / sqrt(3))`` and force floor:
    ``hypot(force_floor_n, 0.05 N / sqrt(3))`` (uniform quantisation error of
    half-width half a quantum).  Other fields are kept; ``source`` is amended.
    """
    if not isinstance(uncertainty, PlateUncertainty):
        raise TypeError("uncertainty must be a PlateUncertainty")
    return dataclasses.replace(
        uncertainty,
        cop_m=math.hypot(uncertainty.cop_m, _COP_HALF_WIDTH_M / _SQRT3),
        force_floor_n=math.hypot(
            uncertainty.force_floor_n, _FORCE_HALF_WIDTH_N / _SQRT3
        ),
        source=f"{uncertainty.source} + open-fp JSON rounding (0.1 N, 0.1 mm)",
    )


def _validated_base(base_url: object) -> str:
    """``scheme://host[:port]`` of a bare http(s) origin; ``ValueError`` otherwise."""
    if not isinstance(base_url, str):
        raise TypeError(f"base_url must be a str, got {type(base_url).__name__}")
    parts: SplitResult = urlsplit(base_url)
    problems = (
        parts.scheme not in _ALLOWED_SCHEMES,
        not parts.hostname,
        "@" in parts.netloc,
        parts.path not in ("", "/"),
        parts.query or "?" in base_url,
        parts.fragment or "#" in base_url,
    )
    if any(problems):
        raise ValueError(
            "base_url must be a bare http(s)://host[:port] origin without "
            f"credentials, path, query or fragment, got {base_url!r}"
        )
    parts.port  # noqa: B018 - raises ValueError for an invalid port
    return f"{parts.scheme}://{parts.netloc}"


def _validated_limit(max_bytes: object) -> int:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise TypeError(f"max_bytes must be an int, got {max_bytes!r}")
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    return max_bytes


def _declared_length(response: httpx.Response, path: str) -> int | None:
    raw = response.headers.get("Content-Length")
    if raw is None:
        return None
    if not raw.isascii() or not raw.isdigit():
        raise OpenFpClientError(f"GET {path}: malformed Content-Length {raw[:20]!r}")
    return int(raw)


def _read_capped(response: httpx.Response, path: str, max_bytes: int) -> bytes:
    """Body of ``response``, refusing more than ``max_bytes`` or a lying length."""
    declared = _declared_length(response, path)
    if declared is not None and declared > max_bytes:
        raise OpenFpClientError(
            f"GET {path}: Content-Length {declared} exceeds the {max_bytes}-byte limit"
        )
    body = bytearray()
    for chunk in response.iter_bytes(_CHUNK_BYTES):
        body += chunk
        if len(body) > max_bytes:
            raise OpenFpClientError(
                f"GET {path}: body exceeds the {max_bytes}-byte limit"
            )
    encoded = "Content-Encoding" in response.headers
    if declared is not None and not encoded and len(body) != declared:
        raise OpenFpClientError(
            f"GET {path}: received {len(body)} bytes but Content-Length said {declared}"
        )
    return bytes(body)


def _check_swing_id(swing_id: object) -> str:
    """The id, if it is exactly a swing id (safe to put in a URL path)."""
    if not isinstance(swing_id, str):
        raise TypeError(f"swing_id must be a str, got {type(swing_id).__name__}")
    if not fmt.ID_PATTERN.fullmatch(swing_id):
        raise ValueError(f"swing_id {swing_id[:40]!r} is not a swing id")
    return swing_id


def _parse_listing(body: bytes) -> tuple[str, ...]:
    where = f"open-fp {_LIST_PATH}"
    entries = _meta.load_json(body, where)
    if not isinstance(entries, list):
        raise ValueError(f"{where}: top-level JSON value must be a list")
    ids: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: entry {index} must be an object")
        _meta.check_keys(entry, fmt.HTTP_SUMMARY_KEYS, f"{where}[{index}]")
        ids.append(_check_swing_id(entry["id"]))
    if len(set(ids)) != len(ids):
        raise ValueError(f"{where}: duplicate swing ids")
    return tuple(ids)


def _parse_swing(
    body: bytes, swing_id: str, declared: PlateDeclarations
) -> VerticalPlateTrial:
    where = f"open-fp {_LIST_PATH}/{swing_id}"
    obj = _meta.load_json(body, where)
    if not isinstance(obj, dict):
        raise ValueError(f"{where}: top-level JSON value must be an object")
    _meta.check_keys(obj, fmt.HTTP_META_KEYS, where)
    meta = _meta.parse_meta_object(
        {k: v for k, v in obj.items() if k != "arrays"}, where, swing_id
    )
    arrays = _json_arrays.reconcile_rounding(
        _json_arrays.arrays_from_json(obj["arrays"], where), where
    )
    check_lead_pct(arrays, meta.handedness, HTTP_LEAD_CHECK, where)
    events = _events_on_time_axis(
        meta.events_rel_s, arrays.t, where, fmt.HTTP_EVENT_EDGE_TOL_S
    )
    header = _Header(meta.handedness, meta.body_weight_n, events)
    widened = dataclasses.replace(
        declared, uncertainty=widen_for_http_rounding(declared.uncertainty)
    )
    source = _Source(fmt.HTTP_SOURCE_FORMAT, {"http": _files.sha256_hex(body)})
    return _build_trial(arrays, header, widened, source)


class OpenFpClient:
    """Read-only client of one open-fp server on the LAN.

    Preconditions: ``base_url`` a bare ``http(s)://host[:port]`` origin;
    ``timeout_s`` finite and positive (applies to connect, read and write);
    ``max_bytes`` a positive int, the cap on every response body;
    ``transport`` an optional ``httpx.BaseTransport`` (tests inject a mock).
    Redirects are never followed and environment proxies are ignored.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float,
        max_bytes: int,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base = _validated_base(base_url)
        timeout = finite_positive(timeout_s, "timeout_s")
        self._max_bytes = _validated_limit(max_bytes)
        if transport is not None and not isinstance(transport, httpx.BaseTransport):
            raise TypeError("transport must be an httpx.BaseTransport or None")
        self._http = httpx.Client(
            transport=transport,
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            headers={"Accept": "application/json"},
        )

    def close(self) -> None:
        """Release the underlying connection pool (idempotent)."""
        self._http.close()

    def __enter__(self) -> OpenFpClient:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def list_swings(self) -> tuple[str, ...]:
        """Ids of the swings on the server, in the server's order.

        ``GET /api/swings``.  Each entry must have exactly the summary keys and
        a valid swing id; repeated ids or JSON members are refused.
        """
        return _parse_listing(self._get(_LIST_PATH))

    def fetch_swing(
        self, swing_id: str, *, declared: PlateDeclarations
    ) -> VerticalPlateTrial:
        """Fetch and validate one swing as a :class:`VerticalPlateTrial`.

        ``GET /api/swings/{swing_id}``; ``swing_id`` and ``declared`` are
        validated before any request.  Raises :class:`UnknownSwingError` on 404.
        The trial's uncertainty is ``declared`` widened for the server's
        rounding (:func:`widen_for_http_rounding`).
        """
        valid_id = _check_swing_id(swing_id)
        decl = _require_declared(declared)
        try:
            body = self._get(f"{_LIST_PATH}/{valid_id}")
        except OpenFpClientError as exc:
            if exc.status_code == _STATUS_NOT_FOUND:
                raise UnknownSwingError(
                    f"the server has no swing {valid_id!r}"
                ) from exc
            raise
        return _parse_swing(body, valid_id, decl)

    def _get(self, path: str) -> bytes:
        """The only transport call: a streamed, size-capped GET of ``path``."""
        url = self._base + path
        logger.debug("GET %s", url)
        try:
            with self._http.stream("GET", url) as response:
                if response.status_code != _STATUS_OK:
                    raise OpenFpClientError(
                        f"GET {path}: HTTP status {response.status_code}",
                        status_code=response.status_code,
                    )
                return _read_capped(response, path, self._max_bytes)
        except httpx.TimeoutException as exc:
            raise OpenFpClientError(f"GET {path}: timed out ({exc})") from exc
        except httpx.HTTPError as exc:
            raise OpenFpClientError(f"GET {path}: request failed ({exc})") from exc
