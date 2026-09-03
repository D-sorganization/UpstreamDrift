"""App-local expected-strokes baseline artifact, kept out of the canonical port.

ADR-0046 Stage 2 wave 3b retired ``strokes_gained_types`` onto the canonical
launch-monitor layer. **This file is the half that deliberately did not
travel**, and it stays here rather than disappearing because UpstreamDrift
consumers need a concrete, validating model where the canonical layer needs
only a structural view.

Why the split exists (ADR-0048, step P12)
-----------------------------------------
P12's row in the port order reads ``strokes_gained_types.py`` *(minus baseline
half)*, and the plan gives the reason: the expected-strokes baseline half is
"genuinely already home" in Tools as
``rate_of_closure.launch_monitor_strokes_gained_baseline``, whose loader
additionally carries a ``MAX_BASELINE_BYTES`` cap and source-URL validation.
G0's ``test_baseline_table_digest_agrees_across_stacks`` pins both stacks'
digests of the same table at
``188a6eafa9eebd8a0f4c9ba288d858ad359e35999ba2706989c75d349f509925``. Porting
these definitions into the canonical package would have installed a second
authority for the same artifact; the canonical package instead types its
``baseline`` argument against the runtime-checkable protocols
``ExpectedStrokesStateLike`` and ``ExpectedStrokesBaselineLike``, so any
object with the right attributes flows straight in and the canonical layer
never has to import ``rate_of_closure``.

Why UpstreamDrift keeps a concrete model anyway
-----------------------------------------------
A protocol validates nothing at a trust boundary. ``src/api/routes/
launch_monitor_analytics.py`` declares ``baseline: ExpectedStrokesBaselineV2``
on a request payload, so this model is what parses untrusted JSON off the
wire: it normalises the course-state dimensions, rejects a non-HTTP(S)
``source_url``, refuses duplicate course states, and **verifies the declared
``table_sha256`` against the states actually supplied**. Replacing it with the
canonical protocol would turn a hash-verified benchmark into an unchecked
`dict`. UpstreamDrift also cannot substitute the already-home Tools loader,
because ``rate_of_closure`` is a measurement dependency of the drift gates,
never a runtime dependency of this application.

The models below satisfy ``ExpectedStrokesBaselineLike`` and
``ExpectedStrokesStateLike`` structurally, which
``tests/unit/launch_monitor/test_canonical_layer_parity.py`` asserts directly
with ``isinstance`` against the runtime-checkable protocols rather than by
reading field lists. That is the whole contract between this file and the
canonical layer.

Nothing here is a fork: these definitions are byte-for-byte the ones
``strokes_gained_types.py`` carried before the retirement, moved rather than
rewritten, and ``BASELINE_CONTRACT_VERSION`` is re-exported from the canonical
module rather than redeclared so the two can never disagree about the string.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from hashlib import sha256
from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shared.python.launch_monitor.strokes_gained_types import (
    BASELINE_CONTRACT_VERSION,
)


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExpectedStrokesStateV2(_ContractModel):
    """One benchmark point for an explicit target-aware course state."""

    lie: str = Field(min_length=1)
    context: str = Field(min_length=1)
    target: str = Field(min_length=1)
    distance_yards: float = Field(ge=0.0)
    expected_strokes: float = Field(ge=0.0)
    standard_error: float | None = Field(default=None, ge=0.0)

    @field_validator("lie", "context", "target")
    @classmethod
    def normalize_dimension(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("course-state dimensions must be non-empty")
        return normalized

    @model_validator(mode="after")
    def require_finite_values(self) -> ExpectedStrokesStateV2:
        values = [self.distance_yards, self.expected_strokes]
        if self.standard_error is not None:
            values.append(self.standard_error)
        if not all(isfinite(value) for value in values):
            raise ValueError("benchmark values must be finite")
        return self


class ExpectedStrokesBaselineV2(_ContractModel):
    """Hash-verified expected-strokes benchmark and publication metadata."""

    contract_version: Literal["launch-monitor-strokes-gained-baseline/2.0.0"] = (
        BASELINE_CONTRACT_VERSION
    )
    baseline_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    license: str = Field(min_length=1)
    table_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    states: tuple[ExpectedStrokesStateV2, ...] = Field(min_length=2)

    @field_validator("baseline_id", "version", "license")
    @classmethod
    def strip_metadata(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("baseline metadata must be non-empty")
        return normalized

    @field_validator("source_url")
    @classmethod
    def require_http_source(cls, value: str) -> str:
        from urllib.parse import urlparse

        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("source_url must be HTTP(S) with a host")
        return value

    @model_validator(mode="after")
    def verify_table(self) -> ExpectedStrokesBaselineV2:
        identities = [
            (state.lie, state.context, state.target, state.distance_yards)
            for state in self.states
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("baseline contains duplicate course states")
        if baseline_table_sha256(self.states) != self.table_sha256:
            raise ValueError("baseline table_sha256 does not match canonical states")
        return self


def _canonical_number(value: float | int) -> str:
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError("baseline numbers must be finite")
    normalized = f"{numeric:.12f}".rstrip("0").rstrip(".")
    return "0" if normalized in {"", "-0"} else normalized


def _coerce_state(value: ExpectedStrokesStateV2 | Mapping[str, Any]) -> dict[str, Any]:
    state = (
        value
        if isinstance(value, ExpectedStrokesStateV2)
        else ExpectedStrokesStateV2.model_validate(value)
    )
    return {
        "context": state.context,
        "distance_yards": _canonical_number(state.distance_yards),
        "expected_strokes": _canonical_number(state.expected_strokes),
        "lie": state.lie,
        "standard_error": (
            None
            if state.standard_error is None
            else _canonical_number(state.standard_error)
        ),
        "target": state.target,
    }


def _state_sort_key(state: Mapping[str, Any]) -> tuple[str, str, str, float]:
    return (
        str(state["lie"]),
        str(state["context"]),
        str(state["target"]),
        float(state["distance_yards"]),
    )


def baseline_table_sha256(
    states: Iterable[ExpectedStrokesStateV2 | Mapping[str, Any]],
) -> str:
    """Hash normalized states independent of JSON number spelling and row order."""

    canonical = [_coerce_state(state) for state in states]
    canonical.sort(key=_state_sort_key)
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


__all__ = [
    "BASELINE_CONTRACT_VERSION",
    "ExpectedStrokesBaselineV2",
    "ExpectedStrokesStateV2",
    "baseline_table_sha256",
]
