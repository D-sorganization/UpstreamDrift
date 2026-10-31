"""Externally sourced putting reference fixtures (epic #14141, step 2).

Loads ``src/config/putting_reference_fixtures.json``, the single fixture
file shared by the Python tests, the standalone HTML studio and the
PyQt6 tests. Every expected value cites data published or measured
outside UpstreamDrift (USGA, Holmes/Penner/Grober, Broadie/ShotLink,
measured-roll studies); none is a snapshot of any model's own output.
Fixtures detect drift against the outside world; they do not on their
own prove physical accuracy.

A fixture whose comparison cannot pass yet (no external value, no
public target, or a documented disagreement with the source) carries a
``pending_reason``; runners mark it ``xfail(strict=True)``.

Design by Contract:
    Preconditions: the file is a JSON object with a ``schema_version``
    and a non-empty ``fixtures`` list; each fixture has a non-empty
    citation, locator and tolerance basis, a boolean ``verified``, an
    ``applies_to`` subset of :data:`VALID_APPLIES_TO`, and a value or
    range unless it is pending.
    Postconditions: fixture ids are unique; fixtures are immutable.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared.python.contracts import ensure, require

#: Runtimes that consume the shared fixture file.
VALID_APPLIES_TO = frozenset({"python", "js", "pyqt6"})

#: Repository location of the shared fixture file.
DEFAULT_FIXTURES_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "putting_reference_fixtures.json"
)

#: ``maps_to`` targets are dotted paths below this package root.
_TARGET_ROOT = "src.shared.python"


def _text(data: dict[str, Any], key: str, context: str) -> str:
    """Return ``data[key]`` as a non-empty string (precondition)."""
    value = data.get(key)
    require(
        isinstance(value, str) and value.strip() != "",
        f"{context}: '{key}' must be a non-empty string",
        value,
    )
    return str(value)


def _object(data: Any, key: str, context: str) -> dict[str, Any]:
    """Return ``data`` as a JSON object (precondition)."""
    require(isinstance(data, dict), f"{context}: '{key}' must be an object", data)
    return dict(data)


@dataclass(frozen=True)
class FixtureSource:
    """Provenance of one expected value.

    ``verified`` is true only when the primary source text was read
    directly; secondary quotations stay false.
    """

    citation: str
    doi_or_url: str
    locator: str
    verified: bool


@dataclass(frozen=True)
class ExpectedValue:
    """External target (point ``value`` or ``min``/``max`` range)."""

    units: str
    tolerance_basis: str
    value: float | None = None
    min: float | None = None
    max: float | None = None
    tolerance_abs: float | None = None
    tolerance_rel: float | None = None

    @property
    def has_target(self) -> bool:
        """True when a point value or a full range is present."""
        return self.value is not None or (self.min is not None and self.max is not None)


@dataclass(frozen=True)
class ReferenceFixture:
    """One externally sourced reference check.

    ``maps_to`` is a dotted path below ``src.shared.python`` (for
    example ``putting_dynamics.stimp_to_rolling_mu``) naming the
    function or constant the fixture is compared against.
    """

    fixture_id: str
    quantity: str
    inputs: dict[str, Any]
    expected: ExpectedValue
    source: FixtureSource
    applies_to: tuple[str, ...]
    maps_to: str | None = None
    pending_reason: str | None = None

    def resolve_target(self) -> Any:
        """Import and return the ``maps_to`` attribute.

        Raises:
            ValueError: If the fixture has no ``maps_to``.
            ImportError, AttributeError: If the path does not resolve.
        """
        require(self.maps_to is not None, f"{self.fixture_id}: no maps_to target")
        module_path, _, attr = f"{_TARGET_ROOT}.{self.maps_to}".rpartition(".")
        return getattr(importlib.import_module(module_path), attr)


def _parse_fixture(data: Any) -> ReferenceFixture:
    data = _object(data, "fixture", "fixture")
    fixture_id = _text(data, "fixture_id", "fixture")
    context = f"fixture {fixture_id!r}"
    pending = data.get("pending_reason")
    if pending is not None:
        _text(data, "pending_reason", context)
    applies_to = data.get("applies_to")
    require(
        isinstance(applies_to, list)
        and len(applies_to) > 0
        and set(applies_to) <= VALID_APPLIES_TO,
        f"{context}: 'applies_to' must be a non-empty subset of "
        f"{sorted(VALID_APPLIES_TO)} (unknown runner)",
        applies_to,
    )
    raw_source = _object(data.get("source"), "source", context)
    verified = raw_source.get("verified")
    require(
        isinstance(verified, bool), f"{context}: 'verified' must be a bool", verified
    )
    raw_expected = _object(data.get("expected"), "expected", context)
    expected = ExpectedValue(
        units=_text(raw_expected, "units", context),
        tolerance_basis=_text(raw_expected, "tolerance_basis", context),
        **{
            key: raw_expected.get(key)
            for key in ("value", "min", "max", "tolerance_abs", "tolerance_rel")
        },
    )
    require(
        pending is not None or expected.has_target,
        f"{context}: expected needs 'value' or 'min'/'max' unless a "
        "'pending_reason' is given",
        raw_expected,
    )
    return ReferenceFixture(
        fixture_id=fixture_id,
        quantity=_text(data, "quantity", context),
        inputs=dict(data.get("inputs") or {}),
        expected=expected,
        source=FixtureSource(
            citation=_text(raw_source, "citation", context),
            doi_or_url=_text(raw_source, "doi_or_url", context),
            locator=_text(raw_source, "locator", context),
            verified=bool(verified),
        ),
        applies_to=tuple(applies_to or ()),
        maps_to=data.get("maps_to"),
        pending_reason=pending,
    )


def load_reference_fixtures(path: Path | None = None) -> tuple[ReferenceFixture, ...]:
    """Load and validate the shared putting reference-fixture file.

    Args:
        path: Fixture file; defaults to :data:`DEFAULT_FIXTURES_PATH`.

    Returns:
        Validated, immutable fixtures in file order.

    Raises:
        FileNotFoundError: If the file is absent.
        ValueError: If the file violates the schema contract.
    """
    fixtures_path = DEFAULT_FIXTURES_PATH if path is None else path
    raw = _object(
        json.loads(fixtures_path.read_text(encoding="utf-8")), "root", "fixture file"
    )
    _text(raw, "schema_version", "fixture file")
    entries = raw.get("fixtures")
    require(
        isinstance(entries, list) and len(entries) > 0,
        "fixture file: 'fixtures' must be a non-empty list",
        entries,
    )
    fixtures = tuple(_parse_fixture(entry) for entry in entries or ())
    ids = [fixture.fixture_id for fixture in fixtures]
    ensure(len(set(ids)) == len(ids), "fixture file: duplicate fixture_id", ids)
    return fixtures
