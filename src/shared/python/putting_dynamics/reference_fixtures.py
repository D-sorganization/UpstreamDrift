"""Externally sourced putting-physics reference fixtures (epic #14141 PR 2).

Loads ``src/config/putting_reference_fixtures.json`` -- the single
JSON fixture file that the Python unit tests, the future standalone
JS putting studio, and the PyQt6 desktop tests all share (one
source of truth, per the discovery workflow in AGENTS.md section A).

**Design rule (non-negotiable, see the fixture file itself for the
full citations)**: every fixture's expected value is drawn from data
measured or published *outside* UpstreamDrift and outside the Gemini
vendor source (USGA Stimpmeter documents, Holmes/Penner/Grober,
Broadie/ShotLink, Drane/Whiston/Quintic measured-roll studies). No
fixture value may be copied from either codebase's own simulation
output -- that would prove self-consistency, not accuracy. A fixture
whose external value is unverified (secondary citation, paywalled
primary source) or does not exist at all (``pending_reason`` set)
documents that gap rather than asserting a fabricated number.

Design by Contract:
    Preconditions:
        - The fixture file must exist and be valid JSON conforming to
          this module's schema.
        - Every fixture must carry a non-empty ``source.citation``,
          ``source.locator`` and ``expected.tolerance_basis``.
        - ``source.verified`` must be a real ``bool``.
        - ``applies_to`` must be a non-empty subset of
          ``{"python", "js", "pyqt6"}``.
    Postconditions:
        - ``fixture_id`` values are unique across the file.
    Invariants:
        - Loaded fixtures are immutable (frozen dataclasses).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared.python.contracts import ensure, require

__all__ = [
    "DEFAULT_FIXTURES_PATH",
    "VALID_APPLIES_TO",
    "ExpectedValue",
    "FixtureSource",
    "ReferenceFixture",
    "load_reference_fixtures",
]

#: Runtimes a fixture can declare relevance to; kept in sync with the
#: three surfaces named in epic #14141 (Python kernel, future JS
#: studio, PyQt6 desktop tests).
VALID_APPLIES_TO = frozenset({"python", "js", "pyqt6"})

#: ``putting_dynamics/reference_fixtures.py`` -> .../src/config/...
DEFAULT_FIXTURES_PATH = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "config"
    / "putting_reference_fixtures.json"
)


def _require_nonempty_str(value: Any, field_name: str, context: str) -> str:
    """Return ``value`` as a non-empty string or raise via DbC ``require``."""
    require(
        isinstance(value, str) and value.strip() != "",
        f"{context}: '{field_name}' must be a non-empty string",
        value,
    )
    return value


@dataclass(frozen=True)
class FixtureSource:
    """Provenance of one fixture's expected value.

    Attributes:
        citation: Human-readable citation (authors, title, venue, year).
        doi_or_url: DOI or URL, or free-text explaining why none exists.
        locator: Page/table/equation locator, or search-method note.
        verified: True only if a human/agent read the primary source
            text directly (not an abstract, not a secondary quotation,
            not a search-engine synthesis) and confirmed the number in
            context. Secondary citations and unresolved gaps are
            ``False`` -- kept honest rather than optimistic.
    """

    citation: str
    doi_or_url: str
    locator: str
    verified: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any], context: str) -> FixtureSource:
        require(isinstance(data, dict), f"{context}: 'source' must be an object", data)
        verified = data.get("verified")
        require(
            isinstance(verified, bool),
            f"{context}: source.verified must be a bool",
            verified,
        )
        return cls(
            citation=_require_nonempty_str(data.get("citation"), "citation", context),
            doi_or_url=_require_nonempty_str(
                data.get("doi_or_url"), "doi_or_url", context
            ),
            locator=_require_nonempty_str(data.get("locator"), "locator", context),
            verified=verified,
        )


@dataclass(frozen=True)
class ExpectedValue:
    """The externally sourced target and its tolerance.

    Exactly one of (``value``) or (``min`` and ``max``) is set, unless
    ``pending_reason`` on the owning :class:`ReferenceFixture` is set --
    a fixture documenting a gap (no external value exists yet) may
    carry neither.

    Attributes:
        value: A single target value, or ``None`` for a range check.
        min: Lower bound of a range check, or ``None``.
        max: Upper bound of a range check, or ``None``.
        units: Physical units of ``value``/``min``/``max``.
        tolerance_abs: Absolute tolerance (same units), or ``None``.
        tolerance_rel: Relative tolerance (fraction), or ``None``.
        tolerance_basis: Mandatory free text explaining *why* that
            tolerance -- tied to the source's stated measurement
            uncertainty, rounding, or sample size, never asserted bare.
    """

    value: float | None
    min: float | None
    max: float | None
    units: str
    tolerance_abs: float | None
    tolerance_rel: float | None
    tolerance_basis: str

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], context: str, *, is_pending: bool
    ) -> ExpectedValue:
        require(
            isinstance(data, dict), f"{context}: 'expected' must be an object", data
        )
        value = data.get("value")
        min_value = data.get("min")
        max_value = data.get("max")
        has_value = value is not None
        has_range = min_value is not None and max_value is not None
        require(
            is_pending or has_value or has_range,
            f"{context}: expected must carry 'value' or both 'min' and 'max' "
            "unless the fixture carries a 'pending_reason'",
            data,
        )
        expected = cls(
            value=value,
            min=min_value,
            max=max_value,
            units=_require_nonempty_str(data.get("units"), "units", context),
            tolerance_abs=data.get("tolerance_abs"),
            tolerance_rel=data.get("tolerance_rel"),
            tolerance_basis=_require_nonempty_str(
                data.get("tolerance_basis"), "tolerance_basis", context
            ),
        )
        return expected


@dataclass(frozen=True)
class ReferenceFixture:
    """One externally sourced putting-physics reference check.

    Attributes:
        fixture_id: Globally unique ``<source-slug>_<quantity-slug>``.
        quantity: Human-readable name of the physical quantity checked.
        inputs: Raw inputs needed to reproduce ``expected`` from the
            cited source's own formula (not UD's internals), so a
            JS/PyQt6 test can reconstruct the check without Python.
        expected: Target value/range, units and tolerance.
        source: Citation, locator and verification status.
        applies_to: Subset of ``{"python", "js", "pyqt6"}``.
        maps_to: Name of the ``putting_dynamics`` façade attribute
            this fixture checks, or ``None`` if no public function
            exists yet (always paired with ``pending_reason``, or the
            fixture compares against a different module such as
            ``physics_constants`` by ``fixture_id``).
        pending_reason: Set when the check cannot currently pass --
            either no external value exists, no mapped function
            exists, or (documented) the current implementation
            disagrees with the cited external source. Test runners
            use this to mark the case ``xfail(strict=True)``.
    """

    fixture_id: str
    quantity: str
    inputs: dict[str, Any]
    expected: ExpectedValue
    source: FixtureSource
    applies_to: tuple[str, ...]
    maps_to: str | None
    pending_reason: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReferenceFixture:
        require(isinstance(data, dict), "fixture must be an object", data)
        fixture_id = _require_nonempty_str(
            data.get("fixture_id"), "fixture_id", "fixture"
        )
        context = f"fixture {fixture_id!r}"

        pending_reason = data.get("pending_reason")
        if pending_reason is not None:
            _require_nonempty_str(pending_reason, "pending_reason", context)

        applies_to_raw = data.get("applies_to")
        require(
            isinstance(applies_to_raw, list) and len(applies_to_raw) > 0,
            f"{context}: 'applies_to' must be a non-empty list",
            applies_to_raw,
        )
        unknown = set(applies_to_raw) - VALID_APPLIES_TO
        require(
            not unknown,
            f"{context}: 'applies_to' contains unknown runner(s) {sorted(unknown)}; "
            f"must be a subset of {sorted(VALID_APPLIES_TO)}",
            applies_to_raw,
        )

        maps_to = data.get("maps_to")
        if maps_to is not None:
            _require_nonempty_str(maps_to, "maps_to", context)

        return cls(
            fixture_id=fixture_id,
            quantity=_require_nonempty_str(data.get("quantity"), "quantity", context),
            inputs=data.get("inputs") or {},
            expected=ExpectedValue.from_dict(
                data.get("expected"), context, is_pending=pending_reason is not None
            ),
            source=FixtureSource.from_dict(data.get("source"), context),
            applies_to=tuple(applies_to_raw),
            maps_to=maps_to,
            pending_reason=pending_reason,
        )


def load_reference_fixtures(
    path: Path | None = None,
) -> tuple[ReferenceFixture, ...]:
    """Load and validate the shared putting reference-fixture file.

    Args:
        path: Fixture file location; defaults to
            :data:`DEFAULT_FIXTURES_PATH`.

    Returns:
        Validated, frozen fixtures in file order.

    Raises:
        FileNotFoundError: If the fixture file is absent.
        ValueError: If any fixture violates the schema contract (via
            :func:`src.shared.python.contracts.require`/``ensure``) --
            including a missing source citation/locator/tolerance
            basis, a duplicate ``fixture_id``, or an unknown
            ``applies_to`` entry.
    """
    fixtures_path = path if path is not None else DEFAULT_FIXTURES_PATH
    if not fixtures_path.exists():
        raise FileNotFoundError(
            f"Putting reference fixtures not found: {fixtures_path}"
        )

    raw = json.loads(fixtures_path.read_text(encoding="utf-8"))
    require(isinstance(raw, dict), "fixture file root must be a JSON object", raw)
    _require_nonempty_str(raw.get("schema_version"), "schema_version", "fixture file")

    raw_fixtures = raw.get("fixtures")
    require(
        isinstance(raw_fixtures, list) and len(raw_fixtures) > 0,
        "fixture file: 'fixtures' must be a non-empty list",
        raw_fixtures,
    )

    fixtures = tuple(ReferenceFixture.from_dict(entry) for entry in raw_fixtures)

    ids = [fixture.fixture_id for fixture in fixtures]
    require(
        len(set(ids)) == len(ids),
        f"fixture file: duplicate fixture_id values in {ids}",
        ids,
    )
    ensure(len(fixtures) > 0, "fixture file: must load at least one fixture", fixtures)
    return fixtures
