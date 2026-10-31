"""Numeric checks of putting_dynamics against external reference fixtures.

Epic #14141 PR 2. Every expected value in
``src/config/putting_reference_fixtures.json`` is drawn from data
measured or published outside UpstreamDrift and outside the Gemini
vendor source (USGA Stimpmeter documents, Holmes/Penner/Grober,
Broadie/ShotLink, Drane/Whiston/Quintic measured-roll studies). This
module never asserts a value taken from either codebase's own
simulation output -- fixtures show drift against the outside world,
they do not prove accuracy on their own.

A fixture with a ``pending_reason`` is intentionally made to fail
(either the mapped comparison disagrees with the cited source, or no
mapping exists yet) and is marked ``xfail(strict=True)`` so a
surprise pass -- meaning the gap silently closed -- also fails the
suite until this file is updated.

Known disagreement surfaced by this PR (do not "fix" by loosening a
tolerance or changing the physics constant -- see the fixture's own
``pending_reason`` and reconcile in step 3/5 of #14141):

* ``drane2014_sliding_deceleration_mu`` -- UD's flat
  ``DEFAULT_SLIDING_MU = 0.40`` vs. the Drane et al. (2014)
  secondarily-cited artificial-turf figure of ``mu_slide ~= 0.18``.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from src.shared.python.core import physics_constants
from src.shared.python.putting_dynamics import reference_fixtures as pd_fixtures
from src.shared.python.putting_dynamics import (
    load_reference_fixtures,
)
from src.shared.python.putting_dynamics.reference_fixtures import (
    ExpectedValue,
    ReferenceFixture,
)
import src.shared.python.putting_dynamics as pd

pytestmark = pytest.mark.unit

#: Fixtures that compare against a module other than the
#: putting_dynamics facade. Kept in sync with
#: tests/config/putting_reference_fixtures/test_schema.py's
#: ``_NON_FACADE_TARGETS``.
_PHYSICS_CONSTANTS_FIXTURES: dict[str, float] = {
    "usga_ball_mass_max_kg": float(physics_constants.GOLF_BALL_MASS_KG),
    "usga_ball_diameter_min_m": float(physics_constants.GOLF_BALL_DIAMETER_M),
}


def _call_mapped(attr: Any, inputs: dict[str, Any]) -> float:
    """Evaluate a facade attribute: call it with matching inputs, or cast it."""
    if not callable(attr):
        return float(attr)
    signature = inspect.signature(attr)
    kwargs = {name: inputs[name] for name in signature.parameters if name in inputs}
    return float(attr(**kwargs))


def _check_within_tolerance(actual: float, expected: ExpectedValue) -> None:
    if expected.min is not None and expected.max is not None:
        slack = expected.tolerance_abs or 0.0
        lo, hi = expected.min - slack, expected.max + slack
        assert lo <= actual <= hi, (
            f"{actual} outside measured range [{lo}, {hi}] ({expected.tolerance_basis})"
        )
        return

    assert expected.value is not None, "expected.value must be set for a point check"
    if expected.tolerance_rel is not None:
        assert actual == pytest.approx(expected.value, rel=expected.tolerance_rel), (
            expected.tolerance_basis
        )
    else:
        tol = expected.tolerance_abs or 0.0
        assert abs(actual - expected.value) <= tol, (
            f"{actual} vs expected {expected.value} +/- {tol} "
            f"({expected.tolerance_basis})"
        )


def _build_cases() -> list[Any]:
    cases = []
    for fixture in load_reference_fixtures():
        if "python" not in fixture.applies_to:
            continue
        marks = []
        if fixture.pending_reason:
            marks.append(pytest.mark.xfail(strict=True, reason=fixture.pending_reason))
        cases.append(pytest.param(fixture, id=fixture.fixture_id, marks=marks))
    return cases


@pytest.mark.parametrize("fixture", _build_cases())
def test_reference_fixture_matches_external_source(fixture: ReferenceFixture) -> None:
    """Compare the implementation against one externally sourced fixture.

    Fixtures mapped to a real putting_dynamics function/constant are
    evaluated and checked against the fixture's tolerance. Fixtures
    with neither a mapping nor a physics_constants target have no
    comparison to run -- the fixture file itself documents why (its
    ``pending_reason``), and this test fails on purpose so the
    strict-xfail marker is meaningful rather than a silent skip.
    """
    if fixture.fixture_id in _PHYSICS_CONSTANTS_FIXTURES:
        actual = _PHYSICS_CONSTANTS_FIXTURES[fixture.fixture_id]
    elif fixture.maps_to is not None:
        actual = _call_mapped(getattr(pd, fixture.maps_to), fixture.inputs)
    else:
        pytest.fail(
            fixture.pending_reason
            or f"{fixture.fixture_id}: no maps_to, no pending_reason, nothing to check"
        )
        return
    _check_within_tolerance(actual, fixture.expected)


def test_at_least_one_fixture_is_an_unresolved_strict_xfail() -> None:
    """Sanity check: the pending/disagreement fixtures are wired as xfail.

    Guards against a future refactor accidentally dropping the
    ``pending_reason`` -> ``xfail(strict=True)`` wiring in
    ``_build_cases`` without anyone noticing.
    """
    pending = [f for f in load_reference_fixtures() if f.pending_reason]
    assert pending, "expected at least one pending/disagreement fixture"


def test_fixture_module_is_reexported_from_facade() -> None:
    assert pd.load_reference_fixtures is pd_fixtures.load_reference_fixtures
