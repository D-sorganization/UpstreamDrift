"""Check ``putting_dynamics`` against externally sourced fixtures (#14141).

Each expected value in ``src/config/putting_reference_fixtures.json``
cites outside data; none is a snapshot of the model's own output. A
fixture with a ``pending_reason`` is a known gap or disagreement and
runs as ``xfail(strict=True)``, so a silent fix also fails until the
fixture file is updated. Fixtures detect drift; they do not prove
physical accuracy.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from src.shared.python.putting_dynamics import (
    ExpectedValue,
    ReferenceFixture,
    load_reference_fixtures,
)

pytestmark = pytest.mark.unit


def _evaluate(fixture: ReferenceFixture) -> float:
    """Call the mapped function with matching inputs, or read the constant."""
    target = fixture.resolve_target()
    if not callable(target):
        return float(target)
    params = inspect.signature(target).parameters
    return float(target(**{k: v for k, v in fixture.inputs.items() if k in params}))


def _assert_within(actual: float, expected: ExpectedValue) -> None:
    if expected.value is None:
        assert expected.min is not None and expected.max is not None
        slack = expected.tolerance_abs or 0.0
        assert expected.min - slack <= actual <= expected.max + slack
        return
    assert actual == pytest.approx(
        expected.value,
        rel=expected.tolerance_rel,
        abs=expected.tolerance_abs or 0.0,
    ), expected.tolerance_basis


def _cases() -> list[Any]:
    return [
        pytest.param(
            fixture,
            id=fixture.fixture_id,
            marks=(
                [pytest.mark.xfail(strict=True, reason=fixture.pending_reason)]
                if fixture.pending_reason
                else []
            ),
        )
        for fixture in load_reference_fixtures()
        if "python" in fixture.applies_to
    ]


@pytest.mark.parametrize("fixture", _cases())
def test_matches_external_reference(fixture: ReferenceFixture) -> None:
    if fixture.maps_to is None or not fixture.expected.has_target:
        pytest.fail(f"{fixture.fixture_id}: {fixture.pending_reason}")
    _assert_within(_evaluate(fixture), fixture.expected)
