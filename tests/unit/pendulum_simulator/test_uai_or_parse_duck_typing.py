"""``_uai_or_parse`` reads SI values by duck type, not class identity (#15522).

When another test purges and re-imports the sidekick widget module, a live
``UnitAwareInput`` no longer passes ``isinstance`` against the freshly imported
class. Its ``value`` is a method, so falling through to ``parse_float`` raised
``TypeError: float() argument must be ... not 'method'``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from src.shared.python.pendulum_simulator.gui.controls_widget_base import (  # noqa: E402
    ControlsWidgetBase,
)

pytestmark = pytest.mark.unit


class _ReimportedUnitAwareInput:
    """Stands in for a UnitAwareInput built from a stale class object."""

    def value(self) -> float:
        return 99.0  # display units; must not be used

    def value_si(self) -> float:
        return 1.25


class _PlainInput:
    value = "3.5"


def test_unit_aware_input_from_stale_class_returns_si_value() -> None:
    assert ControlsWidgetBase._uai_or_parse(_ReimportedUnitAwareInput(), "x") == 1.25


def test_plain_labeled_input_still_parses_text() -> None:
    assert ControlsWidgetBase._uai_or_parse(_PlainInput(), "x") == 3.5


def test_unparseable_text_names_the_field() -> None:
    bad = _PlainInput()
    bad.value = "abc"
    with pytest.raises(ValueError, match="Mass"):
        ControlsWidgetBase._uai_or_parse(bad, "Mass")
