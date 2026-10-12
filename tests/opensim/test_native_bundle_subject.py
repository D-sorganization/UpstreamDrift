"""BundleSubject parameter object enforces its construction contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.engines.physics_engines.opensim.python.tour_matching.native_muscle_bundle import (
    BundleSubject,
)

pytestmark = pytest.mark.unit


def _subject(**overrides: object) -> BundleSubject:
    fields: dict[str, object] = {
        "path": Path("model.osim"),
        "model": object(),
        "names": ("a",),
        "muscles": ("m",),
        "options": {"o": (1.0,)},
    }
    fields.update(overrides)
    return BundleSubject(**fields)  # type: ignore[arg-type]


def test_valid_subject_is_frozen() -> None:
    subject = _subject()
    assert subject.names == ("a",)
    with pytest.raises(FrozenInstanceError):
        subject.names = ("b",)  # type: ignore[misc]


@pytest.mark.parametrize(
    "override",
    [
        {"path": "model.osim"},
        {"names": ["a"]},
        {"muscles": ["m"]},
        {"options": [("o", (1.0,))]},
    ],
)
def test_invalid_inputs_are_rejected(override: dict[str, object]) -> None:
    with pytest.raises(TypeError):
        _subject(**override)
