"""The web plate declarations form mirrors the PyQt6 dialog's fields (#13122).

The PyQt6 form is canonical; the web form adds only ``trial`` (the swing file
named relative to the server's plate-trial root, where the desktop tile uses a
file dialog).  Keys, order, labels, required flags, choices and initial texts
must match, so the two forms cannot drift apart.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.shared.python.force_plates.open_fp_format.declared_import import (
    COP_SKEW_CHOICES,
)
from src.tools.ground_reaction_plots.plate_import_form import (
    ALIGNMENT_KINDS,
    FORM_FIELDS,
    HANDEDNESS_CHOICES,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_WEB_FORM = (
    Path(__file__).resolve().parents[3]
    / "ui"
    / "src"
    / "components"
    / "analysis"
    / "plateImportForm.ts"
)
_FIELD = re.compile(
    r"field\(\s*'(?P<key>[a-z_]+)',\s*'(?P<label>[^']*)',\s*'(?P<section>[A-Za-z]+)'"
    r"(?P<extra>,\s*\{[^}]*\})?,?\s*\)",
)
_WEB_ONLY = ("trial",)


def _web_source() -> str:
    return _WEB_FORM.read_text(encoding="utf-8")


def _web_fields() -> list[dict[str, str]]:
    return [
        m.groupdict() | {"extra": m["extra"] or ""}
        for m in _FIELD.finditer(_web_source())
    ]


def _quoted_tuple(name: str) -> tuple[str, ...]:
    match = re.search(rf"export const {name} = \[([^\]]*)\] as const", _web_source())
    assert match, f"{name} not found in {_WEB_FORM.name}"
    return tuple(re.findall(r"'([^']*)'", match.group(1)))


def test_web_form_has_the_dialog_fields_in_order() -> None:
    web = [f["key"] for f in _web_fields() if f["key"] not in _WEB_ONLY]
    assert web == [f.key for f in FORM_FIELDS]


def test_labels_and_sections_match() -> None:
    web = {f["key"]: f for f in _web_fields()}
    for field in FORM_FIELDS:
        assert web[field.key]["label"] == field.label, field.key
        assert web[field.key]["section"] == field.section, field.key


def test_required_flags_and_initial_texts_match() -> None:
    web = {f["key"]: f["extra"] for f in _web_fields()}
    for field in FORM_FIELDS:
        extra = web[field.key]
        assert ("required: false" in extra) == (not field.required), field.key
        if field.choices:
            continue  # initial choice is checked against the choice tuples below
        expected = f"initial: '{field.initial}'" if field.initial else ""
        assert (expected in extra) if expected else "initial:" not in extra, field.key


def test_choices_match() -> None:
    assert _quoted_tuple("ALIGNMENT_KINDS") == ALIGNMENT_KINDS
    assert _quoted_tuple("HANDEDNESS_CHOICES") == HANDEDNESS_CHOICES
    assert _quoted_tuple("COP_SKEW_CHOICES") == COP_SKEW_CHOICES
