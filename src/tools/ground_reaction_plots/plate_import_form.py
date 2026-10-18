"""Text form behind the plate declarations dialog (OF-07 follow-up, #13087).

A swing file cannot carry the plate's uncertainty, sensing depth, capacity,
how its channels' sampling is skewed, where the stance frame sits in the
model's world, or how the plate clock maps onto model time.  The dialog collects those as text; this module turns the
text into a :class:`PlateImportContext` and names the field at fault when it
cannot.  Physical facts start empty: they are declared, never assumed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from src.shared.python.force_plates import (
    AlignmentKind,
    PlateTimeAlignment,
    ReadingOptions,
    TargetedPlacement,
)
from src.shared.python.force_plates.open_fp_format import PlateDeclarations
from src.shared.python.force_plates.open_fp_format.declared_import import (
    COP_SKEW_CHOICES,
    DeclaredPlate,
    DeclaredReading,
    DeclaredStance,
)
from src.shared.python.force_plates.open_fp_format.overlay_file import DEFAULT_COVERAGE
from src.shared.python.impact_parameters.target_frame import UD_DEFAULT_TARGET_DIR

__all__ = [
    "ALIGNMENT_KINDS",
    "FORM_FIELDS",
    "FormField",
    "PlateFormError",
    "PlateImportContext",
    "context_from_form",
]

#: Clock alignments the form offers.  A ``cross_correlation`` offset was fitted
#: against a model, so the form also asks which one: that alignment can never
#: be evidence for the model it was fitted to (``PlateTimeAlignment.validates``).
ALIGNMENT_KINDS = ("manual", "shot_impact", "absolute_utc", "cross_correlation")
HANDEDNESS_CHOICES = ("RH", "LH")
_FRAME_LABEL = "declared"


@dataclass(frozen=True)
class PlateImportContext:
    """Facts a plate trial file cannot carry: set by the host before importing.

    ``declared`` is the plate uncertainty and geometry, ``placement`` the
    stance frame in the model's world, ``alignment`` the clock mapping and
    ``options`` the reading budget (``None`` for the reader's default).
    """

    declared: PlateDeclarations
    placement: TargetedPlacement
    alignment: PlateTimeAlignment
    options: ReadingOptions | None = None


@dataclass(frozen=True)
class FormField:
    """One dialog row: ``key`` in the text mapping, ``label`` with its unit."""

    key: str
    label: str
    section: str
    required: bool = True
    choices: tuple[str, ...] = ()
    initial: str = ""


_DEFAULT_TARGET_TEXT = ", ".join(f"{v:g}" for v in UD_DEFAULT_TARGET_DIR)

FORM_FIELDS: tuple[FormField, ...] = (
    FormField("device", "Device label (optional)", "Plate", required=False),
    FormField("uncertainty_source", "Uncertainty source", "Plate"),
    FormField("force_floor_n", "Load uncertainty floor [N]", "Plate"),
    FormField("force_fraction", "Load uncertainty fraction [-]", "Plate"),
    FormField("cop_m", "CoP uncertainty [m]", "Plate"),
    FormField("sensing_depth_m", "Sensing plane depth [m]", "Plate"),
    FormField("sensing_depth_sigma_m", "Sensing depth uncertainty [m]", "Plate"),
    FormField("capacity_n", "Pair capacity [N]", "Plate"),
    FormField("skew_s", "Channel timing skew within a sample [s]", "Plate"),
    FormField(
        "cop_skew",
        "CoP channel skews",
        "Plate",
        choices=COP_SKEW_CHOICES,
        initial=COP_SKEW_CHOICES[0],
    ),
    FormField(
        "handedness", "Handedness", "Stance", choices=HANDEDNESS_CHOICES, initial="RH"
    ),
    FormField("origin_m", "Stance origin x, y, z [m]", "Stance"),
    FormField(
        "target_dir", "Target direction x, y, z", "Stance", initial=_DEFAULT_TARGET_TEXT
    ),
    FormField("offset_s", "Clock offset, model = trial + offset [s]", "Clock"),
    FormField("sigma_offset_s", "Clock offset uncertainty [s]", "Clock"),
    FormField(
        "alignment_kind",
        "Alignment kind",
        "Clock",
        choices=ALIGNMENT_KINDS,
        initial=ALIGNMENT_KINDS[0],
    ),
    FormField(
        "derived_from_model",
        "Model the offset was fitted against (cross-correlation only)",
        "Clock",
        required=False,
    ),
    FormField("alignment_note", "How the offset was found", "Clock"),
)

_FIELDS = {field.key: field for field in FORM_FIELDS}
_SECTION_FIRST = {"Plate": "uncertainty_source", "Stance": "origin_m"}


class PlateFormError(ValueError):
    """A form value was missing or refused; ``key`` names the field."""

    def __init__(self, key: str, message: str) -> None:
        super().__init__(f"{key}: {message}")
        self.key = key


def _text(texts: Mapping[str, str], key: str) -> str:
    value = str(texts.get(key, _FIELDS[key].initial)).strip()
    if not value and _FIELDS[key].required:
        raise PlateFormError(key, "required")
    return value


def _number(texts: Mapping[str, str], key: str) -> float:
    text = _text(texts, key)
    try:
        value = float(text)
    except ValueError:
        raise PlateFormError(key, f"expected a number, got {text!r}") from None
    if not math.isfinite(value):
        raise PlateFormError(key, f"expected a finite number, got {text!r}")
    return value


def _vector(texts: Mapping[str, str], key: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in _text(texts, key).split(",")]
    try:
        values = tuple(float(p) for p in parts)
    except ValueError:
        raise PlateFormError(key, "expected three comma-separated numbers") from None
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise PlateFormError(key, "expected three finite comma-separated numbers")
    return values  # type: ignore[return-value]


def _choice(texts: Mapping[str, str], key: str) -> str:
    value = _text(texts, key)
    if value not in _FIELDS[key].choices:
        raise PlateFormError(key, f"must be one of {_FIELDS[key].choices}")
    return value


def _plate(texts: Mapping[str, str]) -> DeclaredPlate:
    return DeclaredPlate(
        force_floor_n=_number(texts, "force_floor_n"),
        force_fraction=_number(texts, "force_fraction"),
        cop_m=_number(texts, "cop_m"),
        sensing_depth_sigma_m=_number(texts, "sensing_depth_sigma_m"),
        uncertainty_source=_text(texts, "uncertainty_source"),
        sensing_depth_m=_number(texts, "sensing_depth_m"),
        capacity_n=_number(texts, "capacity_n"),
        device=_text(texts, "device") or None,
    )


def _options(texts: Mapping[str, str]) -> ReadingOptions:
    reading = DeclaredReading(
        skew_s=_number(texts, "skew_s"), cop_skew=_choice(texts, "cop_skew")
    )
    try:
        return reading.to_options(DEFAULT_COVERAGE)
    except (TypeError, ValueError) as exc:
        raise PlateFormError(_named_key(exc, "Plate"), str(exc)) from exc


def _stance(texts: Mapping[str, str]) -> DeclaredStance:
    return DeclaredStance(
        handedness=_choice(texts, "handedness"),
        origin_m=_vector(texts, "origin_m"),
        target_dir=_vector(texts, "target_dir"),
    )


def _alignment(texts: Mapping[str, str]) -> PlateTimeAlignment:
    offset_s = _number(texts, "offset_s")
    sigma_offset_s = _number(texts, "sigma_offset_s")
    kind = cast(AlignmentKind, _choice(texts, "alignment_kind"))
    note = _text(texts, "alignment_note")
    derived_from_model = _text(texts, "derived_from_model") or None
    try:
        return PlateTimeAlignment(
            offset_s=offset_s,
            sigma_offset_s=sigma_offset_s,
            kind=kind,
            note=note,
            derived_from_model=derived_from_model,
        )
    except (TypeError, ValueError) as exc:
        raise PlateFormError(_named_key(exc, "Clock"), str(exc)) from exc


def _named_key(exc: Exception, section: str) -> str:
    """The form key a domain refusal names, else the section's first field."""
    message = str(exc)
    named = [k for k, f in _FIELDS.items() if f.section == section and k in message]
    return max(named, key=len) if named else _SECTION_FIRST.get(section, "offset_s")


def context_from_form(texts: Mapping[str, str]) -> PlateImportContext:
    """The :class:`PlateImportContext` the form texts declare.

    Missing keys take the field's initial text.  Raises :class:`PlateFormError`
    (a ``ValueError``) naming the field for an unknown key, a missing required
    value, unparseable text, or a value the force-plate domain refuses.
    """
    unknown = sorted(set(texts) - set(_FIELDS))
    if unknown:
        raise PlateFormError(unknown[0], "unknown form field")
    for field in FORM_FIELDS:  # report the topmost missing row first
        _text(texts, field.key)
    plate, stance = _plate(texts), _stance(texts)
    try:
        declared = plate.to_declarations()
    except (TypeError, ValueError) as exc:
        raise PlateFormError(_named_key(exc, "Plate"), str(exc)) from exc
    try:
        placement = stance.to_placement(frame_label=_FRAME_LABEL)
    except (TypeError, ValueError) as exc:
        raise PlateFormError(_named_key(exc, "Stance"), str(exc)) from exc
    return PlateImportContext(declared, placement, _alignment(texts), _options(texts))
