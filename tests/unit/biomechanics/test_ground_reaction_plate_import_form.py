"""Pure-data layer of the plate declarations dialog (OF-07 follow-up, #13087).

The dialog only collects text; ``plate_import_form`` turns it into a
``PlateImportContext`` and names the field at fault when it cannot.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.tools.ground_reaction_plots.plate_import_form import (
    ALIGNMENT_KINDS,
    FORM_FIELDS,
    PlateFormError,
    PlateImportContext,
    context_from_form,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _texts(**changes: str) -> dict[str, str]:
    texts = {
        "device": "Lab pair A",
        "uncertainty_source": "datasheet rev 3",
        "force_floor_n": "2.0",
        "force_fraction": "0.01",
        "cop_m": "0.010",
        "sensing_depth_m": "0.015",
        "sensing_depth_sigma_m": "0.002",
        "capacity_n": "3923",
        "handedness": "RH",
        "origin_m": "1.0, 2.0, 0.0",
        "target_dir": "0, -1, 0",
        "offset_s": "0.25",
        "sigma_offset_s": "0.001",
        "alignment_kind": "manual",
        "alignment_note": "clap on both clocks",
    }
    texts.update(changes)
    return texts


def test_complete_form_builds_the_context() -> None:
    context = context_from_form(_texts())
    assert isinstance(context, PlateImportContext)
    assert context.declared.capacity_n == 3923.0
    assert context.declared.device == "Lab pair A"
    assert context.declared.uncertainty.source == "datasheet rev 3"
    assert context.placement.handedness == "RH"
    assert context.placement.frame_id.startswith("declared:target_")
    np.testing.assert_allclose(context.placement.placement.origin_m, (1.0, 2.0, 0.0))
    assert context.alignment.offset_s == 0.25
    assert context.alignment.kind == "manual"
    assert context.options is None


def test_every_physical_fact_is_required_and_starts_empty() -> None:
    required = {f.key for f in FORM_FIELDS if f.required}
    assert {
        "uncertainty_source",
        "force_floor_n",
        "force_fraction",
        "cop_m",
        "sensing_depth_m",
        "sensing_depth_sigma_m",
        "capacity_n",
        "origin_m",
        "offset_s",
        "sigma_offset_s",
        "alignment_note",
    } <= required
    for field in FORM_FIELDS:
        if field.key not in {"target_dir", "handedness", "alignment_kind"}:
            assert field.initial == "", field.key


@pytest.mark.parametrize("key", ["capacity_n", "origin_m", "alignment_note"])
def test_missing_required_field_is_named(key: str) -> None:
    with pytest.raises(PlateFormError, match=key) as info:
        context_from_form(_texts(**{key: "  "}))
    assert info.value.key == key


def test_blank_device_keeps_the_default_label() -> None:
    assert context_from_form(_texts(device="")).declared.device


@pytest.mark.parametrize(
    ("key", "text"),
    [
        ("capacity_n", "lots"),
        ("force_floor_n", "nan"),
        ("origin_m", "1, 2"),
        ("target_dir", "0, -1, x"),
    ],
)
def test_unparseable_numbers_are_named(key: str, text: str) -> None:
    with pytest.raises(PlateFormError) as info:
        context_from_form(_texts(**{key: text}))
    assert info.value.key == key


@pytest.mark.parametrize(
    ("key", "text"),
    [
        ("capacity_n", "-1"),
        ("sigma_offset_s", "-0.1"),
        ("handedness", "ambi"),
        ("target_dir", "0, 0, 1"),
    ],
)
def test_domain_refusals_become_form_errors(key: str, text: str) -> None:
    with pytest.raises(PlateFormError):
        context_from_form(_texts(**{key: text}))


def test_cross_correlation_is_not_typed_by_hand() -> None:
    assert "cross_correlation" not in ALIGNMENT_KINDS
    with pytest.raises(PlateFormError) as info:
        context_from_form(_texts(alignment_kind="cross_correlation"))
    assert info.value.key == "alignment_kind"


def test_unknown_field_is_refused() -> None:
    with pytest.raises(PlateFormError, match="unknown"):
        context_from_form({**_texts(), "colour": "red"})


def test_gui_still_exports_the_context_type() -> None:
    pytest.importorskip("PyQt6.QtWidgets")
    from src.tools.ground_reaction_plots import gui

    assert gui.PlateImportContext is PlateImportContext
