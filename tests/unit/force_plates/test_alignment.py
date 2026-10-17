"""Clock alignment between a plate trial and the model time axis (OF-05)."""

from __future__ import annotations

import math

import pytest

from src.shared.python.force_plates.alignment import PlateTimeAlignment

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _alignment(**overrides: object) -> PlateTimeAlignment:
    kwargs: dict[str, object] = {
        "offset_s": 1.0,
        "sigma_offset_s": 0.002,
        "kind": "manual",
        "note": "hand aligned on the visible impact frame",
    }
    kwargs.update(overrides)
    return PlateTimeAlignment(**kwargs)  # type: ignore[arg-type]


def test_trial_time_subtracts_offset() -> None:
    assert _alignment(offset_s=1.0).trial_time(1.007) == pytest.approx(0.007)


def test_trial_time_subtracts_offset_and_shift() -> None:
    got = _alignment(offset_s=1.0).trial_time(1.007, shift_s=0.001)
    assert got == pytest.approx(0.006)


def test_negative_offset_moves_trial_time_later() -> None:
    assert _alignment(offset_s=-0.5).trial_time(0.0) == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_trial_time_rejects_non_finite_inputs(bad: float) -> None:
    with pytest.raises(ValueError, match="model_time_s"):
        _alignment().trial_time(bad)
    with pytest.raises(ValueError, match="shift_s"):
        _alignment().trial_time(1.0, shift_s=bad)


def test_trial_time_rejects_non_numbers() -> None:
    with pytest.raises(TypeError, match="model_time_s"):
        _alignment().trial_time("1.0")  # type: ignore[arg-type]


def test_bad_kind_refused() -> None:
    with pytest.raises(ValueError, match="kind"):
        _alignment(kind="wishful")


@pytest.mark.parametrize("sigma", [-0.001, math.nan, math.inf])
def test_bad_sigma_refused(sigma: float) -> None:
    with pytest.raises(ValueError, match="sigma_offset_s"):
        _alignment(sigma_offset_s=sigma)


def test_zero_sigma_is_allowed() -> None:
    assert _alignment(sigma_offset_s=0.0).sigma_offset_s == 0.0


@pytest.mark.parametrize("offset", [math.nan, math.inf])
def test_non_finite_offset_refused(offset: float) -> None:
    with pytest.raises(ValueError, match="offset_s"):
        _alignment(offset_s=offset)


def test_non_numeric_offset_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="offset_s"):
        _alignment(offset_s="1")


@pytest.mark.parametrize("note", ["", "   "])
def test_empty_note_refused(note: str) -> None:
    with pytest.raises(ValueError, match="note"):
        _alignment(note=note)


def test_cross_correlation_requires_model_identity() -> None:
    with pytest.raises(ValueError, match="derived_from_model"):
        _alignment(kind="cross_correlation")
    with pytest.raises(ValueError, match="derived_from_model"):
        _alignment(kind="cross_correlation", derived_from_model="  ")


@pytest.mark.parametrize("kind", ["absolute_utc", "shot_impact", "manual"])
def test_model_identity_forbidden_for_other_kinds(kind: str) -> None:
    with pytest.raises(ValueError, match="derived_from_model"):
        _alignment(kind=kind, derived_from_model="model-a")


def test_validates_is_false_only_for_the_model_it_was_derived_from() -> None:
    xcorr = _alignment(kind="cross_correlation", derived_from_model="model-a")
    assert xcorr.validates("model-a") is False
    assert xcorr.validates("model-b") is True


@pytest.mark.parametrize("kind", ["absolute_utc", "shot_impact", "manual"])
def test_independent_alignments_validate_any_model(kind: str) -> None:
    assert _alignment(kind=kind).validates("model-a") is True


def test_validates_requires_text() -> None:
    with pytest.raises(ValueError, match="model_identity"):
        _alignment().validates("")


def test_alignment_is_frozen() -> None:
    with pytest.raises(AttributeError):
        _alignment().offset_s = 2.0  # type: ignore[misc]
