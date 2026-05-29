"""Tests for :mod:`shared.python.launcher_embed.contract`."""

from __future__ import annotations

import dataclasses

import pytest

from src.shared.python.launcher_embed import EmbedCapabilities


@pytest.mark.unit
def test_defaults_round_trip() -> None:
    """Default-constructed ``EmbedCapabilities`` exposes the documented values."""
    caps = EmbedCapabilities()
    assert caps.supports_embedded is True
    assert caps.prefers_dock is False
    assert caps.min_size == (640, 480)
    assert caps.requires_separate_qapplication is False


@pytest.mark.unit
def test_explicit_values_round_trip() -> None:
    caps = EmbedCapabilities(
        supports_embedded=False,
        prefers_dock=True,
        min_size=(320, 240),
        requires_separate_qapplication=True,
    )
    assert caps.supports_embedded is False
    assert caps.prefers_dock is True
    assert caps.min_size == (320, 240)
    assert caps.requires_separate_qapplication is True


@pytest.mark.unit
def test_min_size_zero_rejected() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        EmbedCapabilities(min_size=(0, 0))


@pytest.mark.unit
def test_min_size_negative_rejected() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        EmbedCapabilities(min_size=(-1, 100))


@pytest.mark.unit
def test_min_size_wrong_shape_rejected() -> None:
    with pytest.raises(ValueError, match="2-tuple"):
        EmbedCapabilities(min_size=(100,))  # type: ignore[arg-type]


@pytest.mark.unit
def test_min_size_not_tuple_rejected() -> None:
    with pytest.raises(ValueError, match="tuple"):
        EmbedCapabilities(min_size=[640, 480])  # type: ignore[arg-type]


@pytest.mark.unit
def test_min_size_non_int_rejected() -> None:
    with pytest.raises(ValueError, match="ints"):
        EmbedCapabilities(min_size=(640.0, 480.0))  # type: ignore[arg-type]


@pytest.mark.unit
def test_frozen_assignment_raises() -> None:
    """``EmbedCapabilities`` is frozen; attribute assignment must fail."""
    caps = EmbedCapabilities()
    with pytest.raises(dataclasses.FrozenInstanceError):
        caps.supports_embedded = False  # type: ignore[misc]
