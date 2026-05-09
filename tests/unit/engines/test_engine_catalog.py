"""Tests for the engine catalog."""

from __future__ import annotations

from src.engines import get_engine_catalog, is_fit_capable


def test_get_engine_catalog() -> None:
    """Test that the engine catalog correctly parses engine capabilities."""
    catalog = get_engine_catalog()

    assert "pendulum" in catalog
    assert "putting_green" in catalog
    assert "mujoco" in catalog

    # Pendulum is fit_capable, putting green is not
    assert catalog["pendulum"]["fit_capable"] is True
    assert catalog["putting_green"]["fit_capable"] is False


def test_is_fit_capable() -> None:
    """Test the is_fit_capable helper function."""
    assert is_fit_capable("pendulum") is True
    assert is_fit_capable("mujoco") is True
    assert is_fit_capable("putting_green") is False
    assert is_fit_capable("nonexistent_engine") is False
