"""Tests for upstream_drift_tools.ui.managers.unit_preferences_manager (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.upstream_drift_tools.ui.managers.unit_preferences_manager import (
    UNIT_CATEGORIES,
    UNIT_PRESETS,
    UnitPreferencesManager,
    get_unit_preferences_manager,
)


class TestUnitCategories:
    def test_has_temperature(self) -> None:
        assert "temperature" in UNIT_CATEGORIES

    def test_has_pressure(self) -> None:
        assert "pressure" in UNIT_CATEGORIES

    def test_categories_nonempty(self) -> None:
        assert len(UNIT_CATEGORIES) > 0


class TestUnitPresets:
    def test_unit_preferences_manager_is_dict(self) -> None:
        assert isinstance(UNIT_PRESETS, dict)

    def test_unit_preferences_manager_nonempty(self) -> None:
        assert len(UNIT_PRESETS) > 0


class TestGetUnitPreferencesManager:
    def test_returns_manager(self) -> None:
        mgr = get_unit_preferences_manager()
        assert isinstance(mgr, UnitPreferencesManager)

    def test_unit_preferences_manager_singleton(self) -> None:
        m1 = get_unit_preferences_manager()
        m2 = get_unit_preferences_manager()
        assert m1 is m2

    def test_get_preferred_unit(self) -> None:
        mgr = get_unit_preferences_manager()
        unit = mgr.get_preferred_unit("temperature")
        assert isinstance(unit, str)

    def test_get_si_unit(self) -> None:
        mgr = get_unit_preferences_manager()
        si_unit = mgr.get_si_unit("temperature")
        assert isinstance(si_unit, str)
        assert len(si_unit) > 0
