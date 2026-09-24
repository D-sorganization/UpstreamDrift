"""Tests for unified display unit system and conversion helpers (issue #8886)."""

from __future__ import annotations

import pytest

from src.shared.python.ui import units as units_module
from src.shared.python.ui.units import (
    UnitSystem,
    distance_suffix,
    format_distance,
    format_speed,
    format_spin,
    from_display_distance,
    from_display_mass,
    from_display_speed,
    get_unit_preference,
    mass_suffix,
    set_unit_preference,
    speed_suffix,
    spin_suffix,
    to_display_distance,
    to_display_mass,
    to_display_speed,
)

pytestmark = pytest.mark.unit


class TestUnitSystemEnum:
    def test_unit_system_values(self) -> None:
        assert UnitSystem.METRIC.value == "metric"
        assert UnitSystem.IMPERIAL.value == "imperial"
        assert str(UnitSystem.METRIC) == "metric"
        assert str(UnitSystem.IMPERIAL) == "imperial"


class TestUnitPreference:
    def test_get_and_set_unit_preference(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setattr(units_module, "user_config_dir", lambda: tmp_path)
        set_unit_preference(UnitSystem.IMPERIAL)
        assert get_unit_preference() == UnitSystem.IMPERIAL
        set_unit_preference(UnitSystem.METRIC)
        assert get_unit_preference() == UnitSystem.METRIC

    def test_set_unit_preference_string(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setattr(units_module, "user_config_dir", lambda: tmp_path)
        set_unit_preference("imperial")
        assert get_unit_preference() == UnitSystem.IMPERIAL
        set_unit_preference("metric")
        assert get_unit_preference() == UnitSystem.METRIC

    def test_set_unit_preference_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid UnitSystem"):
            set_unit_preference("nautical")  # type: ignore[arg-type]


class TestConversions:
    def test_distance_round_trip_metric(self) -> None:
        m = 150.0
        disp = to_display_distance(m, UnitSystem.METRIC)
        assert disp == pytest.approx(150.0)
        assert from_display_distance(disp, UnitSystem.METRIC) == pytest.approx(m)

    def test_distance_round_trip_imperial_yards(self) -> None:
        m = 100.0
        disp = to_display_distance(m, UnitSystem.IMPERIAL, unit="yd")
        assert disp == pytest.approx(109.3613, rel=1e-3)
        assert from_display_distance(
            disp, UnitSystem.IMPERIAL, unit="yd"
        ) == pytest.approx(m)

    def test_distance_round_trip_imperial_feet(self) -> None:
        m = 3.048  # exactly 10 ft
        disp = to_display_distance(m, UnitSystem.IMPERIAL, unit="ft")
        assert disp == pytest.approx(10.0, rel=1e-3)
        assert from_display_distance(
            disp, UnitSystem.IMPERIAL, unit="ft"
        ) == pytest.approx(m)

    def test_speed_round_trip_metric(self) -> None:
        ms = 45.0
        disp = to_display_speed(ms, UnitSystem.METRIC)
        assert disp == pytest.approx(45.0)
        assert from_display_speed(disp, UnitSystem.METRIC) == pytest.approx(ms)

    def test_speed_round_trip_imperial(self) -> None:
        ms = 44.704  # ~100 mph
        disp = to_display_speed(ms, UnitSystem.IMPERIAL)
        assert disp == pytest.approx(100.0, rel=1e-3)
        assert from_display_speed(disp, UnitSystem.IMPERIAL) == pytest.approx(ms)

    def test_mass_round_trip_metric(self) -> None:
        kg = 0.200
        disp = to_display_mass(kg, UnitSystem.METRIC)
        assert disp == pytest.approx(0.200)
        assert from_display_mass(disp, UnitSystem.METRIC) == pytest.approx(kg)

    def test_mass_round_trip_imperial(self) -> None:
        kg = 0.45359237  # 1 lb
        disp = to_display_mass(kg, UnitSystem.IMPERIAL)
        assert disp == pytest.approx(1.0, rel=1e-3)
        assert from_display_mass(disp, UnitSystem.IMPERIAL) == pytest.approx(kg)


class TestFormatting:
    def test_format_distance_metric(self) -> None:
        formatted = format_distance(100.0, UnitSystem.METRIC, include_secondary=True)
        assert "100.0 m" in formatted
        assert "yd" in formatted

        single = format_distance(100.0, UnitSystem.METRIC, include_secondary=False)
        assert single == "100.0 m"

    def test_format_distance_imperial(self) -> None:
        formatted = format_distance(100.0, UnitSystem.IMPERIAL, include_secondary=True)
        assert "yd" in formatted.split("(")[0]
        assert "100.0 m" in formatted.split("(")[1]

        single = format_distance(100.0, UnitSystem.IMPERIAL, include_secondary=False)
        assert "yd" in single
        assert "m" not in single

    def test_format_distance_feet(self) -> None:
        formatted = format_distance(
            3.048, UnitSystem.IMPERIAL, unit="ft", include_secondary=True
        )
        assert "10.0 ft" in formatted
        assert "3.0 m" in formatted or "3.05 m" in formatted

    def test_format_speed_metric(self) -> None:
        formatted = format_speed(45.0, UnitSystem.METRIC, include_secondary=True)
        assert "45.0 m/s" in formatted
        assert "mph" in formatted

        single = format_speed(45.0, UnitSystem.METRIC, include_secondary=False)
        assert single == "45.0 m/s"

    def test_format_speed_imperial(self) -> None:
        formatted = format_speed(45.0, UnitSystem.IMPERIAL, include_secondary=True)
        assert "mph" in formatted.split("(")[0]
        assert "45.0 m/s" in formatted.split("(")[1]

    def test_format_spin(self) -> None:
        assert format_spin(2500.0) == "2500 rpm"
        assert format_spin(2500.0, 150.0) == "2500 rpm backspin, 150 rpm sidespin"
        assert format_spin(2500.0, -150.0) == "2500 rpm backspin, -150 rpm sidespin"

    def test_suffixes(self) -> None:
        assert distance_suffix(UnitSystem.METRIC) == " m"
        assert distance_suffix(UnitSystem.IMPERIAL, unit="yd") == " yd"
        assert distance_suffix(UnitSystem.IMPERIAL, unit="ft") == " ft"
        assert speed_suffix(UnitSystem.METRIC) == " m/s"
        assert speed_suffix(UnitSystem.IMPERIAL) == " mph"
        assert mass_suffix(UnitSystem.METRIC) == " kg"
        assert mass_suffix(UnitSystem.IMPERIAL) == " lb"
        assert spin_suffix() == " rpm"


class TestPreferencesDialogUnits:
    def test_preferences_dialog_units_group(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        from PyQt6.QtWidgets import QApplication, QMainWindow
        from src.shared.python.ui.preferences_dialog import PreferencesDialog

        _ = QApplication.instance() or QApplication([])
        monkeypatch.setattr(
            "src.shared.python.ui.preferences_dialog.PREFS_DIR", tmp_path
        )
        monkeypatch.setattr(
            "src.shared.python.ui.preferences_dialog.PREFS_FILE",
            tmp_path / "preferences.json",
        )

        parent = QMainWindow()
        dlg = PreferencesDialog(parent)
        try:
            assert hasattr(dlg, "unit_system_combo")
            assert dlg.unit_system_combo.currentText() == "Metric (m, m/s, kg)"
            dlg.unit_system_combo.setCurrentText("Imperial (yd, mph, lb)")
            prefs = dlg._collect_preferences()
            assert prefs.unit_system == "imperial"
        finally:
            dlg.close()
            dlg.deleteLater()
            parent.close()
            parent.deleteLater()
