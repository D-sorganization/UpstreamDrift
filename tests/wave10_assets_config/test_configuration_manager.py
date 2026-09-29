"""Tests for ConfigurationManager and SimulationConfig (config_utils gaps)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.shared.python.config.configuration_manager import (
    ConfigurationManager,
    SimulationConfig,
)
from src.shared.python.core import GolfModelingError


class TestSimulationConfigValidate:
    def test_defaults_validate(self) -> None:
        SimulationConfig().validate()  # no exception

    def test_negative_height_raises(self) -> None:
        cfg = SimulationConfig(height_m=-0.1)
        with pytest.raises(GolfModelingError, match="height_m"):
            cfg.validate()

    def test_zero_height_raises(self) -> None:
        cfg = SimulationConfig(height_m=0.0)
        with pytest.raises(GolfModelingError):
            cfg.validate()

    def test_negative_weight_raises(self) -> None:
        cfg = SimulationConfig(weight_percent=-1)
        with pytest.raises(GolfModelingError, match="weight_percent"):
            cfg.validate()

    def test_negative_club_length_raises(self) -> None:
        cfg = SimulationConfig(club_length=0.0)
        with pytest.raises(GolfModelingError, match="club_length"):
            cfg.validate()

    def test_invalid_control_mode_raises(self) -> None:
        cfg = SimulationConfig(control_mode="bogus")
        with pytest.raises(GolfModelingError, match="control_mode"):
            cfg.validate()

    def test_valid_control_modes(self) -> None:
        for mode in ("pd", "lqr", "poly"):
            SimulationConfig(control_mode=mode).validate()


class TestConfigurationManagerRoundTrip:
    def test_load_missing_returns_defaults(self, tmp_path: Path) -> None:
        mgr = ConfigurationManager(tmp_path / "nonexistent.json")
        cfg = mgr.load()
        assert isinstance(cfg, SimulationConfig)
        assert cfg.height_m == SimulationConfig().height_m

    def test_save_then_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.json"
        mgr = ConfigurationManager(path)
        original = SimulationConfig(height_m=1.9, club_length=1.2)
        mgr.save(original)
        loaded = mgr.load()
        assert loaded.height_m == 1.9
        assert loaded.club_length == 1.2

    def test_load_ignores_unknown_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps({"height_m": 1.85, "junk_key": "ignored"}))
        cfg = ConfigurationManager(path).load()
        assert cfg.height_m == 1.85

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.json"
        path.write_text("{not-valid-json")
        with pytest.raises(GolfModelingError, match="malformed JSON"):
            ConfigurationManager(path).load()

    def test_load_validates_loaded_config(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps({"height_m": -1.0}))
        with pytest.raises(GolfModelingError):
            ConfigurationManager(path).load()

    def test_save_to_unwritable_dir_raises(self, tmp_path: Path) -> None:
        # Use a path whose parent is a file, not a directory
        impossible = tmp_path / "afile"
        impossible.write_text("x")
        mgr = ConfigurationManager(impossible / "child" / "cfg.json")
        with pytest.raises(GolfModelingError, match="Failed to save"):
            mgr.save(SimulationConfig())

    def test_config_path_attribute_set(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.json"
        mgr = ConfigurationManager(path)
        assert mgr.config_path == path
