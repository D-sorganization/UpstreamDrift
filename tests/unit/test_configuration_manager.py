"""Unit tests for the ConfigurationManager."""

import json
from pathlib import Path
import pytest
from shared.python.configuration_manager import ConfigurationManager, SimulationConfig
from shared.python.common_utils import GolfModelingError

def test_default_config():
    """Test that default configuration is valid."""
    config = SimulationConfig()
    config.validate()  # Should not raise
    assert config.height_m == 1.8
    assert config.weight_percent == 100.0
    assert config.control_mode == "pd"

def test_config_validation():
    """Test validation logic."""
    with pytest.raises(GolfModelingError):
        SimulationConfig(height_m=-1.0).validate()
    
    with pytest.raises(GolfModelingError):
        SimulationConfig(control_mode="invalid").validate()

def test_save_load(tmp_path):
    """Test saving and loading configuration."""
    config_file = tmp_path / "test_config.json"
    manager = ConfigurationManager(config_file)
    
    # Save default
    config = SimulationConfig()
    config.height_m = 2.0
    config.colors["shirt"] = [1.0, 0.0, 0.0, 1.0]
    manager.save(config)
    
    assert config_file.exists()
    
    # Load back
    loaded_config = manager.load()
    assert loaded_config.height_m == 2.0
    assert loaded_config.colors["shirt"] == [1.0, 0.0, 0.0, 1.0]

def test_load_partial(tmp_path):
    """Test loading a config file with missing or extra fields."""
    config_file = tmp_path / "partial.json"
    data = {"height_m": 1.5, "extra_field": "ignore_me"}
    with open(config_file, "w") as f:
        json.dump(data, f)
        
    manager = ConfigurationManager(config_file)
    config = manager.load()
    
    assert config.height_m == 1.5
    assert config.weight_percent == 100.0  # Default value
    assert not hasattr(config, "extra_field")

def test_load_nonexistent():
    """Test loading a non-existent file returns defaults."""
    manager = ConfigurationManager(Path("nonexistent.json"))
    config = manager.load()
    assert config.height_m == 1.8

