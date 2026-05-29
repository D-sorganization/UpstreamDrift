from sidekick.calculators.electrical.config import ElectrodeConfig


def test_electrode_config_initialization() -> None:
    config = ElectrodeConfig()

    assert config.k_scaling_factor == 0.035
    assert config.glass_depth == 15.0
    assert "status_ok" in config.colors
    assert "default" in config.color_schemes


def test_electrode_config_status_color() -> None:
    config = ElectrodeConfig()

    assert config.status_color("ok") == "#C8FFC8"
    assert config.status_color("warn") == "#FFFFB4"
    assert config.status_color("error") == "#FF9696"
    assert config.status_color("unknown") == "#C8FFC8"


def test_electrode_config_scheme_color() -> None:
    config = ElectrodeConfig()

    assert config.scheme_color("default", "direct_glass") == "#4169E1"
    assert config.scheme_color("default", "via_metal_down") == "#DC143C"
    assert config.scheme_color("unknown_scheme", "unknown_path") == "lightblue"
