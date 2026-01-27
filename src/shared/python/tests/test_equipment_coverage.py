import pytest

from shared.python.equipment import CLUB_CONFIGS, get_club_config


def test_get_club_config_success():
    for club_type in CLUB_CONFIGS:
        config = get_club_config(club_type)
        assert isinstance(config, dict)
        assert "head_mass" in config
        assert "shaft_length" in config


def test_get_club_config_invalid():
    with pytest.raises(ValueError) as exc:
        get_club_config("invalid_club")
    assert "Invalid club_type" in str(exc.value)


def test_config_integrity():
    # Ensure all configs have standard fields
    required_fields = ["grip_length", "head_mass", "total_length"]
    for config in CLUB_CONFIGS.values():
        for field in required_fields:
            assert field in config
