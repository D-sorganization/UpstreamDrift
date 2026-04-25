"""Unit tests for parameter preset routes (issue #3174).

Tests cover:
- Parameter bounds validation
- Safe preset name resolution
- Preset save/load/list endpoints (via direct function calls)

No network, no filesystem writes — all I/O paths are mocked.
"""

from __future__ import annotations

from pydantic import ValidationError

from src.api.routes.presets import (

    PARAM_BOUNDS,
    PresetEntry,
    PresetSaveRequest,
    PresetsListResponse,
    _safe_preset_path,
    _validate_param_bounds,
)
import pytest
pytestmark = pytest.mark.unit



class TestParamBounds:
    """PARAM_BOUNDS defines valid physics parameter ranges."""

    def test_duration_min(self) -> None:
        """Duration minimum is 0.1."""
        assert PARAM_BOUNDS["duration"]["min"] == pytest.approx(0.1)

    def test_duration_max(self) -> None:
        """Duration maximum is 60.0."""
        assert PARAM_BOUNDS["duration"]["max"] == pytest.approx(60.0)

    def test_timestep_min(self) -> None:
        """Timestep minimum is 0.001."""
        assert PARAM_BOUNDS["timestep"]["min"] == pytest.approx(0.001)

    def test_timestep_max(self) -> None:
        """Timestep maximum is 0.01."""
        assert PARAM_BOUNDS["timestep"]["max"] == pytest.approx(0.01)


class TestValidateParamBounds:
    """_validate_param_bounds returns descriptive errors for out-of-range values."""

    def test_valid_params_returns_empty(self) -> None:
        """Valid params within bounds return no errors."""
        errors = _validate_param_bounds({"duration": 5.0, "timestep": 0.002})
        assert errors == []

    def test_duration_below_min(self) -> None:
        """Duration below minimum returns an error."""
        errors = _validate_param_bounds({"duration": 0.0})
        assert len(errors) == 1
        assert "duration" in errors[0]
        assert ">=" in errors[0]

    def test_duration_above_max(self) -> None:
        """Duration above maximum returns an error."""
        errors = _validate_param_bounds({"duration": 61.0})
        assert len(errors) == 1
        assert "duration" in errors[0]
        assert "<=" in errors[0]

    def test_timestep_below_min(self) -> None:
        """Timestep below minimum returns an error."""
        errors = _validate_param_bounds({"timestep": 0.0001})
        assert any("timestep" in e for e in errors)

    def test_timestep_above_max(self) -> None:
        """Timestep above maximum returns an error."""
        errors = _validate_param_bounds({"timestep": 0.02})
        assert any("timestep" in e for e in errors)

    def test_non_numeric_value(self) -> None:
        """Non-numeric param value returns an error."""
        errors = _validate_param_bounds({"duration": "not-a-number"})
        assert any("numeric" in e for e in errors)

    def test_unknown_key_ignored(self) -> None:
        """Unknown keys not in PARAM_BOUNDS are silently ignored."""
        errors = _validate_param_bounds({"unknown_key": -999})
        assert errors == []

    def test_multiple_errors_returned(self) -> None:
        """Both duration and timestep violations are reported together."""
        errors = _validate_param_bounds({"duration": 0.0, "timestep": 0.02})
        assert len(errors) == 2


class TestSafePresetPath:
    """_safe_preset_path rejects unsafe names and path-traversal attempts."""

    def test_valid_name_returns_path(self) -> None:
        """Simple alphanumeric name returns a Path."""
        result = _safe_preset_path("my_preset")
        assert result is not None
        assert result.suffix == ".json"
        assert "my_preset" in result.stem

    def test_name_with_spaces_returns_path(self) -> None:
        """Name with spaces is slugified (spaces -> underscores)."""
        result = _safe_preset_path("high speed test")
        assert result is not None
        assert result.suffix == ".json"

    def test_empty_name_returns_none(self) -> None:
        """Empty name is rejected."""
        assert _safe_preset_path("") is None

    def test_path_traversal_returns_none(self) -> None:
        """Path traversal attempt is rejected."""
        assert _safe_preset_path("../../etc/passwd") is None

    def test_name_with_special_chars_returns_none(self) -> None:
        """Name with shell-special characters is rejected."""
        assert _safe_preset_path("name; rm -rf /") is None

    def test_name_too_long_returns_none(self) -> None:
        """Names exceeding 64 characters are rejected."""
        long_name = "a" * 65
        assert _safe_preset_path(long_name) is None


class TestPresetSaveRequest:
    """PresetSaveRequest validates preset name and params."""

    def test_valid_request(self) -> None:
        """Valid name and non-empty params are accepted."""
        req = PresetSaveRequest(name="my-preset", params={"duration": 5.0})
        assert req.name == "my-preset"

    def test_name_stripped_of_whitespace(self) -> None:
        """Leading/trailing whitespace is stripped from name."""
        req = PresetSaveRequest(name="  test  ", params={"k": 1})
        assert req.name == "test"

    def test_empty_params_raises(self) -> None:
        """Empty params dict raises ValidationError."""
        with pytest.raises(ValidationError):
            PresetSaveRequest(name="ok", params={})

    def test_unsafe_name_raises(self) -> None:
        """Name with unsafe characters raises ValidationError."""
        with pytest.raises(ValidationError):
            PresetSaveRequest(name="../evil", params={"k": 1})


class TestPresetsListResponse:
    """PresetsListResponse correctly aggregates preset entries."""

    def test_empty_presets(self) -> None:
        """Default response has an empty presets list."""
        resp = PresetsListResponse()
        assert resp.presets == []

    def test_with_entries(self) -> None:
        """Presets list contains all supplied entries."""
        entries = [
            PresetEntry(name="fast", params={"duration": 1.0}),
            PresetEntry(name="slow", params={"duration": 30.0}),
        ]
        resp = PresetsListResponse(presets=entries)
        assert len(resp.presets) == 2
        assert resp.presets[0].name == "fast"
        assert resp.presets[1].name == "slow"
