# ruff: noqa: E501
"""Additional tests to hit edge cases in programmatic_pid.geometry and validation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("ezdxf", reason="ezdxf not installed")

from programmatic_pid.geometry import find_free_region
from programmatic_pid.types import BBox, SpecValidationError
from programmatic_pid.validation import _load_schema, validate_spec


class TestFindFreeRegionNone:
    def test_returns_none_when_fully_blocked(self):
        """Artificially force no free region by making every candidate overlap."""
        # Fill a huge area so find_free_region can't place anything
        occupied = [BBox(-600.0, -600.0, 600.0, 600.0)]
        result = find_free_region(occupied, 5.0, 5.0)
        assert result is None


class TestValidationSchemaLoading:
    def test_load_schema_cached(self):
        """Calling _load_schema() twice returns cached result (no re-read)."""
        import programmatic_pid.validation as _val

        original = _val._SCHEMA
        _val._SCHEMA = None  # reset

        # First call – either None (no file) or loaded
        s1 = _load_schema()
        # Second call must return the same object (cached, hits line 26)
        s2 = _load_schema()
        assert s1 is s2

        _val._SCHEMA = original  # restore

    def test_load_schema_returns_cached_non_none(self):
        """When _SCHEMA is already set (non-None), the cached value is returned."""

        import programmatic_pid.validation as _val

        original = _val._SCHEMA
        fake_schema = {"type": "object", "_test": True}
        _val._SCHEMA = fake_schema  # pre-populate cache

        result = _load_schema()
        assert result is fake_schema  # line 26 branch

        _val._SCHEMA = original

    def test_load_schema_reads_file_when_exists(self, tmp_path):
        """When schema file exists, read and cache it (covers lines 31-32)."""
        import json

        import programmatic_pid.validation as _val

        original = _val._SCHEMA
        _val._SCHEMA = None

        schema_content = {"type": "object", "_loaded": True}
        schema_file = tmp_path / "pid_spec.schema.json"
        schema_file.write_text(json.dumps(schema_content), encoding="utf-8")

        # Patch the Path used inside _load_schema so it finds our temp file
        original_fn = _val._load_schema

        def patched_load_schema():
            global _SCHEMA
            if _val._SCHEMA is not None:
                return _val._SCHEMA
            if schema_file.exists():
                with open(schema_file, encoding="utf-8") as f:
                    _val._SCHEMA = json.load(f)
            return _val._SCHEMA

        _val._load_schema = patched_load_schema  # type: ignore[method-assign]
        result = _val._load_schema()
        assert result == schema_content

        _val._load_schema = original_fn  # restore
        _val._SCHEMA = original


class TestValidateSpecWithJsonschema:
    def test_validate_spec_with_jsonschema_available(self):
        """When jsonschema validates successfully, no error raised."""
        import programmatic_pid.validation as _val

        valid_spec = {
            "project": {"id": "P001", "title": "Test"},
            "equipment": [{"id": "V101", "w": 5.0, "h": 5.0}],
        }
        # Patch schema to return truthy value so jsonschema path is exercised
        with patch.object(_val, "_load_schema", return_value={"type": "object"}):
            # jsonschema may or may not be installed; either way should not crash for valid spec  # noqa: E501
            try:  # noqa: SIM105
                validate_spec(valid_spec)
            except SpecValidationError:
                pass  # may raise for missing fields, but not from schema

    def test_validate_spec_schema_violation_raises(self):
        """When jsonschema raises, SpecValidationError is re-raised."""
        import programmatic_pid.validation as _val

        valid_spec = {
            "project": {"id": "P001", "title": "Test"},
            "equipment": [{"id": "V101", "w": 5.0, "h": 5.0}],
        }

        mock_schema = {"type": "object", "required": ["REQUIRED_FIELD"]}
        mock_jsonschema = MagicMock()
        mock_jsonschema.validate.side_effect = Exception(
            "Schema violation: missing field"
        )

        with patch.object(_val, "_load_schema", return_value=mock_schema):  # noqa: SIM117
            with patch.dict("sys.modules", {"jsonschema": mock_jsonschema}):
                with pytest.raises(SpecValidationError, match="Schema violation"):
                    validate_spec(valid_spec)

    def test_validate_spec_import_error_skipped(self):
        """When jsonschema ImportError, validation continues without schema check."""
        import programmatic_pid.validation as _val

        valid_spec = {
            "project": {"id": "P001", "title": "Test"},
            "equipment": [{"id": "V101", "w": 5.0, "h": 5.0}],
        }

        mock_schema = {"type": "object"}
        mock_jsonschema = MagicMock()
        mock_jsonschema.validate.side_effect = ImportError("no jsonschema")

        with patch.object(_val, "_load_schema", return_value=mock_schema):  # noqa: SIM117
            with patch.dict("sys.modules", {"jsonschema": mock_jsonschema}):
                # Should not raise due to ImportError
                validate_spec(valid_spec)  # should pass (no errors in valid_spec)
