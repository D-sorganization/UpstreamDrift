# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for import_results module."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest
from conftest import make_test_result

from movement_optimizer.export import export_result_json
from movement_optimizer.import_results import (
    EXPORT_FORMAT_VERSION,
    import_result_from_json,
    import_results_from_json,
)
from movement_optimizer.persistence import save_solution


class TestImportResultFromJson:
    def test_valid_json_with_format_version(self, tmp_path):
        """Successfully loads a valid JSON file that contains format_version."""
        data = {
            "format_version": EXPORT_FORMAT_VERSION,
            "cost": 12.5,
            "success": True,
        }
        path = tmp_path / "result.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        result = import_result_from_json(path)

        assert result["format_version"] == EXPORT_FORMAT_VERSION
        assert result["cost"] == 12.5
        assert result["success"] is True

    def test_file_not_found(self, tmp_path):
        """Raises FileNotFoundError for a missing file."""
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError, match="Result file not found"):
            import_result_from_json(missing)

    def test_invalid_json_raises_value_error(self, tmp_path):
        """Raises ValueError for files containing invalid JSON."""
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{{", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            import_result_from_json(path)

    def test_non_dict_json_raises_value_error(self, tmp_path):
        """Raises ValueError when the JSON root is not an object (dict)."""
        path = tmp_path / "list.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(ValueError, match="expected JSON object"):
            import_result_from_json(path)

    def test_legacy_file_without_format_version_emits_warning(self, tmp_path, caplog):
        """Legacy files without format_version are accepted with a warning."""
        data = {"cost": 99.0, "success": False}
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="movement_optimizer.import_results"):
            result = import_result_from_json(path)

        assert result["cost"] == 99.0
        assert any("format_version" in msg for msg in caplog.messages)

    def test_incompatible_format_version_raises_value_error(self, tmp_path):
        """Raises ValueError when the file's format_version is not supported."""
        data = {"format_version": "99.0", "cost": 0.0}
        path = tmp_path / "future.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="Incompatible format_version"):
            import_result_from_json(path)

    def test_round_trip_with_export_result_json(self, tmp_path):
        """Data written by export_result_json is readable by import_result_from_json."""
        original = {"cost": 7.25, "n_evals": 200}
        out_path = str(tmp_path / "round_trip.json")
        export_result_json(original, out_path)

        loaded = import_result_from_json(out_path)

        assert loaded["cost"] == original["cost"]
        assert loaded["n_evals"] == original["n_evals"]
        assert loaded["format_version"] == EXPORT_FORMAT_VERSION

    def test_accepts_path_as_string(self, tmp_path):
        """import_result_from_json accepts a plain string path."""
        data = {"format_version": EXPORT_FORMAT_VERSION, "value": 1}
        path = tmp_path / "str_path.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        result = import_result_from_json(str(path))
        assert result["value"] == 1


class TestImportResultsFromJson:
    def test_imports_saved_solution_as_optimization_result(self, tmp_path):
        original = make_test_result()
        path = tmp_path / "solution.json"
        save_solution(
            path,
            original,
            {"body_mass": 80.0, "height": 1.8},
            "squat",
            60.0,
        )

        imported = import_results_from_json(path)

        np.testing.assert_allclose(imported.t, original.t)
        np.testing.assert_allclose(imported.q, original.q)
        np.testing.assert_allclose(imported.torques, original.torques)
        assert imported.success is original.success
        assert imported.cost == pytest.approx(original.cost)
        assert imported.n_evals == original.n_evals

    def test_rejects_future_saved_solution_format(self, tmp_path):
        original = make_test_result()
        path = tmp_path / "future_solution.json"
        save_solution(path, original, {"body_mass": 80.0}, "squat", 60.0)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["format_version"] = 999
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="unsupported format_version"):
            import_results_from_json(path)
