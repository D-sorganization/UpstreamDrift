# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for persistence module -- save/load solutions and app state."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import make_test_result

from movement_optimizer.config import load_app_paths
from movement_optimizer.persistence import (
    RESULT_FORMAT_VERSION,
    SCHEMA_VERSION,
    InvalidStateFileError,
    load_app_state,
    load_solution,
    save_app_state,
    save_solution,
)


class TestSaveSolution:
    def test_round_trip_preserves_all_fields(self, tmp_path):
        result = make_test_result()
        body_params = {
            "body_mass": 80.0,
            "height": 1.80,
            "seg_multipliers": {"lower_leg": 1.0, "upper_leg": 1.0, "torso": 1.0},
        }
        path = tmp_path / "solution.json"

        save_solution(str(path), result, body_params, "squat", 60.0)
        loaded = load_solution(str(path))

        assert loaded["exercise_type"] == "squat"
        assert loaded["bar_mass"] == 60.0
        assert loaded["body_params"] == body_params
        assert loaded["format_version"] == RESULT_FORMAT_VERSION
        assert "export_timestamp" in loaded
        assert "export_app_version" in loaded
        assert loaded["metadata"]["success"] is True
        assert loaded["metadata"]["cost"] == pytest.approx(42.5)
        assert loaded["metadata"]["com_horizontal_range_cm"] == pytest.approx(3.2)
        assert loaded["metadata"]["elapsed_s"] == pytest.approx(1.5)
        assert loaded["metadata"]["n_evals"] == 100

    def test_round_trip_preserves_arrays(self, tmp_path):
        result = make_test_result()
        body_params = {"body_mass": 75.0, "height": 1.75}
        path = tmp_path / "solution.json"

        save_solution(str(path), result, body_params, "deadlift", 100.0)
        loaded = load_solution(str(path))

        for key in ("t", "q", "qd", "qdd", "torques", "power", "com", "bar"):
            original = getattr(result, key)
            restored = np.array(loaded["arrays"][key])
            np.testing.assert_allclose(restored, original, atol=1e-10)

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_solution("/nonexistent/path/solution.json")

    def test_load_corrupt_file_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{{")
        with pytest.raises(json.JSONDecodeError):
            load_solution(str(path))


class TestAppState:
    def test_round_trip_state(self, tmp_path):
        result = make_test_result()
        results_dict = {"squat": result}
        slider_values = {
            "body_mass": 80.0,
            "height": 1.80,
            "lower_leg": 1.0,
            "upper_leg": 1.1,
            "torso": 0.9,
            "bar_mass": 60.0,
            "duration": 2.0,
            "smoothness": 1.0,
        }

        state_dir = tmp_path / ".movement_optimizer"
        save_app_state(results_dict, slider_values, state_dir=str(state_dir))
        loaded = load_app_state(state_dir=str(state_dir))

        assert loaded is not None
        assert loaded["slider_values"] == slider_values
        assert "squat" in loaded["results"]
        np.testing.assert_allclose(
            np.array(loaded["results"]["squat"]["arrays"]["t"]),
            result.t,
            atol=1e-10,
        )

    def test_load_missing_state_returns_none(self, tmp_path):
        state_dir = tmp_path / "nonexistent"
        loaded = load_app_state(state_dir=str(state_dir))
        assert loaded is None

    def test_load_corrupt_state_returns_none(self, tmp_path):
        state_dir = tmp_path / ".movement_optimizer"
        state_dir.mkdir(parents=True)
        (state_dir / "last_state.json").write_text("corrupted{{{")
        loaded = load_app_state(state_dir=str(state_dir))
        assert loaded is None

    def test_env_override_controls_default_state_path(self, tmp_path, monkeypatch):
        override_dir = tmp_path / "custom-state"
        monkeypatch.setenv("MOVEMENT_OPTIMIZER_STATE_DIR", str(override_dir))

        save_app_state({"squat": make_test_result()}, {"body_mass": 80.0})
        loaded = load_app_state()

        assert loaded is not None
        assert load_app_paths().state_file == override_dir / "last_state.json"
        assert Path(load_app_paths().state_file).exists()


def _make_valid_app_state_payload() -> dict:
    """Return a minimal but schema-valid app-state JSON payload."""
    return {
        "schema_version": SCHEMA_VERSION,
        "slider_values": {
            "body_mass": 80.0,
            "height": 1.80,
            "lower_leg": 1.0,
            "upper_leg": 1.0,
            "torso": 1.0,
            "bar_mass": 60.0,
            "duration": 2.0,
            "smoothness": 1.0,
        },
        "results": {},
    }


def _write_state(tmp_path, payload) -> Path:
    state_dir = tmp_path / ".movement_optimizer"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "last_state.json").write_text(json.dumps(payload), encoding="utf-8")
    return state_dir


class TestSchemaValidationAppState:
    """Issue #403: explicit schema validation on app-state load."""

    def test_round_trip_includes_schema_version(self, tmp_path):
        state_dir = tmp_path / ".movement_optimizer"
        save_app_state({}, {"body_mass": 80.0}, state_dir=str(state_dir))

        raw = (state_dir / "last_state.json").read_text(encoding="utf-8")
        payload = json.loads(raw)
        assert payload["schema_version"] == SCHEMA_VERSION

        loaded = load_app_state(state_dir=str(state_dir))
        assert loaded is not None
        assert loaded["schema_version"] == SCHEMA_VERSION

    def test_missing_schema_version_raises(self, tmp_path):
        payload = _make_valid_app_state_payload()
        del payload["schema_version"]
        state_dir = _write_state(tmp_path, payload)
        with pytest.raises(InvalidStateFileError, match="schema_version"):
            load_app_state(state_dir=str(state_dir))

    def test_unknown_schema_version_raises(self, tmp_path):
        payload = _make_valid_app_state_payload()
        payload["schema_version"] = SCHEMA_VERSION + 99
        state_dir = _write_state(tmp_path, payload)
        with pytest.raises(InvalidStateFileError, match="unsupported schema_version"):
            load_app_state(state_dir=str(state_dir))

    def test_missing_required_key_raises(self, tmp_path):
        payload = _make_valid_app_state_payload()
        del payload["slider_values"]
        state_dir = _write_state(tmp_path, payload)
        with pytest.raises(InvalidStateFileError, match="slider_values"):
            load_app_state(state_dir=str(state_dir))

    def test_wrong_type_for_slider_values_raises(self, tmp_path):
        payload = _make_valid_app_state_payload()
        payload["slider_values"] = "not a dict"
        state_dir = _write_state(tmp_path, payload)
        with pytest.raises(InvalidStateFileError, match="slider_values"):
            load_app_state(state_dir=str(state_dir))

    def test_wrong_type_for_individual_slider_raises(self, tmp_path):
        payload = _make_valid_app_state_payload()
        payload["slider_values"]["body_mass"] = "heavy"
        state_dir = _write_state(tmp_path, payload)
        with pytest.raises(InvalidStateFileError, match=r"slider_values\.body_mass"):
            load_app_state(state_dir=str(state_dir))

    def test_out_of_range_slider_value_raises(self, tmp_path):
        payload = _make_valid_app_state_payload()
        payload["slider_values"]["body_mass"] = 5000.0  # absurd
        state_dir = _write_state(tmp_path, payload)
        with pytest.raises(InvalidStateFileError, match="out of range"):
            load_app_state(state_dir=str(state_dir))

    def test_unknown_exercise_in_results_raises(self, tmp_path):
        payload = _make_valid_app_state_payload()
        payload["results"]["not_a_real_exercise"] = {
            "arrays": {},
            "metadata": {},
        }
        state_dir = _write_state(tmp_path, payload)
        with pytest.raises(InvalidStateFileError, match="not_a_real_exercise"):
            load_app_state(state_dir=str(state_dir))

    def test_top_level_not_object_raises(self, tmp_path):
        state_dir = tmp_path / ".movement_optimizer"
        state_dir.mkdir(parents=True)
        (state_dir / "last_state.json").write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(InvalidStateFileError, match="JSON object"):
            load_app_state(state_dir=str(state_dir))

    def test_results_block_missing_arrays_raises(self, tmp_path):
        payload = _make_valid_app_state_payload()
        payload["results"]["squat"] = {"metadata": {}}
        state_dir = _write_state(tmp_path, payload)
        with pytest.raises(InvalidStateFileError, match="arrays"):
            load_app_state(state_dir=str(state_dir))


class TestSchemaValidationSolution:
    """Issue #403: explicit schema validation on solution load."""

    def _valid_solution_payload(self) -> dict:
        result = make_test_result()
        return {
            "schema_version": SCHEMA_VERSION,
            "exercise_type": "squat",
            "bar_mass": 60.0,
            "body_params": {"body_mass": 80.0, "height": 1.80},
            "arrays": {
                k: getattr(result, k).tolist()
                for k in ("t", "q", "qd", "qdd", "torques", "power", "com", "bar")
            },
            "metadata": {
                "success": True,
                "cost": 42.5,
                "com_horizontal_range_cm": 3.2,
                "elapsed_s": 1.5,
                "n_evals": 100,
            },
        }

    def test_round_trip_solution_includes_schema_version(self, tmp_path):
        result = make_test_result()
        path = tmp_path / "solution.json"
        save_solution(str(path), result, {"body_mass": 80.0}, "squat", 60.0)

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["schema_version"] == SCHEMA_VERSION

        loaded = load_solution(str(path))
        assert loaded["schema_version"] == SCHEMA_VERSION
        assert loaded["exercise_type"] == "squat"

    def test_missing_schema_version_raises(self, tmp_path):
        payload = self._valid_solution_payload()
        del payload["schema_version"]
        path = tmp_path / "solution.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(InvalidStateFileError, match="schema_version"):
            load_solution(str(path))

    def test_unknown_schema_version_raises(self, tmp_path):
        payload = self._valid_solution_payload()
        payload["schema_version"] = 999
        path = tmp_path / "solution.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(InvalidStateFileError, match="unsupported schema_version"):
            load_solution(str(path))

    def test_missing_required_key_raises(self, tmp_path):
        payload = self._valid_solution_payload()
        del payload["bar_mass"]
        path = tmp_path / "solution.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(InvalidStateFileError, match="bar_mass"):
            load_solution(str(path))

    def test_wrong_type_for_bar_mass_raises(self, tmp_path):
        payload = self._valid_solution_payload()
        payload["bar_mass"] = "sixty"
        path = tmp_path / "solution.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(InvalidStateFileError, match="bar_mass"):
            load_solution(str(path))

    def test_out_of_range_bar_mass_raises(self, tmp_path):
        payload = self._valid_solution_payload()
        payload["bar_mass"] = -10.0
        path = tmp_path / "solution.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(InvalidStateFileError, match="out of range"):
            load_solution(str(path))

    def test_unknown_exercise_type_raises(self, tmp_path):
        payload = self._valid_solution_payload()
        payload["exercise_type"] = "moonwalk"
        path = tmp_path / "solution.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(InvalidStateFileError, match="moonwalk"):
            load_solution(str(path))

    def test_arrays_field_not_list_raises(self, tmp_path):
        payload = self._valid_solution_payload()
        payload["arrays"]["t"] = "not a list"
        path = tmp_path / "solution.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(InvalidStateFileError, match=r"arrays\.t"):
            load_solution(str(path))

    def test_metadata_missing_required_key_raises(self, tmp_path):
        payload = self._valid_solution_payload()
        del payload["metadata"]["cost"]
        path = tmp_path / "solution.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(InvalidStateFileError, match="cost"):
            load_solution(str(path))
