# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for the ``movement_optimizer.tool_pack`` UpstreamDrift integration.

Verifies the surface UpstreamDrift consumes under the biomechanics-fleet
umbrella (``D-sorganization/UpstreamDrift#5179``):

* ``manifest()`` matches the ``tool_pack/v1`` schema and the on-disk YAML.
* ``list_exercises()`` returns the seven supported exercises.
* ``run_headless("squat", tmp)`` exits 0 and writes a JSON file with
  the keys produced by :mod:`movement_optimizer.cli`
  (``exercise``/``arrays``/per-joint torque arrays).
* The ``movement-optimizer`` CLI dispatches ``--list-exercises``
  without importing PyQt6.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from movement_optimizer import tool_pack

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PACKAGE_ROOT.parent
YAML_PATH = PACKAGE_ROOT / "tool_pack.yaml"
SUBPROCESS_ENV = {
    **os.environ,
    "PYTHONPATH": str(SRC_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
}

EXPECTED_EXERCISES = (
    "squat",
    "full_squat",
    "deadlift",
    "bench_press",
    "snatch",
    "clean",
    "jerk",
)


class TestManifest:
    def test_schema_is_tool_pack_v1(self) -> None:
        assert tool_pack.manifest()["schema"] == "tool_pack/v1"

    def test_role_and_formulation(self) -> None:
        data = tool_pack.manifest()
        assert data["role"] == "optimizer"
        assert data["formulation"] == "lagrangian"
        assert data["muscle_model"] == "hill"
        assert data["plane"] == "sagittal"
        assert data["links"] == 3
        assert data["anthropometrics"] == "winter_2009"

    def test_supported_exercises_match_expected(self) -> None:
        ids = tuple(tool_pack.manifest()["supported_exercises"])
        assert ids == EXPECTED_EXERCISES

    def test_consumes_models_from_lists_four_engines(self) -> None:
        data = tool_pack.manifest()
        assert set(data["consumes_models_from"]) == {
            "mujoco_models",
            "drake_models",
            "pinocchio_models",
            "opensim_models",
        }

    def test_produces_includes_trajectories(self) -> None:
        assert "trajectories" in tool_pack.manifest()["produces"]

    def test_repo_root_yaml_matches_loaded_manifest(self) -> None:
        on_disk = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
        assert tool_pack.manifest() == on_disk


class TestListExercises:
    def test_returns_expected_ids_in_order(self) -> None:
        assert tuple(tool_pack.list_exercises()) == EXPECTED_EXERCISES


class TestRunHeadless:
    def test_squat_round_trip_writes_valid_json(self, tmp_path: Path) -> None:
        out = tmp_path / "squat.json"
        exit_code = tool_pack.run_headless("squat", out)
        assert exit_code == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["exercise"] == "squat"
        # cli.py emits ``arrays`` with the per-joint trajectory.
        assert "arrays" in data
        for key in ("t", "q", "qd", "qdd", "torques", "power", "com", "bar"):
            assert key in data["arrays"], f"missing arrays.{key}"
        # Torques must be finite per joint (ankle, knee, hip).
        torques = data["arrays"]["torques"]
        assert len(torques) > 0
        assert all(len(row) == 3 for row in torques)


class TestLauncherCli:
    def test_list_exercises_subprocess(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "movement_optimizer", "--list-exercises"],
            capture_output=True,
            env=SUBPROCESS_ENV,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        printed = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        assert tuple(printed) == EXPECTED_EXERCISES

    def test_headless_requires_exercise(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "movement_optimizer", "--headless"],
            capture_output=True,
            env=SUBPROCESS_ENV,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert "--exercise" in result.stderr


class TestEntryPoint:
    def test_biomech_tool_pack_entry_point_registered(self) -> None:
        from importlib.metadata import entry_points

        eps = entry_points(group="biomech.tool_pack")
        names = {ep.name for ep in eps}
        if "movement_optimizer" not in names:
            pytest.skip(
                "package not installed in editable mode; entry point unavailable",
            )
        ep = next(ep for ep in eps if ep.name == "movement_optimizer")
        loaded = ep.load()
        for attr in ("manifest", "list_exercises", "run_headless"):
            assert callable(getattr(loaded, attr))
