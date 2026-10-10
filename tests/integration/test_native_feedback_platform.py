"""Public native matching/replay entry points coexist without requiring SDK imports.

This is an integration availability contract, never native execution evidence.
"""

import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("module_name", "operation"),
    [
        ("mujoco.python.native_torque_replay", "replay_native_torque_bundle"),
        ("drake.python.native_torque_replay", "replay_native_drake_torque_bundle"),
        (
            "pinocchio.python.native_torque_replay",
            "replay_native_pinocchio_torque_bundle",
        ),
        (
            "opensim.python.tour_matching.native_moco_runner",
            "prepare_native_moco",
        ),
        (
            "opensim.python.tour_matching.native_moco_replay",
            "replay_native_moco_bundle",
        ),
        (
            "myosuite.python.native_direct_model_replay",
            "replay_direct_model_actuator_commands",
        ),
    ],
)
def test_native_matching_and_replay_operations_coexist(
    module_name: str, operation: str
) -> None:
    module = importlib.import_module(f"src.engines.physics_engines.{module_name}")
    assert callable(getattr(module, operation))


def test_owned_simscape_replay_coexists_with_guarded_native_solver() -> None:
    simscape = importlib.import_module(
        "src.engines.Simscape_Multibody_Models.python.native_owned_replay"
    )
    solver = importlib.import_module("src.engines.myosuite_project_command_solve")
    assert callable(simscape.owned_simscape_replay_files)
    assert callable(solver.solve_native_command_plan)


def test_muscle_matching_cli_retains_prepare_and_numerical_seed_routes(
    tmp_path: Path,
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.cli import (
        build_parser,
    )

    parser = build_parser()
    prepared = parser.parse_args(
        [
            "moco-native",
            "--request",
            str(tmp_path / "request.json"),
            "--output-dir",
            str(tmp_path / "results"),
            "--prepare-only",
        ]
    )
    assert prepared.command == "moco-native" and prepared.prepare_only
    seed = parser.parse_args(
        [
            "moco-native-guess",
            "--model",
            str(tmp_path / "model.osim"),
            "--source-sha256",
            "a" * 64,
            "--trc",
            str(tmp_path / "capture.trc"),
            "--output-dir",
            str(tmp_path / "seed"),
        ]
    )
    assert seed.command == "moco-native-guess"
