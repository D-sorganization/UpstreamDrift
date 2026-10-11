"""Public native matching/replay entry points coexist without requiring SDK imports.

This is an integration availability contract, never native execution evidence.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from src.engines.physics_engines.drake.python.native_torque_replay import (
    replay_native_drake_torque_bundle,
)
from src.engines.physics_engines.mujoco.python.native_torque_replay import (
    replay_native_torque_bundle,
)
from src.engines.physics_engines.myosuite.python.native_direct_model_replay import (
    replay_direct_model_actuator_commands,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_moco_replay import (
    replay_native_moco_bundle,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_moco_runner import (
    prepare_native_moco,
)
from src.engines.physics_engines.pinocchio.python.native_torque_replay import (
    replay_native_pinocchio_torque_bundle,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "operation",
    [
        replay_native_torque_bundle,
        replay_native_drake_torque_bundle,
        replay_native_pinocchio_torque_bundle,
        prepare_native_moco,
        replay_native_moco_bundle,
        replay_direct_model_actuator_commands,
    ],
)
def test_native_matching_and_replay_operations_coexist(
    operation: Callable[..., object],
) -> None:
    assert callable(operation)


def test_owned_simscape_replay_coexists_with_guarded_native_solver() -> None:
    from src.engines.Simscape_Multibody_Models.python.native_owned_replay import (
        owned_simscape_replay_files,
    )
    from src.engines.myosuite_project_command_solve import solve_native_command_plan

    assert callable(owned_simscape_replay_files)
    assert callable(solve_native_command_plan)


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
