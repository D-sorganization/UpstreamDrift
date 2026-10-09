"""Generate a source-bound MuJoCo 3.8 activation-action diagnostic receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import crocoddyl
import mujoco as mj
import numpy as np

from src.engines.physics_engines.mujoco.python.native_activation_bundle import (
    build_native_activation_bundle,
    replay_native_activation_bundle,
)
from src.engines.physics_engines.mujoco.python.native_activation_manifold import (
    load_native_activation_provider,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _initial_state(model: mj.MjModel) -> np.ndarray:
    data = mj.MjData(model)
    data.qpos[:3] = [0.03, -0.02, 0.01]
    data.qpos[3:7] = [0.99, 0.0, 0.0, np.sqrt(1 - 0.99**2)]
    data.qpos[7] = 0.2
    data.qvel[:] = [0.02, -0.01, 0.03, 0.01, -0.02, 0.01, 0.1]
    data.act[0] = 0.3
    data.qacc_warmstart[:] = np.linspace(0.001, 0.007, model.nv)
    full = np.empty(mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION))
    mj.mj_getState(model, data, full, mj.mjtState.mjSTATE_INTEGRATION)
    return full


def _fresh_step(
    path: Path, full: np.ndarray, command: float, physical: np.ndarray | None = None
) -> np.ndarray:
    model = mj.MjModel.from_xml_path(str(path))
    model.opt.disableflags |= int(
        mj.mjtDisableBit.mjDSBL_AUTORESET | mj.mjtDisableBit.mjDSBL_WARMSTART
    )
    data = mj.MjData(model)
    mj.mj_setState(model, data, full, mj.mjtState.mjSTATE_INTEGRATION)
    if physical is not None:
        data.qpos[:] = physical[: model.nq]
        data.qvel[:] = physical[model.nq : model.nq + model.nv]
        data.act[:] = physical[model.nq + model.nv :]
    data.ctrl[:] = command
    mj.mj_step(model, data)
    result = np.empty_like(full)
    mj.mj_getState(model, data, result, mj.mjtState.mjSTATE_INTEGRATION)
    return result


def _directional_errors(
    path: Path, full: np.ndarray, command: float
) -> dict[str, float]:
    provider = load_native_activation_provider(path)
    physical = provider.project_physical(full)
    derivative = provider.linearize(physical, np.array([command]))
    dx = np.linspace(-0.2, 0.3, provider.state.ndx)
    du = 0.1
    expected = derivative.A @ dx + derivative.B[:, 0] * du
    result: dict[str, float] = {}
    for epsilon in (2e-5, 1e-5):
        minus = _fresh_step(
            path,
            full,
            command - epsilon * du,
            provider.state.integrate(physical, -epsilon * dx),
        )
        plus = _fresh_step(
            path,
            full,
            command + epsilon * du,
            provider.state.integrate(physical, epsilon * dx),
        )
        actual = provider.state.diff(
            provider.project_physical(minus), provider.project_physical(plus)
        ) / (2 * epsilon)
        result[f"epsilon_{epsilon:g}_max_abs"] = float(
            np.max(np.abs(actual - expected))
        )
    return result


def _history_independence(path: Path, full: np.ndarray) -> float:
    provider = load_native_activation_provider(path)
    model = provider.model
    data = mj.MjData(model)
    mj.mj_setState(model, data, full, mj.mjtState.mjSTATE_INTEGRATION)
    data.time = 0.125
    data.qacc_warmstart[:] = np.linspace(-0.7, 0.4, model.nv)
    changed = np.empty_like(full)
    mj.mj_getState(model, data, changed, mj.mjtState.mjSTATE_INTEGRATION)
    first = provider.project_physical(_fresh_step(path, full, 0.4))
    second = provider.project_physical(_fresh_step(path, changed, 0.4))
    return float(np.max(np.abs(first - second)))


def run_receipt(path: Path) -> dict[str, object]:
    """Run one bounded actual provider/replay diagnostic on the fixture."""
    if mj.__version__ != "3.8.0" or crocoddyl.__version__ != "3.2.1":
        raise ValueError("receipt requires MuJoCo 3.8.0 and Crocoddyl 3.2.1")
    repository = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    provider = load_native_activation_provider(path)
    full = _initial_state(provider.model)
    physical = provider.project_physical(full)
    derivative = provider.linearize(physical, np.array([0.4]))
    directions = _directional_errors(path, full, 0.4)
    history_delta = _history_independence(path, full)
    commands = np.array([[0.4], [0.5], [0.45]])
    bundle = build_native_activation_bundle(
        path, full, commands, experiment_id="f05e-native-muscle"
    )
    replay = replay_native_activation_bundle(bundle, path)
    independent = full.copy()
    largest_replay_delta = 0.0
    for index, command in enumerate(commands, 1):
        independent = _fresh_step(path, independent, float(command[0]))
        largest_replay_delta = max(
            largest_replay_delta,
            float(np.max(np.abs(replay.integration_states[index] - independent))),
        )
    return {
        "schema_version": "f05e-native-activation-diagnostic/1.0.0",
        "issue": 11958,
        "scope": "self-contained contact-free floating-root built-in muscle; no golfer qualification",
        "runtime": {"mujoco": mj.__version__, "crocoddyl": crocoddyl.__version__},
        "dimensions": {
            "nq": provider.model.nq,
            "nv": provider.model.nv,
            "na": provider.model.na,
            "nu": provider.model.nu,
            "physical_nx": provider.state.nx,
            "tangent_ndx": provider.state.ndx,
            "integration_state_size": len(full),
        },
        "identity": {
            "source_model_sha256": provider.identity.source_model_sha256,
            "loaded_native_model_sha256": provider.identity.loaded_native_model_sha256,
            "compiled_law_sha256": provider.identity.compiled_law_sha256,
            "provider_sha256": provider.identity.provider_sha256,
            "adapter_sha256": provider.identity.adapter_sha256,
            "ordered_input_channel_ids": provider.identity.ordered_input_channel_ids,
            "input_kind": provider.identity.input_kind,
            "source_closure": provider.identity.source_closure,
            "history_policy": provider.identity.history_policy,
        },
        "native_derivative": {
            "A_shape": derivative.A.shape,
            "B_shape": derivative.B.shape,
            "activation_to_hinge_velocity": float(derivative.A[-2, -1]),
            "command_to_activation": float(derivative.B[-1, 0]),
            "directional_errors": directions,
        },
        "warmstart_clock_physical_delta": history_delta,
        "native_replay": {
            "steps": len(commands),
            "time_seconds": tuple(float(t) for t in replay.time_seconds),
            "max_independent_full_state_delta": largest_replay_delta,
            "initial_state_sha256": bundle.integrity.initial_state_sha256,
            "applied_input_sha256": replay.applied_input_sha256,
            "policy_sha256": replay.policy_sha256,
            "input_channel_schema_sha256": bundle.integrity.input_channel_schema_sha256,
        },
        "diagnostic_kernel_wall_s": time.perf_counter() - started,
        "optimization_acceptance": "not_run; no F05 solver selection",
        "source_sha256": {
            source.relative_to(repository).as_posix(): _sha(source)
            for source in (
                path,
                Path(__file__).resolve(),
                repository
                / "src/engines/physics_engines/mujoco/python/native_activation_manifold.py",
                repository
                / "src/engines/physics_engines/mujoco/python/native_activation_bundle.py",
                repository
                / "tests/unit/motion_matching/test_native_activation_manifold.py",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_receipt(args.fixture.resolve(strict=True))
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
