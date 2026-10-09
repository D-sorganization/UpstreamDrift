"""Common admission through the authoritative Tools replay contract."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sidekick.lab.mocap import ExperimentReplayBundle


_MUJOCO_GLOBAL_CALLBACKS = (
    "control",
    "passive",
    "act_dyn",
    "act_gain",
    "act_bias",
    "sensor",
    "contactfilter",
    "time",
)


def require_no_global_mujoco_callbacks(mujoco: Any) -> None:
    """Reject process-global MuJoCo callbacks at a frozen replay boundary."""
    for callback_name in _MUJOCO_GLOBAL_CALLBACKS:
        getter = getattr(mujoco, f"get_mjcb_{callback_name}", None)
        if not callable(getter):
            raise ValueError("MuJoCo callback inspection API is incomplete")
        if getter() is not None:
            raise ValueError(
                "process-global MuJoCo callbacks are forbidden in independent replay"
            )


def validate_native_replay_bundle(
    bundle: ExperimentReplayBundle, contracts: Any
) -> ExperimentReplayBundle:
    """Revalidate integrity and required capabilities using Tools' authority."""
    validated = contracts.load_experiment_replay_bundle(
        contracts.dumps_experiment_replay_bundle(bundle)
    )
    if validated.blocking_capabilities:
        raise ValueError("required native replay capabilities are unavailable")
    return validated


def native_replay_admission_bytes() -> bytes:
    """Bind this executed admission helper into each adapter's provider hash."""
    return Path(__file__).read_bytes()


def native_replay_contract_types() -> Any:
    """Load the implemented Tools authority through the governed UD seam."""
    from src.shared.python._seam_redirect import load_pinned_tools_package

    mocap = load_pinned_tools_package("sidekick.lab.mocap")

    if not hasattr(mocap, "ExperimentReplayBundle"):
        raise RuntimeError("native replay requires the merged Tools T01 contract")
    return mocap


def validate_frozen_torque_history(
    times: NDArray[np.float64],
    values: NDArray[np.float64],
    step_size: float,
    limits: NDArray[np.float64],
) -> None:
    """Require a complete fixed-grid post-limit ZOH history and terminal sentinel."""
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("native timestep must be finite and positive")
    if (
        times.ndim != 1
        or times.size < 2
        or times[0] != 0
        or not np.isfinite(times).all()
        or not np.allclose(np.diff(times), step_size, atol=1e-12, rtol=0)
    ):
        raise ValueError("time grid must begin at zero and match native timestep")
    if values.shape != (times.size, limits.size) or not np.isfinite(values).all():
        raise ValueError("finite torque rows must match ordered native motors")
    if not np.array_equal(values[-1], values[-2]):
        raise ValueError("terminal ZOH sentinel must equal the last executed input")
    if not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("native motor effort limits must be finite and positive")
    if np.any(np.abs(values) > limits):
        raise ValueError("saved torque must already satisfy native effort limits")


def require_native_replay_equivalence(
    bundle: Any, expected: Any, contracts: Any
) -> None:
    """Reject changed model, restored state, channels or executed replay policy."""
    for field in ("model", "initial_state", "policy"):
        if getattr(bundle, field) != getattr(expected, field):
            raise ValueError("native model, state or executed policy identity differs")
    history = bundle.input_history
    if history.channels != expected.input_history.channels:
        raise ValueError("native channel identity differs")
    if (
        history.input_kind != contracts.ActuationInputKind.ACTUATOR_TORQUE
        or history.interpolation != contracts.InputInterpolation.ZERO_ORDER_HOLD
    ):
        raise ValueError("native replay requires held actuator torque")
