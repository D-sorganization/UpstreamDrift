"""Optimize a torque timeseries to match a desired club trajectory.

Cost modes
----------

The cost function is selectable via ``--cost-mode``. All modes minimise the
position-space tracking error of the clubface; later modes add additional
terms aligned with ``motion_matching/shared/COST_FUNCTION_SPEC.md``.

* ``--cost-mode position`` (legacy default, backward compatible)
    ``J = mean(|pred - target|^2) + w_e * ||tau - tau_0||^2 + w_s * ||dtau||^2``
    The same formula PR #3966 shipped with. Tracks position, velocity, and
    acceleration channels exposed by the surrogate; ignores orientation.

* ``--cost-mode position_orientation``
    Adds the quaternion geodesic orientation term defined in
    COST_FUNCTION_SPEC.md when the desired CSV exposes ``club_quat_w/x/y/z``
    columns:
        ``J += w_o * mean( (2 * acos(|q_sim . q_meas|))^2 )``
    Treats ``q`` and ``-q`` as identical (sign-invariant). The geodesic helper
    is reused from ``src.shared.python.motion_matching._geodesic``.

* ``--cost-mode full`` (canonical)
    Adds the total-work regularizer (also from COST_FUNCTION_SPEC.md):
        ``J += lambda * integral( sum_j |tau_j * omega_j| ) dt``
    via ``src.shared.python.motion_matching.cost.compute_total_work``. The
    legacy ``effort + smoothness`` regularizer is dropped in this mode in
    favour of the canonical work integral.

Selecting ``--regularizer-kind`` (advanced)
    Independently of ``--cost-mode``, the regularizer can be forced via
    ``--regularizer-kind {effort_smoothness, total_work}``. ``cost-mode full``
    sets it to ``total_work`` automatically; ``position`` and
    ``position_orientation`` default to ``effort_smoothness``.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.shared.python.motion_matching._geodesic import quaternion_geodesic_angles
from src.shared.python.motion_matching.cost import (
    compute_total_work as _shared_total_work,
)

LOGGER = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "runs" / "club_direct_10_cpu" / "best_model.pt"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "processed" / "optimized_club_torques.csv"

CostMode = Literal["position", "position_orientation", "full"]
RegularizerKind = Literal["effort_smoothness", "total_work"]

DEFAULT_ORIENTATION_WEIGHT = 0.1
DEFAULT_LAMBDA = 1e-4

TARGET_COLUMN_MAP = {
    "clubface_x": "ClubLogs_CHGlobalPosition_1",
    "clubface_y": "ClubLogs_CHGlobalPosition_2",
    "clubface_z": "ClubLogs_CHGlobalPosition_3",
    "clubface_vx": "ClubLogs_CHGlobalVelocity_1",
    "clubface_vy": "ClubLogs_CHGlobalVelocity_2",
    "clubface_vz": "ClubLogs_CHGlobalVelocity_3",
    "clubface_ax": "ClubLogs_CHGlobalAcceleration_1",
    "clubface_ay": "ClubLogs_CHGlobalAcceleration_2",
    "clubface_az": "ClubLogs_CHGlobalAcceleration_3",
}

QUAT_COLUMN_CANDIDATES: tuple[tuple[str, str, str, str], ...] = (
    ("club_quat_w", "club_quat_x", "club_quat_y", "club_quat_z"),
    ("clubface_qw", "clubface_qx", "clubface_qy", "clubface_qz"),
)


@dataclass(frozen=True)
class CostConfig:
    """Configuration for the per-step optimization cost.

    Attributes:
        mode: One of ``position``, ``position_orientation``, ``full``.
        regularizer_kind: ``effort_smoothness`` (legacy) or ``total_work``.
        effort_weight: Weight on ``||tau - tau_0||^2``. Used by
            ``effort_smoothness``.
        smoothness_weight: Weight on ``||dtau||^2``. Used by
            ``effort_smoothness``.
        orientation_weight: Weight on the quaternion-geodesic term.
        lambda_: Strength of the ``total_work`` regularizer.
    """

    mode: CostMode = "position"
    regularizer_kind: RegularizerKind = "effort_smoothness"
    effort_weight: float = 1e-6
    smoothness_weight: float = 1e-4
    orientation_weight: float = DEFAULT_ORIENTATION_WEIGHT
    lambda_: float = DEFAULT_LAMBDA


@dataclass(frozen=True)
class OptimizationOptions:
    steps: int
    learning_rate: float
    effort_weight: float
    smoothness_weight: float
    device_name: str
    cost_mode: CostMode = "position"
    regularizer_kind: RegularizerKind | None = None
    orientation_weight: float = DEFAULT_ORIENTATION_WEIGHT
    lambda_: float = DEFAULT_LAMBDA

    @classmethod
    def from_kwargs(
        cls,
        options: OptimizationOptions | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> OptimizationOptions:
        positional = (
            "steps",
            "learning_rate",
            "effort_weight",
            "smoothness_weight",
            "device_name",
            "cost_mode",
            "regularizer_kind",
            "orientation_weight",
            "lambda_",
        )
        if len(args) > len(positional):
            raise TypeError("optimize_sequence received too many positional options")
        for key, value in zip(positional, args, strict=False):
            if key in kwargs:
                raise TypeError(f"optimize_sequence got multiple values for {key}")
            kwargs[key] = value
        if options is not None and not kwargs:
            return options
        valid = set(cls.__dataclass_fields__)
        unknown = sorted(set(kwargs) - valid)
        if unknown:
            joined = ", ".join(unknown)
            raise TypeError(f"unexpected optimize_sequence option(s): {joined}")
        if options is None:
            required = {
                "steps",
                "learning_rate",
                "effort_weight",
                "smoothness_weight",
                "device_name",
            }
            missing = sorted(required - set(kwargs))
            if missing:
                joined = ", ".join(missing)
                raise TypeError(f"missing optimize_sequence option(s): {joined}")
            return cls(**kwargs)
        return dataclass_replace(options, **kwargs)


@dataclass(frozen=True)
class _OptimizationContext:
    cost: CostConfig
    model: nn.Module
    input_columns: list[str]
    target_columns: list[str]
    control_columns: list[str]
    control_indices: list[int]
    desired_time: np.ndarray
    x_base: torch.Tensor
    target: torch.Tensor
    selected: torch.Tensor
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor
    quat_target: torch.Tensor | None
    time_t: torch.Tensor


@dataclass(frozen=True)
class _StepLosses:
    loss: torch.Tensor
    tracking_loss: torch.Tensor
    effort_loss: torch.Tensor
    smoothness_loss: torch.Tensor
    orientation_loss: torch.Tensor
    regulariser_loss: torch.Tensor


def resolve_cost_config(
    mode: CostMode,
    *,
    regularizer_kind: RegularizerKind | None = None,
    effort_weight: float = 1e-6,
    smoothness_weight: float = 1e-4,
    orientation_weight: float = DEFAULT_ORIENTATION_WEIGHT,
    lambda_: float = DEFAULT_LAMBDA,
) -> CostConfig:
    """Map a ``cost_mode`` to a :class:`CostConfig` with sensible defaults.

    ``cost-mode full`` forces ``total_work`` if the user did not override.
    """
    if mode not in ("position", "position_orientation", "full"):
        raise ValueError(
            "cost_mode must be one of 'position', 'position_orientation', "
            f"'full'; got {mode!r}"
        )
    if regularizer_kind is None:
        regularizer_kind = "total_work" if mode == "full" else "effort_smoothness"
    if regularizer_kind not in ("effort_smoothness", "total_work"):
        raise ValueError(
            "regularizer_kind must be 'effort_smoothness' or 'total_work'; "
            f"got {regularizer_kind!r}"
        )
    return CostConfig(
        mode=mode,
        regularizer_kind=regularizer_kind,
        effort_weight=effort_weight,
        smoothness_weight=smoothness_weight,
        orientation_weight=orientation_weight,
        lambda_=lambda_,
    )


def find_quaternion_columns(frame: pd.DataFrame) -> tuple[str, str, str, str] | None:
    """Return the 4-tuple of quaternion column names if present, else None."""
    for names in QUAT_COLUMN_CANDIDATES:
        if all(name in frame.columns for name in names):
            return names
    return None


def quaternion_orientation_term(
    q_sim: torch.Tensor,
    q_meas: torch.Tensor,
) -> torch.Tensor:
    """Mean squared quaternion geodesic angle.

    ``mean( (2 * acos(clip(|q_sim . q_meas|, 0, 1)))^2 )``. Sign-invariant in
    each row (handles the ``q == -q`` ambiguity). Differentiable and
    autograd-compatible so it can be used inside the optimizer step.

    Args:
        q_sim:  ``(N, 4)`` quaternions in ``[w, x, y, z]`` order.
        q_meas: ``(N, 4)`` quaternions in ``[w, x, y, z]`` order.

    Raises:
        ValueError: If shapes mismatch or are not ``(N, 4)``.
    """
    if q_sim.shape != q_meas.shape:
        raise ValueError(
            f"shape mismatch: q_sim {tuple(q_sim.shape)} vs "
            f"q_meas {tuple(q_meas.shape)}"
        )
    if q_sim.ndim != 2 or q_sim.shape[1] != 4:
        raise ValueError(
            f"quaternions must have shape (N, 4); got {tuple(q_sim.shape)}"
        )
    dots = torch.abs(torch.sum(q_sim * q_meas, dim=1))
    dots = torch.clamp(dots, min=0.0, max=1.0)
    angles = 2.0 * torch.arccos(dots)
    return torch.mean(angles * angles)


def quaternion_orientation_term_numpy(
    q_sim: np.ndarray,
    q_meas: np.ndarray,
) -> float:
    """NumPy mirror of :func:`quaternion_orientation_term` (for tests).

    Reuses the shared ``_geodesic`` helper so behaviour matches
    ``compute_cost``.
    """
    angles = quaternion_geodesic_angles(q_sim, q_meas)
    return float(np.mean(angles * angles))


def total_work_regularizer(
    tau: torch.Tensor,
    omega: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    """Differentiable total-work regularizer.

    ``trapz(time, sum_j |tau_j * omega_j|)``. Mirrors
    :func:`src.shared.python.motion_matching.cost.compute_total_work` but
    returns a torch scalar so autograd can drive optimisation.
    """
    if tau.shape != omega.shape:
        raise ValueError(
            f"tau and omega shapes must match; got {tuple(tau.shape)} vs "
            f"{tuple(omega.shape)}"
        )
    if tau.shape[0] != time.shape[0]:
        raise ValueError(
            "tau/omega rows must match length(time); "
            f"got {tau.shape[0]} vs {time.shape[0]}"
        )
    integrand = torch.sum(torch.abs(tau * omega), dim=1)
    return torch.trapezoid(integrand, time)


def total_work_numpy(
    tau: np.ndarray,
    omega: np.ndarray,
    time: np.ndarray,
) -> float:
    """NumPy entry point that defers to the canonical shared helper."""
    from src.shared.python.motion_matching.cost import SimOutput

    sim_out = SimOutput(
        butt=np.zeros((len(time), 3), dtype=np.float64),
        clubhead=np.zeros((len(time), 3), dtype=np.float64),
        club_quat=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(time), 1)),
        time=np.asarray(time, dtype=np.float64),
        tau=np.asarray(tau, dtype=np.float64),
        omega=np.asarray(omega, dtype=np.float64),
    )
    return _shared_total_work(sim_out)


def _read_state_reference(path: Path, input_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in input_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Reference body CSV is missing input columns: {missing}")
    return frame


def _control_columns(input_columns: list[str]) -> list[str]:
    # Imported lazily so the module can be imported without the sibling script
    # available on sys.path (e.g. during unit tests of pure-Python helpers).
    from export_torque_polynomials import TORQUE_TO_POLYNOMIAL_BASE

    controls = []
    for column in input_columns:
        if column in TORQUE_TO_POLYNOMIAL_BASE:
            controls.append(column)
    return controls


def _interpolate_reference(
    reference: pd.DataFrame,
    desired_time: np.ndarray,
    input_columns: list[str],
) -> np.ndarray:
    if "time" not in reference.columns:
        values = reference[input_columns].to_numpy(dtype=np.float32)
        if len(values) == 1:
            return np.repeat(values, len(desired_time), axis=0)
        if len(values) != len(desired_time):
            raise ValueError(
                "Reference CSV without time must have one row or target row count"
            )
        return values

    ref_time = reference["time"].to_numpy(dtype=float)
    output = np.zeros((len(desired_time), len(input_columns)), dtype=np.float32)
    for idx, column in enumerate(input_columns):
        output[:, idx] = np.interp(
            desired_time,
            ref_time,
            reference[column].to_numpy(dtype=float),
        )
    return output


def _desired_club_targets(
    desired: pd.DataFrame,
    target_columns: list[str],
) -> tuple[np.ndarray, list[int]]:
    available: dict[str, str] = {
        model_column: model_column
        for model_column in target_columns
        if model_column in desired.columns
    }
    for source_column, model_column in TARGET_COLUMN_MAP.items():
        if source_column in desired.columns and model_column in target_columns:
            available.setdefault(model_column, source_column)
    if not available:
        raise ValueError("Desired club CSV has no recognizable club target columns")
    ordered_model_columns = list(available.keys())
    target_indices = [target_columns.index(column) for column in ordered_model_columns]
    target_values = desired[[available[column] for column in ordered_model_columns]]
    return target_values.to_numpy(dtype=np.float32), target_indices


def _desired_quaternions(desired: pd.DataFrame) -> np.ndarray | None:
    """Return ``(N, 4)`` quaternion array if the CSV exposes one."""
    cols = find_quaternion_columns(desired)
    if cols is None:
        return None
    return desired[list(cols)].to_numpy(dtype=np.float32)


def optimize_sequence(
    checkpoint_path: Path,
    desired_club_csv: Path,
    reference_body_csv: Path,
    output_csv: Path,
    *legacy_args: Any,
    options: OptimizationOptions | None = None,
    **kwargs: Any,
) -> None:
    opts = OptimizationOptions.from_kwargs(options, legacy_args, kwargs)
    context = _build_optimization_context(
        checkpoint_path,
        desired_club_csv,
        reference_body_csv,
        opts,
    )
    controls, best_loss, history = _run_optimization(context, opts)
    _write_optimization_outputs(
        output_csv,
        checkpoint_path,
        desired_club_csv,
        reference_body_csv,
        context,
        controls,
        best_loss,
        history,
    )


def _build_optimization_context(
    checkpoint_path: Path,
    desired_club_csv: Path,
    reference_body_csv: Path,
    options: OptimizationOptions,
) -> _OptimizationContext:
    from train_dynamics_surrogate import (
        DynamicsMLP,  # lazy: avoid sibling-import on unit tests
    )

    cost = resolve_cost_config(
        options.cost_mode,
        regularizer_kind=options.regularizer_kind,
        effort_weight=options.effort_weight,
        smoothness_weight=options.smoothness_weight,
        orientation_weight=options.orientation_weight,
        lambda_=options.lambda_,
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=options.device_name, weights_only=False
    )
    input_columns = list(checkpoint["input_columns"])
    target_columns = list(checkpoint["target_columns"])
    control_columns = _control_columns(input_columns)
    if not control_columns:
        raise ValueError(
            "Checkpoint input columns do not include mapped torque controls"
        )

    model = DynamicsMLP(
        input_dim=len(input_columns),
        output_dim=len(target_columns),
        hidden_sizes=list(checkpoint["config"]["hidden_sizes"]),
    ).to(options.device_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    desired = pd.read_csv(desired_club_csv)
    if "time" not in desired.columns:
        raise ValueError("Desired club CSV must include a time column")
    desired_time = desired["time"].to_numpy(dtype=float)
    desired_quat = _desired_quaternions(desired)
    if cost.mode != "position" and desired_quat is None:
        LOGGER.warning(
            "cost_mode=%s requested but desired CSV has no quaternion columns; "
            "orientation term will be skipped",
            cost.mode,
        )

    reference = _read_state_reference(reference_body_csv, input_columns)
    x_raw = _interpolate_reference(reference, desired_time, input_columns)
    target_raw, target_indices = _desired_club_targets(desired, target_columns)
    control_indices = [input_columns.index(column) for column in control_columns]
    x_mean = torch.as_tensor(
        checkpoint["x_mean"], dtype=torch.float32, device=options.device_name
    )
    x_std = torch.as_tensor(
        checkpoint["x_std"], dtype=torch.float32, device=options.device_name
    )
    y_mean = torch.as_tensor(
        checkpoint["y_mean"], dtype=torch.float32, device=options.device_name
    )
    y_std = torch.as_tensor(
        checkpoint["y_std"], dtype=torch.float32, device=options.device_name
    )
    x_base = torch.as_tensor(x_raw, dtype=torch.float32, device=options.device_name)
    target = torch.as_tensor(
        target_raw, dtype=torch.float32, device=options.device_name
    )
    selected = torch.as_tensor(
        target_indices, dtype=torch.long, device=options.device_name
    )
    time_t = torch.as_tensor(
        desired_time, dtype=torch.float32, device=options.device_name
    )
    quat_target_t = (
        torch.as_tensor(desired_quat, dtype=torch.float32, device=options.device_name)
        if desired_quat is not None
        else None
    )
    return _OptimizationContext(
        cost=cost,
        model=model,
        input_columns=input_columns,
        target_columns=target_columns,
        control_columns=control_columns,
        control_indices=control_indices,
        desired_time=desired_time,
        x_base=x_base,
        target=target,
        selected=selected,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        quat_target=quat_target_t,
        time_t=time_t,
    )


def _run_optimization(
    context: _OptimizationContext,
    options: OptimizationOptions,
) -> tuple[torch.Tensor, float, list[dict[str, float]]]:
    controls, initial_controls = _initial_controls(context)
    optimizer = torch.optim.Adam([controls], lr=options.learning_rate)
    best_loss = float("inf")
    best_controls = controls.detach().clone()
    history: list[dict[str, float]] = []

    for step in range(1, options.steps + 1):
        losses = _optimization_step_losses(context, controls, initial_controls)
        optimizer.zero_grad(set_to_none=True)
        losses.loss.backward()
        optimizer.step()
        loss_value = float(losses.loss.detach().cpu())
        history.append(_step_history(step, loss_value, losses))
        if loss_value < best_loss:
            best_loss = loss_value
            best_controls = controls.detach().clone()
    return best_controls, best_loss, history


def _initial_controls(
    context: _OptimizationContext,
) -> tuple[nn.Parameter, torch.Tensor]:
    initial_controls = context.x_base[:, context.control_indices].detach().clone()
    controls = nn.Parameter(initial_controls.clone())
    return controls, initial_controls


def _optimization_step_losses(
    context: _OptimizationContext,
    controls: torch.Tensor,
    initial_controls: torch.Tensor,
) -> _StepLosses:
    candidate = context.x_base.clone()
    candidate[:, context.control_indices] = controls
    pred_scaled = context.model((candidate - context.x_mean) / context.x_std)
    pred = pred_scaled * context.y_std + context.y_mean
    matched = pred.index_select(1, context.selected)
    tracking_loss = torch.mean((matched - context.target) ** 2)

    orientation_loss = torch.zeros(
        (), dtype=torch.float32, device=context.x_base.device
    )
    if context.cost.regularizer_kind == "total_work":
        regulariser_loss = _total_work_loss(controls, context.time_t)
        effort_loss = torch.zeros((), dtype=torch.float32, device=context.x_base.device)
        smoothness_loss = torch.zeros(
            (), dtype=torch.float32, device=context.x_base.device
        )
    else:
        effort_loss = torch.mean((controls - initial_controls) ** 2)
        smoothness_loss = _smoothness_loss(controls)
        regulariser_loss = torch.zeros(
            (), dtype=torch.float32, device=context.x_base.device
        )

    loss = tracking_loss + context.cost.orientation_weight * orientation_loss
    if context.cost.regularizer_kind == "total_work":
        loss = loss + context.cost.lambda_ * regulariser_loss
    else:
        loss = (
            loss
            + context.cost.effort_weight * effort_loss
            + context.cost.smoothness_weight * smoothness_loss
        )
    return _StepLosses(
        loss=loss,
        tracking_loss=tracking_loss,
        effort_loss=effort_loss,
        smoothness_loss=smoothness_loss,
        orientation_loss=orientation_loss,
        regulariser_loss=regulariser_loss,
    )


def _total_work_loss(controls: torch.Tensor, time_t: torch.Tensor) -> torch.Tensor:
    if len(controls) <= 1:
        return torch.zeros((), dtype=torch.float32, device=controls.device)
    dt = (time_t[1:] - time_t[:-1]).clamp(min=1e-9)
    omega = (controls[1:] - controls[:-1]) / dt.unsqueeze(1)
    tau_mid = 0.5 * (controls[1:] + controls[:-1])
    t_mid = 0.5 * (time_t[1:] + time_t[:-1])
    return total_work_regularizer(tau_mid, omega, t_mid)


def _smoothness_loss(controls: torch.Tensor) -> torch.Tensor:
    if len(controls) <= 1:
        return torch.zeros((), dtype=torch.float32, device=controls.device)
    return torch.mean((controls[1:] - controls[:-1]) ** 2)


def _step_history(
    step: int,
    loss_value: float,
    losses: _StepLosses,
) -> dict[str, float]:
    return {
        "step": step,
        "loss": loss_value,
        "tracking_loss": float(losses.tracking_loss.detach().cpu()),
        "effort_loss": float(losses.effort_loss.detach().cpu()),
        "smoothness_loss": float(losses.smoothness_loss.detach().cpu()),
        "orientation_loss": float(losses.orientation_loss.detach().cpu()),
        "regulariser_loss": float(losses.regulariser_loss.detach().cpu()),
    }


def _write_optimization_outputs(
    output_csv: Path,
    checkpoint_path: Path,
    desired_club_csv: Path,
    reference_body_csv: Path,
    context: _OptimizationContext,
    best_controls: torch.Tensor,
    best_loss: float,
    history: list[dict[str, float]],
) -> None:
    output = pd.DataFrame({"time": context.desired_time})
    for idx, column in enumerate(context.control_columns):
        output[column] = best_controls[:, idx].detach().cpu().numpy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    output_csv.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "desired_club_csv": str(desired_club_csv),
                "reference_body_csv": str(reference_body_csv),
                "output_csv": str(output_csv),
                "rows": int(len(output)),
                "control_columns": context.control_columns,
                "best_loss": best_loss,
                "cost_mode": context.cost.mode,
                "regularizer_kind": context.cost.regularizer_kind,
                "orientation_weight": context.cost.orientation_weight,
                "lambda_": context.cost.lambda_,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Wrote optimized torque sequence to %s", output_csv)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the argparse parser. Exposed for testability."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--desired-club-csv", type=Path, required=True)
    parser.add_argument("--reference-body-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--effort-weight", type=float, default=1e-6)
    parser.add_argument("--smoothness-weight", type=float, default=1e-4)
    parser.add_argument(
        "--orientation-weight",
        type=float,
        default=DEFAULT_ORIENTATION_WEIGHT,
        help="Weight on the quaternion-geodesic orientation term.",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=DEFAULT_LAMBDA,
        help="Strength of the total-work regularizer.",
    )
    parser.add_argument(
        "--cost-mode",
        choices=("position", "position_orientation", "full"),
        default="position",
        help=(
            "Which cost terms to include. 'position' (legacy default) tracks "
            "position+vel+accel only. 'position_orientation' adds the "
            "quaternion-geodesic orientation term. 'full' additionally swaps "
            "the legacy effort+smoothness regularizer for the canonical "
            "total-work integral."
        ),
    )
    parser.add_argument(
        "--regularizer-kind",
        choices=("effort_smoothness", "total_work"),
        default=None,
        help=(
            "Override the regularizer choice. Defaults: 'effort_smoothness' "
            "for cost-mode position/position_orientation; 'total_work' for "
            "cost-mode full."
        ),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    optimize_sequence(
        checkpoint_path=args.checkpoint,
        desired_club_csv=args.desired_club_csv,
        reference_body_csv=args.reference_body_csv,
        output_csv=args.output_csv,
        steps=args.steps,
        learning_rate=args.learning_rate,
        effort_weight=args.effort_weight,
        smoothness_weight=args.smoothness_weight,
        device_name=args.device,
        cost_mode=args.cost_mode,
        regularizer_kind=args.regularizer_kind,
        orientation_weight=args.orientation_weight,
        lambda_=args.lambda_,
    )


if __name__ == "__main__":
    main()
