"""Optimize a torque timeseries to match a desired club trajectory."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from train_dynamics_surrogate import DynamicsMLP

from src.shared.python.motion_matching.control_names import TORQUE_TO_POLYNOMIAL_BASE

LOGGER = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "runs" / "club_direct_10_cpu" / "best_model.pt"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "processed" / "optimized_club_torques.csv"

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


def _read_state_reference(path: Path, input_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in input_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Reference body CSV is missing input columns: {missing}")
    return frame


def _control_columns(input_columns: list[str]) -> list[str]:
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
    available: dict[str, str] = {}
    for model_column in target_columns:
        if model_column in desired.columns:
            available[model_column] = model_column
    for source_column, model_column in TARGET_COLUMN_MAP.items():
        if source_column in desired.columns and model_column in target_columns:
            available.setdefault(model_column, source_column)
    if not available:
        raise ValueError("Desired club CSV has no recognizable club target columns")
    ordered_model_columns = list(available.keys())
    target_indices = [target_columns.index(column) for column in ordered_model_columns]
    target_values = desired[[available[column] for column in ordered_model_columns]]
    return target_values.to_numpy(dtype=np.float32), target_indices


def optimize_sequence(
    checkpoint_path: Path,
    desired_club_csv: Path,
    reference_body_csv: Path,
    output_csv: Path,
    steps: int,
    learning_rate: float,
    effort_weight: float,
    smoothness_weight: float,
    device_name: str,
) -> None:
    checkpoint = torch.load(
        checkpoint_path, map_location=device_name, weights_only=False
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
    ).to(device_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    desired = pd.read_csv(desired_club_csv)
    if "time" not in desired.columns:
        raise ValueError("Desired club CSV must include a time column")
    desired_time = desired["time"].to_numpy(dtype=float)

    reference = _read_state_reference(reference_body_csv, input_columns)
    x_raw = _interpolate_reference(reference, desired_time, input_columns)
    target_raw, target_indices = _desired_club_targets(desired, target_columns)

    control_indices = [input_columns.index(column) for column in control_columns]
    x_mean = torch.as_tensor(
        checkpoint["x_mean"], dtype=torch.float32, device=device_name
    )
    x_std = torch.as_tensor(
        checkpoint["x_std"], dtype=torch.float32, device=device_name
    )
    y_mean = torch.as_tensor(
        checkpoint["y_mean"], dtype=torch.float32, device=device_name
    )
    y_std = torch.as_tensor(
        checkpoint["y_std"], dtype=torch.float32, device=device_name
    )

    x_base = torch.as_tensor(x_raw, dtype=torch.float32, device=device_name)
    target = torch.as_tensor(target_raw, dtype=torch.float32, device=device_name)
    selected = torch.as_tensor(target_indices, dtype=torch.long, device=device_name)
    initial_controls = x_base[:, control_indices].detach().clone()
    controls = nn.Parameter(initial_controls.clone())
    optimizer = torch.optim.Adam([controls], lr=learning_rate)

    best_loss = float("inf")
    best_controls = controls.detach().clone()
    history: list[dict[str, float]] = []

    for step in range(1, steps + 1):
        candidate = x_base.clone()
        candidate[:, control_indices] = controls
        pred_scaled = model((candidate - x_mean) / x_std)
        pred = pred_scaled * y_std + y_mean
        matched = pred.index_select(1, selected)

        tracking_loss = torch.mean((matched - target) ** 2)
        effort_loss = torch.mean((controls - initial_controls) ** 2)
        if len(controls) > 1:
            smoothness_loss = torch.mean((controls[1:] - controls[:-1]) ** 2)
        else:
            smoothness_loss = torch.zeros((), dtype=torch.float32, device=device_name)
        loss = (
            tracking_loss
            + effort_weight * effort_loss
            + smoothness_weight * smoothness_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        history.append(
            {
                "step": step,
                "loss": loss_value,
                "tracking_loss": float(tracking_loss.detach().cpu()),
                "effort_loss": float(effort_loss.detach().cpu()),
                "smoothness_loss": float(smoothness_loss.detach().cpu()),
            }
        )
        if loss_value < best_loss:
            best_loss = loss_value
            best_controls = controls.detach().clone()

    output = pd.DataFrame({"time": desired_time})
    for idx, column in enumerate(control_columns):
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
                "control_columns": control_columns,
                "best_loss": best_loss,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Wrote optimized torque sequence to %s", output_csv)


def parse_args() -> argparse.Namespace:
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
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


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
    )


if __name__ == "__main__":
    main()
