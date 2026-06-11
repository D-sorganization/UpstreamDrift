"""Optimize joint torques against a trained forward dynamics surrogate.

This is the first inverse-control utility. It does not train a second inverse
network. Instead, it loads a trained forward surrogate and adjusts only the
torque/force input columns so the predicted kinematic outputs match a desired
state for one independent sample.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from train_dynamics_surrogate import DynamicsMLP

from src.shared.python.motion_matching._checkpoint_artifacts import (
    load_surrogate_checkpoint,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "runs" / "inverse_ready_10_cpu" / "best_model.pt"
LOGGER = logging.getLogger(__name__)


def _load_json_values(path: Path) -> dict[str, float]:
    values = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): float(value) for key, value in values.items()}


def _control_columns(input_columns: list[str]) -> list[str]:
    controls = []
    for column in input_columns:
        lower = column.lower()
        if "torque" in lower or "force" in lower:
            controls.append(column)
    return controls


def _fill_input_vector(
    input_columns: list[str],
    current_state: dict[str, float],
    initial_controls: dict[str, float],
) -> np.ndarray:
    x = np.zeros(len(input_columns), dtype=np.float32)
    for i, column in enumerate(input_columns):
        if column in initial_controls:
            x[i] = initial_controls[column]
        elif column in current_state:
            x[i] = current_state[column]
        else:
            raise ValueError(f"Missing current-state value for input column: {column}")
    return x


def optimize_torques(
    checkpoint_path: Path,
    current_state_path: Path,
    desired_state_path: Path,
    output_path: Path,
    initial_controls_path: Path | None,
    steps: int,
    learning_rate: float,
    effort_weight: float,
    device_name: str,
) -> None:
    checkpoint = load_surrogate_checkpoint(
        checkpoint_path,
        map_location=device_name,
        artifact_name="Simscape dynamics surrogate checkpoint",
    )
    input_columns = list(checkpoint["input_columns"])
    target_columns = list(checkpoint["target_columns"])
    hidden_sizes = list(checkpoint["config"]["hidden_sizes"])

    model = DynamicsMLP(
        input_dim=len(input_columns),
        output_dim=len(target_columns),
        hidden_sizes=hidden_sizes,
    ).to(device_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    current_state = _load_json_values(current_state_path)
    desired_state = _load_json_values(desired_state_path)
    initial_controls = (
        _load_json_values(initial_controls_path) if initial_controls_path else {}
    )

    control_columns = _control_columns(input_columns)
    control_indices = [input_columns.index(column) for column in control_columns]
    target_indices = [target_columns.index(column) for column in desired_state]

    x_raw = _fill_input_vector(input_columns, current_state, initial_controls)
    target_raw = np.asarray(
        [desired_state[column] for column in desired_state], dtype=np.float32
    )

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

    initial_control_values = x_base[control_indices].detach().clone()
    controls = nn.Parameter(initial_control_values.clone())
    optimizer = torch.optim.Adam([controls], lr=learning_rate)

    best_loss = float("inf")
    best_controls = controls.detach().clone()
    history: list[dict[str, float]] = []

    for step in range(1, steps + 1):
        candidate = x_base.clone()
        candidate[control_indices] = controls
        candidate_scaled = (candidate - x_mean) / x_std
        pred_scaled = model(candidate_scaled.unsqueeze(0)).squeeze(0)
        pred = pred_scaled * y_std + y_mean
        matched = pred.index_select(0, selected)

        tracking_loss = torch.mean((matched - target) ** 2)
        effort_loss = torch.mean((controls - initial_control_values) ** 2)
        loss = tracking_loss + effort_weight * effort_loss

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
            }
        )
        if loss_value < best_loss:
            best_loss = loss_value
            best_controls = controls.detach().clone()

    result = {
        "checkpoint": str(checkpoint_path),
        "current_state": str(current_state_path),
        "desired_state": str(desired_state_path),
        "optimized_controls": {
            column: float(value)
            for column, value in zip(
                control_columns, best_controls.cpu().numpy(), strict=True
            )
        },
        "best_loss": best_loss,
        "history": history,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOGGER.info("%s", json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--current-state", type=Path, required=True)
    parser.add_argument("--desired-state", type=Path, required=True)
    parser.add_argument("--initial-controls", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "runs" / "inverse_control" / "torques.json",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--effort-weight", type=float, default=1e-6)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    optimize_torques(
        checkpoint_path=args.checkpoint,
        current_state_path=args.current_state,
        desired_state_path=args.desired_state,
        output_path=args.output,
        initial_controls_path=args.initial_controls,
        steps=args.steps,
        learning_rate=args.learning_rate,
        effort_weight=args.effort_weight,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
