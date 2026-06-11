"""Optimize body kinematic targets to match a desired club state.

This is the first half of the two-model control approach:

1. Use a body-to-club surrogate to find a low-motion body kinematic state that
   produces the desired club position, velocity, and acceleration.
2. Feed that body target into the torque inverse step for the body dynamics
   surrogate.
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

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "runs" / "body_to_club_10_cpu" / "best_model.pt"
LOGGER = logging.getLogger(__name__)


def _load_json_values(path: Path) -> dict[str, float]:
    values = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): float(value) for key, value in values.items()}


def _fill_vector(columns: list[str], values: dict[str, float]) -> np.ndarray:
    missing = [column for column in columns if column not in values]
    if missing:
        raise ValueError(f"Missing values for columns: {missing}")
    return np.asarray([values[column] for column in columns], dtype=np.float32)


def optimize_body_kinematics(
    checkpoint_path: Path,
    reference_body_path: Path,
    desired_club_path: Path,
    output_path: Path,
    steps: int,
    learning_rate: float,
    motion_weight: float,
    acceleration_weight: float,
    device_name: str,
) -> None:
    checkpoint = torch.load(
        checkpoint_path, map_location=device_name, weights_only=True
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

    reference_body = _load_json_values(reference_body_path)
    desired_club = _load_json_values(desired_club_path)
    target_indices = [target_columns.index(column) for column in desired_club]

    reference_raw = _fill_vector(input_columns, reference_body)
    desired_raw = np.asarray(
        [desired_club[column] for column in desired_club], dtype=np.float32
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

    reference = torch.as_tensor(reference_raw, dtype=torch.float32, device=device_name)
    desired = torch.as_tensor(desired_raw, dtype=torch.float32, device=device_name)
    selected = torch.as_tensor(target_indices, dtype=torch.long, device=device_name)

    candidate = nn.Parameter(reference.detach().clone())
    optimizer = torch.optim.Adam([candidate], lr=learning_rate)
    acceleration_indices = [
        index
        for index, column in enumerate(input_columns)
        if "acceleration" in column.lower()
    ]

    best_loss = float("inf")
    best_candidate = candidate.detach().clone()
    history: list[dict[str, float]] = []

    for step in range(1, steps + 1):
        pred_scaled = model(((candidate - x_mean) / x_std).unsqueeze(0)).squeeze(0)
        pred = pred_scaled * y_std + y_mean
        matched = pred.index_select(0, selected)

        tracking_loss = torch.mean((matched - desired) ** 2)
        motion_loss = torch.mean((candidate - reference) ** 2)
        if acceleration_indices:
            accel_loss = torch.mean(candidate[acceleration_indices] ** 2)
        else:
            accel_loss = torch.zeros((), dtype=torch.float32, device=device_name)
        loss = (
            tracking_loss
            + motion_weight * motion_loss
            + acceleration_weight * accel_loss
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
                "motion_loss": float(motion_loss.detach().cpu()),
                "acceleration_loss": float(accel_loss.detach().cpu()),
            }
        )
        if loss_value < best_loss:
            best_loss = loss_value
            best_candidate = candidate.detach().clone()

    result = {
        "checkpoint": str(checkpoint_path),
        "reference_body": str(reference_body_path),
        "desired_club": str(desired_club_path),
        "optimized_body_kinematics": {
            column: float(value)
            for column, value in zip(
                input_columns, best_candidate.cpu().numpy(), strict=True
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
    parser.add_argument("--reference-body", type=Path, required=True)
    parser.add_argument("--desired-club", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "runs" / "two_stage_control" / "body_targets.json",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--motion-weight", type=float, default=1e-5)
    parser.add_argument("--acceleration-weight", type=float, default=1e-8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    optimize_body_kinematics(
        checkpoint_path=args.checkpoint,
        reference_body_path=args.reference_body,
        desired_club_path=args.desired_club,
        output_path=args.output,
        steps=args.steps,
        learning_rate=args.learning_rate,
        motion_weight=args.motion_weight,
        acceleration_weight=args.acceleration_weight,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
