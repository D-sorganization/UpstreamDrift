"""CLI driver for the dynamics-consistent two-stage trajectory optimizer.

See :mod:`two_stage_optimizer` for the underlying algorithms. This script wires
the stages together against trained PyTorch surrogate checkpoints and writes
the optimized body-target CSV, torque CSV, and a JSON history summary.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from two_stage_optimizer import run_two_stage

LOGGER = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the two-stage runner.

    :param argv: optional argument vector for testing; defaults to ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--body-checkpoint",
        type=Path,
        required=True,
        help="Path to the body-to-club surrogate checkpoint (.pt).",
    )
    parser.add_argument(
        "--dynamics-checkpoint",
        type=Path,
        required=True,
        help="Path to the body-dynamics surrogate checkpoint (.pt).",
    )
    parser.add_argument(
        "--desired-club-csv",
        type=Path,
        required=True,
        help="CSV containing the desired clubface trajectory (must include 'time').",
    )
    parser.add_argument(
        "--reference-body-csv",
        type=Path,
        required=True,
        help="CSV containing the reference (rest) body kinematic trajectory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "runs" / "two_stage_pipeline",
        help="Directory for body_targets.csv, torques.csv, history.json.",
    )
    parser.add_argument("--stage-a-steps", type=int, default=300)
    parser.add_argument("--stage-b-steps", type=int, default=300)
    parser.add_argument("--motion-weight", type=float, default=1e-3)
    parser.add_argument("--smooth-weight", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--dt", type=float, default=1.0 / 240.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args(argv)


def _build_surrogate_callable(  # pragma: no cover - exercised in integration only
    checkpoint_path: Path, device: str
) -> tuple[torch.nn.Module, list[str], list[str]]:
    from train_dynamics_surrogate import DynamicsMLP

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    input_columns = list(checkpoint["input_columns"])
    target_columns = list(checkpoint["target_columns"])
    model = DynamicsMLP(
        input_dim=len(input_columns),
        output_dim=len(target_columns),
        hidden_sizes=list(checkpoint["config"]["hidden_sizes"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, input_columns, target_columns


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI glue
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    desired = pd.read_csv(args.desired_club_csv)
    if "time" not in desired.columns:
        raise ValueError("desired-club-csv must include a 'time' column")
    reference = pd.read_csv(args.reference_body_csv)

    body_model, body_inputs, body_targets = _build_surrogate_callable(
        args.body_checkpoint, args.device
    )
    dyn_model, dyn_inputs, dyn_targets = _build_surrogate_callable(
        args.dynamics_checkpoint, args.device
    )

    n_steps = len(desired)
    n_joints = len(body_inputs)
    q_rest = torch.as_tensor(
        (
            reference[body_inputs].to_numpy(dtype=np.float32)
            if all(c in reference.columns for c in body_inputs)
            else np.zeros((n_steps, n_joints), dtype=np.float32)
        ),
        device=args.device,
    )
    if q_rest.shape[0] != n_steps:
        q_rest = q_rest[:1].expand(n_steps, n_joints).contiguous()

    # Best-effort club target columns
    club_cols = [c for c in body_targets if c in desired.columns]
    if not club_cols:
        raise ValueError("No body-to-club target columns found in desired CSV")
    club = torch.as_tensor(
        desired[club_cols].to_numpy(dtype=np.float32), device=args.device
    )

    def fk(q: torch.Tensor) -> torch.Tensor:
        idx = [body_targets.index(c) for c in club_cols]
        out = body_model(q)
        return out.index_select(1, torch.as_tensor(idx, device=q.device))

    def dyn(q: torch.Tensor, q_dot: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        x = torch.cat([q, q_dot, tau], dim=1)
        out = dyn_model(x[:, : len(dyn_inputs)])
        return out[:, : q.shape[1]]

    result = run_two_stage(
        forward_kinematics=fk,
        surrogate=dyn,
        clubface_target=club,
        q_rest=q_rest,
        dt=args.dt,
        stage_a_steps=args.stage_a_steps,
        stage_b_steps=args.stage_b_steps,
        motion_weight=args.motion_weight,
        smooth_weight=args.smooth_weight,
        learning_rate=args.learning_rate,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    body_df = pd.DataFrame(result.stage_a.q.cpu().numpy(), columns=body_inputs)
    body_df.insert(0, "time", desired["time"].to_numpy())
    body_df.to_csv(args.output_dir / "body_targets.csv", index=False)

    tau_cols = [f"tau_{i}" for i in range(result.stage_b.tau.shape[1])]
    tau_df = pd.DataFrame(result.stage_b.tau.cpu().numpy(), columns=tau_cols)
    tau_df.insert(0, "time", desired["time"].to_numpy())
    tau_df.to_csv(args.output_dir / "torques.csv", index=False)

    (args.output_dir / "history.json").write_text(
        json.dumps(
            {
                "stage_a": {
                    "final_loss": result.stage_a.final_loss,
                    "history": result.stage_a.history,
                },
                "stage_b": {
                    "final_loss": result.stage_b.final_loss,
                    "history": result.stage_b.history,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Two-stage pipeline complete; outputs at %s", args.output_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
