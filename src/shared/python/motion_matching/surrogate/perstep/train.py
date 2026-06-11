"""Train a PyTorch MLP that maps golf swing state/load samples to kinematics."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except OSError as exc:  # Broken local torch installs fail this way on import.
    raise SystemExit(
        "PyTorch failed to import. Use Python 3.12 with a valid Torch install, "
        "or create a CUDA venv as documented in README.md."
    ) from exc

from src.shared.python.motion_matching._checkpoint_artifacts import load_checkpoint_dict

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "data" / "processed" / "golf_inverse_ready.parquet"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "runs" / "baseline_mlp"
LOGGER = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    dataset: str
    output_dir: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_sizes: list[int]
    validation_fraction: float
    test_fraction: float
    seed: int
    device: str
    use_amp: bool


@dataclass(frozen=True)
class _PreparedTrainingData:
    x_raw: np.ndarray
    y_raw: np.ndarray
    x_scaled: np.ndarray
    y_scaled: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    input_columns: list[str]
    target_columns: list[str]


@dataclass(frozen=True)
class _RuntimeObjects:
    model: DynamicsMLP
    optimizer: torch.optim.Optimizer
    loss_fn: nn.MSELoss
    scaler: torch.amp.GradScaler
    loader: DataLoader
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    device: torch.device
    pin_memory: bool
    best_path: Path


class DynamicsMLP(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_sizes: list[int]
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.SiLU())
            prev = hidden
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_column_metadata(parquet_path: Path) -> tuple[list[str], list[str]]:
    metadata = pq.ParquetFile(parquet_path).schema_arrow.metadata or {}
    input_columns = json.loads(metadata[b"input_columns"].decode("utf-8"))
    target_columns = json.loads(metadata[b"target_columns"].decode("utf-8"))
    return input_columns, target_columns


def _load_arrays(
    parquet_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    input_columns, target_columns = _load_column_metadata(parquet_path)
    read_columns = list(dict.fromkeys([*input_columns, *target_columns]))
    table = pq.read_table(parquet_path, columns=read_columns)
    frame = table.to_pandas()

    x = frame[input_columns].to_numpy(dtype=np.float32, copy=True)
    y = frame[target_columns].to_numpy(dtype=np.float32, copy=True)

    finite_mask = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    dropped = int((~finite_mask).sum())
    if dropped:
        LOGGER.info("Dropping %s rows with non-finite values", dropped)
    return x[finite_mask], y[finite_mask], input_columns, target_columns


def _standardize(
    train: np.ndarray,
    values: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, eps)
    return (values - mean) / std, mean, std


def _make_splits(
    n: int,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_test = int(n * test_fraction)
    n_val = int(n * validation_fraction)
    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]
    return train_idx, val_idx, test_idx


def _rmse_by_column(
    pred: np.ndarray, target: np.ndarray, columns: list[str]
) -> dict[str, float]:
    diff = pred - target
    # ⚡ Bolt: np.einsum is ~30-40% faster than np.mean((pred - target)**2, axis=0) and avoids intermediate squared array allocations
    rmse = np.sqrt(np.einsum("ij,ij->j", diff, diff) / diff.shape[0])
    return {column: float(value) for column, value in zip(columns, rmse, strict=True)}


def _r2_by_column(
    pred: np.ndarray, target: np.ndarray, columns: list[str]
) -> dict[str, float]:
    diff = target - pred
    # ⚡ Bolt: np.einsum is ~30-40% faster than np.sum((target - pred)**2, axis=0)
    residual = np.einsum("ij,ij->j", diff, diff)
    centered = target - np.mean(target, axis=0)
    total = np.einsum("ij,ij->j", centered, centered)
    r2 = np.where(total > 0, 1.0 - residual / total, np.nan)
    return {column: float(value) for column, value in zip(columns, r2, strict=True)}


def compute_phase_stratified_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    time_values: np.ndarray,
    phase_breakpoints: dict[str, float] | None = None,
) -> dict[str, float]:
    """Optional per-phase RMSE report.

    Time is *not* a model input. ``time_values`` is metadata only, used
    here to bucket evaluation rows into swing phases for residual reporting.
    Returns an empty dict when ``time_values`` has no positive range.
    """
    try:
        from .surrogate_validation import evaluate_per_phase, phase_stratified_split
    except ImportError:
        LOGGER.warning("surrogate_validation module not found. Skipping phase metrics.")
        return {}

    if time_values.size == 0:
        return {}
    if float(np.max(time_values)) <= float(np.min(time_values)):
        return {}
    masks = phase_stratified_split(
        {"t": np.asarray(time_values)},
        time_col="t",
        phase_breakpoints=phase_breakpoints,
    )
    return evaluate_per_phase(pred, target, masks)


def _normalized_rmse_by_column(
    pred: np.ndarray,
    target: np.ndarray,
    columns: list[str],
) -> dict[str, float]:
    diff = pred - target
    # ⚡ Bolt: np.einsum is ~30-40% faster than np.mean((pred - target)**2, axis=0)
    rmse = np.sqrt(np.einsum("ij,ij->j", diff, diff) / diff.shape[0])
    std = np.std(target, axis=0)
    normalized = np.where(std > 0, rmse / std, np.nan)
    return {
        column: float(value) for column, value in zip(columns, normalized, strict=True)
    }


def train(config: TrainConfig) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_training(config.seed)
    data = _prepare_training_data(config)
    runtime = _build_runtime(config, data, output_dir)
    history, best_val = _fit_model(config, data, runtime)
    metrics = _evaluate_best_model(config, data, runtime, best_val)
    _write_training_outputs(output_dir, history, metrics)


def _seed_training(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _prepare_training_data(config: TrainConfig) -> _PreparedTrainingData:
    x_raw, y_raw, input_columns, target_columns = _load_arrays(Path(config.dataset))
    train_idx, val_idx, test_idx = _make_splits(
        len(x_raw),
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )

    x_train_raw = x_raw[train_idx]
    y_train_raw = y_raw[train_idx]
    x_scaled, x_mean, x_std = _standardize(x_train_raw, x_raw)
    y_scaled, y_mean, y_std = _standardize(y_train_raw, y_raw)
    return _PreparedTrainingData(
        x_raw=x_raw,
        y_raw=y_raw,
        x_scaled=x_scaled,
        y_scaled=y_scaled,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        input_columns=input_columns,
        target_columns=target_columns,
    )


def _build_runtime(
    config: TrainConfig,
    data: _PreparedTrainingData,
    output_dir: Path,
) -> _RuntimeObjects:
    train_ds = TensorDataset(
        torch.from_numpy(data.x_scaled[data.train_idx]),
        torch.from_numpy(data.y_scaled[data.train_idx]),
    )
    device = torch.device(config.device)
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    model = DynamicsMLP(
        input_dim=data.x_scaled.shape[1],
        output_dim=data.y_scaled.shape[1],
        hidden_sizes=config.hidden_sizes,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.use_amp and device.type == "cuda"
    )
    return _RuntimeObjects(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scaler=scaler,
        loader=loader,
        val_x=torch.from_numpy(data.x_scaled[data.val_idx]),
        val_y=torch.from_numpy(data.y_scaled[data.val_idx]),
        test_x=torch.from_numpy(data.x_scaled[data.test_idx]),
        device=device,
        pin_memory=pin_memory,
        best_path=output_dir / "best_model.pt",
    )


def _fit_model(
    config: TrainConfig,
    data: _PreparedTrainingData,
    runtime: _RuntimeObjects,
) -> tuple[list[dict[str, float]], float]:
    history: list[dict[str, float]] = []
    best_val = math.inf
    for epoch in range(1, config.epochs + 1):
        train_loss = _train_one_epoch(config, runtime)
        val_loss = _validation_loss(runtime)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        _log_training_epoch(epoch, train_loss, val_loss)
        if val_loss < best_val:
            best_val = val_loss
            _save_best_checkpoint(config, data, runtime)
    return history, best_val


def _train_one_epoch(config: TrainConfig, runtime: _RuntimeObjects) -> float:
    runtime.model.train()
    train_loss = 0.0
    train_count = 0
    for batch_x, batch_y in runtime.loader:
        batch_x = batch_x.to(runtime.device, non_blocking=runtime.pin_memory)
        batch_y = batch_y.to(runtime.device, non_blocking=runtime.pin_memory)
        runtime.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            "cuda", enabled=config.use_amp and runtime.device.type == "cuda"
        ):
            loss = runtime.loss_fn(runtime.model(batch_x), batch_y)
        runtime.scaler.scale(loss).backward()
        runtime.scaler.step(runtime.optimizer)
        runtime.scaler.update()
        train_loss += float(loss.detach().cpu()) * len(batch_x)
        train_count += len(batch_x)
    return train_loss / train_count


def _validation_loss(runtime: _RuntimeObjects) -> float:
    runtime.model.eval()
    with torch.no_grad():
        val_pred = runtime.model(runtime.val_x.to(runtime.device)).cpu()
        return float(runtime.loss_fn(val_pred, runtime.val_y).cpu())


def _save_best_checkpoint(
    config: TrainConfig,
    data: _PreparedTrainingData,
    runtime: _RuntimeObjects,
) -> None:
    torch.save(
        {
            "model_state_dict": runtime.model.state_dict(),
            "input_columns": data.input_columns,
            "target_columns": data.target_columns,
            "x_mean": data.x_mean.tolist(),
            "x_std": data.x_std.tolist(),
            "y_mean": data.y_mean.tolist(),
            "y_std": data.y_std.tolist(),
            "config": asdict(config),
        },
        runtime.best_path,
    )


def _evaluate_best_model(
    config: TrainConfig,
    data: _PreparedTrainingData,
    runtime: _RuntimeObjects,
    best_val: float,
) -> dict[str, object]:
    checkpoint = load_checkpoint_dict(
        runtime.best_path,
        map_location=runtime.device,
        required_keys=("model_state_dict",),
        artifact_name="per-step surrogate checkpoint",
    )
    runtime.model.load_state_dict(checkpoint["model_state_dict"])
    runtime.model.eval()
    with torch.no_grad():
        pred_scaled = runtime.model(runtime.test_x.to(runtime.device)).cpu().numpy()
    pred = pred_scaled * data.y_std + data.y_mean
    target = data.y_raw[data.test_idx]
    diff = pred - target
    return {
        "best_val_loss_scaled": best_val,
        "test_rmse_mean_unscaled": float(np.sqrt(np.vdot(diff, diff) / diff.size)),
        "test_rmse_by_target_unscaled": _rmse_by_column(
            pred, target, data.target_columns
        ),
        "test_nrmse_by_target_std": _normalized_rmse_by_column(
            pred, target, data.target_columns
        ),
        "test_r2_by_target": _r2_by_column(pred, target, data.target_columns),
        "n_rows": int(len(data.x_raw)),
        "n_train": int(len(data.train_idx)),
        "n_val": int(len(data.val_idx)),
        "n_test": int(len(data.test_idx)),
        "input_dim": len(data.input_columns),
        "target_dim": len(data.target_columns),
        "device": str(runtime.device),
    }


def _write_training_outputs(
    output_dir: Path,
    history: list[dict[str, float]],
    metrics: dict[str, object],
) -> None:
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    LOGGER.info("%s", json.dumps(metrics, indent=2))


def _log_training_epoch(epoch: int, train_loss: float, val_loss: float) -> None:
    LOGGER.info(
        "epoch=%03d train_loss=%.6f val_loss=%.6f",
        epoch,
        train_loss,
        val_loss,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=[512, 512, 256, 256]
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    config = TrainConfig(
        dataset=str(args.dataset),
        output_dir=str(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_sizes=args.hidden_sizes,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        device=args.device,
        use_amp=not args.no_amp,
    )
    train(config)


if __name__ == "__main__":
    main()
