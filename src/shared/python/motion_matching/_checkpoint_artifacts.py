"""Safe torch checkpoint loading helpers for motion-matching artifacts."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

SURROGATE_CHECKPOINT_REQUIRED_KEYS = (
    "model_state_dict",
    "input_columns",
    "target_columns",
    "x_mean",
    "x_std",
    "y_mean",
    "y_std",
    "config",
)


def load_checkpoint_dict(
    path: str | Path,
    *,
    map_location: Any = None,
    required_keys: tuple[str, ...] = (),
    artifact_name: str = "checkpoint",
) -> dict[str, Any]:
    """Load a tensor/plain-metadata checkpoint without enabling pickle globals."""
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    try:
        payload = _load_weights_only_checkpoint(ckpt_path, map_location=map_location)
    except pickle.UnpicklingError as exc:
        raise ValueError(
            f"{artifact_name} at {ckpt_path} cannot be loaded safely; "
            "migrate it to a tensor/plain-metadata checkpoint before loading"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"{artifact_name} at {ckpt_path} is not a dict payload "
            f"(got {type(payload).__name__})"
        )
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(
            f"{artifact_name} at {ckpt_path} missing required key(s): "
            f"{', '.join(missing)}"
        )
    return payload


def load_surrogate_checkpoint(
    path: str | Path,
    *,
    map_location: Any = None,
    artifact_name: str = "surrogate checkpoint",
) -> dict[str, Any]:
    """Load the standard DynamicsMLP surrogate checkpoint artifact safely."""
    return load_checkpoint_dict(
        path,
        map_location=map_location,
        required_keys=SURROGATE_CHECKPOINT_REQUIRED_KEYS,
        artifact_name=artifact_name,
    )


def _load_weights_only_checkpoint(
    path: Path,
    *,
    map_location: Any,
) -> Any:
    import torch

    return torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )


def require_schema_version(
    payload: dict[str, Any],
    expected: str,
    *,
    artifact_name: str = "checkpoint",
) -> None:
    """Validate the checkpoint schema marker."""
    actual = payload.get("schema_version")
    if actual != expected:
        raise ValueError(
            f"{artifact_name} schema_version must be {expected!r}; got {actual!r}"
        )
