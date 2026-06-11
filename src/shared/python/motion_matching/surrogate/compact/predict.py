"""Inference helpers for the compact-schema swing surrogate.

Exposed surface:
    SwingSurrogate.from_checkpoint(path)
        Factory bound to :class:`SwingSurrogate` via ``__init_subclass__``-free
        monkey-patching at import time.
    predict_trajectory(model, theta) -> dict[str, np.ndarray]
        Vectorised inference returning physical-unit channel arrays.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import (
    CHANNEL_SLICES,
    CoeffNormalizer,
    SurrogateConfig,
    SwingSurrogate,
    az_pol_to_shaft_axis,
)

_LOGGER = logging.getLogger(__name__)

# Conversion factor reading as "<numerator> per <denominator>": multiply
# a value in mph by ``_MPS_PER_MPH`` to get metres-per-second.
_MPS_PER_MPH: float = 1.0 / 2.2369362920544


# --------------------------------------------------------------------------- #
# Checkpoint factory                                                          #
# --------------------------------------------------------------------------- #


def _config_from_payload(payload: dict[str, Any]) -> SurrogateConfig:
    """Reconstruct a :class:`SurrogateConfig` from the saved-payload dict."""
    raw = dict(payload.get("config", {}))
    if "coeff_bounds" in raw and raw["coeff_bounds"] is not None:
        raw["coeff_bounds"] = tuple(raw["coeff_bounds"])
    if "decoder_hidden" in raw and raw["decoder_hidden"] is not None:
        raw["decoder_hidden"] = int(raw["decoder_hidden"])
    return SurrogateConfig(**raw)


def _from_checkpoint(
    cls: type[SwingSurrogate],
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> SwingSurrogate:
    """Load a serialised :class:`SwingSurrogate` from ``path``.

    Args:
        cls: Always ``SwingSurrogate`` (bound by ``classmethod``).
        path: Checkpoint file (typically ``checkpoint_best.pt``).
        map_location: Torch device for the loaded weights.
        strict: Pass-through to ``Module.load_state_dict``.

    Returns:
        A configured :class:`SwingSurrogate` with weights loaded and the
        normaliser remembered as the ``coeff_normalizer`` attribute.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        ValueError: If the payload schema is unrecognised.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    payload = torch.load(p, map_location=map_location, weights_only=False)
    if "model_state_dict" not in payload:
        raise ValueError(
            f"checkpoint missing 'model_state_dict' key (got keys={list(payload)})"
        )
    cfg = _config_from_payload(payload)
    model = cls(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    model.eval()
    norm_meta = payload.get("normalizer", {})
    bounds = norm_meta.get("coeff_bounds", cfg.coeff_bounds)
    n_joints = norm_meta.get("n_joints", cfg.n_joints)
    model.coeff_normalizer = CoeffNormalizer(  # type: ignore[attr-defined, assignment]
        n_joints=n_joints, coeff_bounds=tuple(bounds)
    )
    return model


# Bind ``from_checkpoint`` as a classmethod on SwingSurrogate at import time.
# (Spec asks for ``SwingSurrogate.from_checkpoint(path)``.)
SwingSurrogate.from_checkpoint = classmethod(_from_checkpoint)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Vectorised prediction                                                       #
# --------------------------------------------------------------------------- #


def _coerce_theta(theta: Any) -> torch.Tensor:
    """Accept numpy arrays / lists / tensors; return ``(B, D)`` float32 tensor."""
    if isinstance(theta, torch.Tensor):
        t = theta.to(dtype=torch.float32)
    else:
        t = torch.as_tensor(np.asarray(theta), dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    if t.ndim != 2:
        raise ValueError(
            f"theta must be 1-D (D,) or 2-D (B, D); got ndim={t.ndim}, "
            f"shape={tuple(t.shape)}"
        )
    return t


def predict_trajectory(
    model: SwingSurrogate,
    theta: np.ndarray | torch.Tensor | list[float],
    *,
    device: str | torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Run the surrogate on raw (physical-unit) coefficients.

    Args:
        model: A trained :class:`SwingSurrogate` (loaded via
            :meth:`SwingSurrogate.from_checkpoint` so ``coeff_normalizer``
            is set, or constructed manually with one assigned).
        theta: ``(D,)`` or ``(B, D)`` polynomial coefficients in physical
            units (the bounds documented in PROJECT_SPEC.md §4).
        device: Optional override; defaults to the device the model is on.

    Returns:
        Dict of numpy arrays in physical units::

            r_clubhead:    (B, T, 3)  metres
            v_clubhead:    (B, T, 3)  m/s
            r_grip:        (B, T, 3)  metres
            clubhead_speed:(B, T)     mph
            shaft_axis:    (B, T, 3)  unit vector (denormalised from az/polar)

    Raises:
        AttributeError: If ``model.coeff_normalizer`` is missing.
        ValueError: If ``theta`` shape doesn't match the model.
    """
    if not hasattr(model, "coeff_normalizer"):
        raise AttributeError(
            "model.coeff_normalizer is missing — load via "
            "SwingSurrogate.from_checkpoint() or assign a CoeffNormalizer"
        )
    normalizer: CoeffNormalizer = model.coeff_normalizer  # type: ignore[attr-defined, assignment]
    target_device = (
        torch.device(device) if device is not None else next(model.parameters()).device
    )
    theta_t = _coerce_theta(theta).to(target_device)
    if theta_t.shape[-1] != model.cfg.coeff_dim:
        raise ValueError(
            f"theta trailing dim {theta_t.shape[-1]} != model.cfg.coeff_dim "
            f"({model.cfg.coeff_dim})"
        )

    model.eval()
    with torch.no_grad():
        coeffs_norm = normalizer.normalize(theta_t)
        traj = model(coeffs_norm)
    traj_np = traj.detach().cpu().numpy()
    return _split_channels(traj_np)


def _split_channels(traj: np.ndarray) -> dict[str, np.ndarray]:
    """Slice a ``(B, T, 12)`` array into the documented physical-unit dict.

    Conversion notes:
        * ``r_*`` and ``v_*`` are already in metres / m/s — they pass through.
        * ``clubhead_speed`` is in mph in the compact dataset, so it stays mph.
        * ``shaft_axis`` is reconstructed from the (azimuth, polar) channels
          back into a unit ``(x, y, z)`` 3-vec.
    """
    if traj.ndim != 3 or traj.shape[-1] != 12:
        raise ValueError(f"expected (B, T, 12) trajectory, got shape={traj.shape}")
    out: dict[str, np.ndarray] = {}
    for name, (lo, hi) in CHANNEL_SLICES.items():
        if name == "clubhead_speed":
            out[name] = traj[..., lo:hi].squeeze(-1)
        elif name == "shaft_axis_az_pol":
            az_pol = torch.as_tensor(traj[..., lo:hi], dtype=torch.float32)
            shaft = az_pol_to_shaft_axis(az_pol).numpy()
            out["shaft_axis"] = shaft
        else:
            out[name] = traj[..., lo:hi]
    return out


def predict_clubhead_speed_ms(
    model: SwingSurrogate,
    theta: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Convenience wrapper: clubhead speed in metres/second.

    The surrogate emits mph (matching the compact dataset's
    ``clubhead_speed_mph`` column); this helper converts to SI for callers
    that prefer metres.
    """
    out = predict_trajectory(model, theta)
    return out["clubhead_speed"] * _MPS_PER_MPH
