"""SwingInverseCVAE: conditional VAE mapping hand-path trajectories to
polynomial-coefficient posteriors (Option 3, GH issue #4076).

This module defines the *model* only. Training is in ``training.py`` and
inference is in ``predict.py``.

Architecture
------------
* **Conditioning input ``c``**: ``(batch, T, 12)`` float32 trajectory with
  the same per-channel layout the forward surrogate consumes
  (butt(3) + clubhead(3) + grip-quat(4) + 2 reserved). The cVAE sees the
  trajectory as the conditioning signal, never as the reconstruction target.
* **Encoder** ``f_enc(c)``: 1-D causal convolutional stack over the time
  axis (kernel 5, stride 2, three blocks) followed by global average
  pooling. Output is a ``(batch, encoder_dim)`` summary used by both the
  posterior head and the decoder. ~250 K params at the default 12->128
  channel widths.
* **Posterior** ``q(z|x, c)``: MLP head over ``concat(summary, x_proj)``
  emitting ``(mu_q, logvar_q)``. ``x_proj`` is a small linear projection
  of the *true* coefficient vector during training. At inference time the
  prior head is used instead.
* **Prior** ``p(z|c)``: parallel MLP head over ``summary`` only emitting
  ``(mu_p, logvar_p)``. Sampling-time only.
* **Decoder** ``p(theta|z, c)``: MLP on ``concat(z, summary)`` emitting a
  ``(batch, 189)`` raw coefficient vector. Output passes through a
  ``tanh``-and-scale step that respects the per-letter physical bounds:
  ``|A,B|<=1000``, ``|C,D|<=500``, ``|E,F|<=100``, ``|G|<=25``
  (PROJECT_SPEC.md §4).
* **Latent dim** ``z``: 32 by default, sized to give the posterior enough
  capacity to express bimodality across the 27-joint x 7-letter (= 189)
  coefficient space without over-fitting an 8-trial smoke fixture.
* **Total parameter count** at defaults: ~1.0 M (well within the 1-4 M
  budget called out in the issue).

DbC
---
``forward`` validates trajectory dtype/shape and the optional ``coeffs``
argument; both paths raise ``TypeError``/``ValueError`` with descriptive
messages. The 189 coefficient output is post-clamped to the physical bounds
so downstream consumers never see out-of-range values, even on an untrained
model. The KL terms are non-negative by construction (see
``kl_divergence`` for the closed-form expression).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import torch
from torch import Tensor, nn

from src.shared.python.motion_matching._checkpoint_artifacts import (
    load_checkpoint_dict,
    require_schema_version,
)

# ---------------------------------------------------------------------------
# Constants from PROJECT_SPEC.md §4 / COMPACT_DATASET_SCHEMA.md
# ---------------------------------------------------------------------------

DEFAULT_N_JOINTS: int = 27
COEFFICIENTS_PER_JOINT: int = 7
DEFAULT_COEFFICIENT_DIM: int = DEFAULT_N_JOINTS * COEFFICIENTS_PER_JOINT  # 189
DEFAULT_TRAJECTORY_CHANNELS: int = 12
DEFAULT_LATENT_DIM: int = 32

# Per-letter symmetric bounds in physical units (Newton-metres for torque
# coefficients of polynomial in t).
COEFFICIENT_LETTER_BOUNDS: tuple[float, ...] = (
    1000.0,  # A
    1000.0,  # B
    500.0,  # C
    500.0,  # D
    100.0,  # E
    100.0,  # F
    25.0,  # G
)


def build_coefficient_bound_vector(
    n_joints: int = DEFAULT_N_JOINTS,
    *,
    scale_factor: float = 1.0,
) -> Tensor:
    """Return the symmetric upper bound for each of ``n_joints * 7`` coefficients.

    Layout matches the dataset's flat 189-vec ordering:
    ``(joint_index * 7) + letter_index`` so element ``i`` of the returned
    tensor is the bound for ``coefficients[i]``.

    Args:
        n_joints: Number of joints (must be positive).
        scale_factor: Multiplicative factor on the per-letter bounds. The
            empirical compact dataset has letter-G coefficients reaching
            roughly ±1000 N·m — the spec's ±25 N·m bound clamps a healthy
            chunk of the real distribution. Pass ``scale_factor > 1`` (e.g.
            40) to widen the symmetric range while keeping the per-letter
            ratio. ``1.0`` keeps the spec's nominal bounds.

    Raises:
        ValueError: If ``n_joints <= 0`` or ``scale_factor <= 0``.
    """
    if n_joints <= 0:
        raise ValueError(f"n_joints must be positive, got {n_joints}")
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")
    bounds = torch.tensor(COEFFICIENT_LETTER_BOUNDS, dtype=torch.float32).repeat(
        n_joints
    )
    return bounds * float(scale_factor)


# ---------------------------------------------------------------------------
# Configs and lightweight result dataclasses
# ---------------------------------------------------------------------------


_BoundStrategy = Literal["spec", "empirical"]

#: Empirical scale factor matching the deterministic regressor's
#: ``coefficient_scale_factor=50`` default. The compact dataset's
#: letter-G coefficients reach roughly 40-50× the spec's nominal ±25 N·m
#: bound, so ``"spec"`` clamps a real fraction of the training
#: distribution; ``"empirical"`` widens the bounds symmetrically by 50×.
EMPIRICAL_BOUND_SCALE: float = 50.0


@dataclass(frozen=True)
class CVAEConfig:
    """Architectural hyperparameters for :class:`SwingInverseCVAE`.

    Defaults yield ~1.0 M parameters. Latent dim 32 was chosen empirically
    so the prior can express coefficient multi-modality without dwarfing
    the 189-dim output space (z covers about 17% of theta's dim, a common
    rule of thumb for Gaussian cVAEs).

    The decoder applies ``tanh(raw) * coefficient_bounds`` so every output
    is clamped to a symmetric range. Two strategies are available:

    * ``"spec"`` — use PROJECT_SPEC.md §4 bounds verbatim
      (``|A,B|<=1000``, ``|C,D|<=500``, ``|E,F|<=100``, ``|G|<=25``).
    * ``"empirical"`` — multiply the per-letter bounds by
      ``EMPIRICAL_BOUND_SCALE`` (default 50.0), matching the deterministic
      regressor's ``coefficient_scale_factor`` default. This is the right
      choice when training on the real compact dataset where letter G
      coefficients reach ~±1000 N·m and the ``"spec"`` clamp truncates
      a healthy slice of the distribution.

    ``"spec"`` is preserved as default for back-compat with already-trained
    research checkpoints.
    """

    n_joints: int = DEFAULT_N_JOINTS
    coefficients_per_joint: int = COEFFICIENTS_PER_JOINT
    trajectory_channels: int = DEFAULT_TRAJECTORY_CHANNELS
    latent_dim: int = DEFAULT_LATENT_DIM
    encoder_channels: tuple[int, ...] = (64, 128, 256)
    encoder_kernel: int = 5
    encoder_stride: int = 2
    decoder_hidden: int = 1024
    coeff_proj_dim: int = 128
    dropout: float = 0.1
    coefficient_bound_strategy: _BoundStrategy = "spec"

    @property
    def coefficient_dim(self) -> int:
        return self.n_joints * self.coefficients_per_joint

    @property
    def coefficient_bound_scale(self) -> float:
        """Multiplier applied to the per-letter spec bounds in the decoder."""
        if self.coefficient_bound_strategy == "spec":
            return 1.0
        if self.coefficient_bound_strategy == "empirical":
            return EMPIRICAL_BOUND_SCALE
        raise ValueError(
            "coefficient_bound_strategy must be 'spec' or 'empirical'; "
            f"got {self.coefficient_bound_strategy!r}"
        )


@dataclass(frozen=True)
class EncoderOutput:
    """Posterior + prior parameters produced by a forward pass.

    All tensors share the leading batch dim. ``z`` is the (re)parameterised
    latent drawn from the posterior at training time, or from the prior
    when no ``coeffs`` were supplied.
    """

    mu_q: Tensor
    logvar_q: Tensor
    mu_p: Tensor
    logvar_p: Tensor
    z: Tensor


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class _Conv1dEncoder(nn.Module):
    """Causal-style 1-D conv stack consuming ``(B, T, C)`` -> ``(B, D)``."""

    def __init__(
        self,
        in_channels: int,
        widths: tuple[int, ...],
        kernel: int,
        stride: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        padding = kernel // 2
        for width in widths:
            layers.append(
                nn.Conv1d(
                    prev, width, kernel_size=kernel, stride=stride, padding=padding
                )
            )
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = width
        self.body = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, traj: Tensor) -> Tensor:
        # traj: (B, T, C) -> (B, C, T)
        h = traj.transpose(1, 2)
        h = self.body(h)
        return h.mean(dim=-1)  # (B, D)


def _gaussian_head(in_features: int, hidden: int, latent_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.GELU(),
        nn.Linear(hidden, 2 * latent_dim),
    )


def _split_gaussian(params: Tensor) -> tuple[Tensor, Tensor]:
    mu, logvar = torch.chunk(params, 2, dim=-1)
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    return mu, logvar


def _reparameterise(mu: Tensor, logvar: Tensor, *, sample: bool) -> Tensor:
    if not sample:
        return mu
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)


def kl_divergence(
    mu_q: Tensor, logvar_q: Tensor, mu_p: Tensor, logvar_p: Tensor
) -> Tensor:
    """Batched closed-form KL between two diagonal Gaussians.

    KL(q || p) = 0.5 * sum_i [ (var_q + (mu_q - mu_p)^2) / var_p
                                - 1 + (logvar_p - logvar_q) ]

    Returns a 1-D tensor of length B (mean over latent dims is *not* taken;
    callers reduce as they see fit). Output is non-negative by construction
    (subject to floating-point round-off near zero).
    """
    elementwise = kl_divergence_per_dim(mu_q, logvar_q, mu_p, logvar_p)
    return elementwise.sum(dim=-1)


def kl_divergence_per_dim(
    mu_q: Tensor, logvar_q: Tensor, mu_p: Tensor, logvar_p: Tensor
) -> Tensor:
    """Element-wise KL between two diagonal Gaussians ``(B, latent_dim)``.

    Returned with the latent axis intact so callers can apply free-bits
    (per-dim KL clamping) before summing — see :mod:`.training` for the
    standard usage. Non-negative up to floating-point round-off.
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    diff = mu_q - mu_p
    return 0.5 * ((var_q + diff * diff) / var_p - 1.0 + (logvar_p - logvar_q))


class SwingInverseCVAE(nn.Module):
    """Conditional VAE: trajectory -> coefficient posterior.

    See module-level docstring for the architecture. The class exposes the
    public surface required by :mod:`training` and :mod:`predict`:

    * ``forward(trajectory, coeffs=None)`` — returns ``(coeff_pred, EncoderOutput)``.
    * ``sample(trajectory, n_samples)`` — draws from the prior conditioned
      on a single trajectory; used at inference time.
    * ``from_checkpoint(path)`` — load a saved state dict + config.
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __init__(self, cfg: CVAEConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or CVAEConfig()
        if cfg.n_joints <= 0:
            raise ValueError(f"n_joints must be positive, got {cfg.n_joints}")
        if cfg.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {cfg.latent_dim}")
        if cfg.trajectory_channels <= 0:
            raise ValueError(
                f"trajectory_channels must be positive, got {cfg.trajectory_channels}"
            )
        self.cfg = cfg

        self.encoder = _Conv1dEncoder(
            in_channels=cfg.trajectory_channels,
            widths=cfg.encoder_channels,
            kernel=cfg.encoder_kernel,
            stride=cfg.encoder_stride,
            dropout=cfg.dropout,
        )
        encoded_dim = self.encoder.out_dim

        self.coeff_projector = nn.Sequential(
            nn.Linear(cfg.coefficient_dim, cfg.coeff_proj_dim),
            nn.GELU(),
        )

        self.posterior_head = _gaussian_head(
            encoded_dim + cfg.coeff_proj_dim, cfg.decoder_hidden, cfg.latent_dim
        )
        self.prior_head = _gaussian_head(
            encoded_dim, cfg.decoder_hidden, cfg.latent_dim
        )

        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim + encoded_dim, cfg.decoder_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.decoder_hidden, cfg.decoder_hidden),
            nn.GELU(),
            nn.Linear(cfg.decoder_hidden, cfg.coefficient_dim),
        )

        # Stored as a non-trainable buffer so it moves with .to(device).
        # The strategy multiplier widens the symmetric range when the
        # config requests empirical bounds (see ``CVAEConfig`` docstring).
        self.register_buffer(
            "coefficient_bounds",
            build_coefficient_bound_vector(
                cfg.n_joints, scale_factor=cfg.coefficient_bound_scale
            ),
            persistent=False,
        )

    # ---------------- internal helpers (LOD-friendly) ----------------

    def _encode_context(self, trajectory: Tensor) -> Tensor:
        return self.encoder(trajectory)

    def _posterior(self, context: Tensor, coeffs: Tensor) -> tuple[Tensor, Tensor]:
        proj = self.coeff_projector(coeffs)
        joined = torch.cat([context, proj], dim=-1)
        return _split_gaussian(self.posterior_head(joined))

    def _prior(self, context: Tensor) -> tuple[Tensor, Tensor]:
        return _split_gaussian(self.prior_head(context))

    def _decode(self, z: Tensor, context: Tensor) -> Tensor:
        raw = self.decoder(torch.cat([z, context], dim=-1))
        # Bound-aware activation: tanh -> scale by per-letter symmetric bound.
        bounds = self.coefficient_bounds
        assert isinstance(bounds, Tensor)
        return torch.tanh(raw) * bounds

    # ---------------- input validation ----------------

    def _validate_trajectory(self, trajectory: Tensor) -> None:
        if not isinstance(trajectory, Tensor):
            raise TypeError(
                f"trajectory must be torch.Tensor, got {type(trajectory).__name__}"
            )
        if trajectory.dim() != 3:
            raise ValueError(
                f"trajectory must be 3-D (B, T, C); got shape {tuple(trajectory.shape)}"
            )
        if trajectory.shape[-1] != self.cfg.trajectory_channels:
            raise ValueError(
                f"trajectory last-dim must be {self.cfg.trajectory_channels} "
                f"(matching surrogate); got {trajectory.shape[-1]}"
            )
        if trajectory.dtype != torch.float32:
            raise TypeError(f"trajectory must be float32; got {trajectory.dtype}")

    def _validate_coeffs(self, coeffs: Tensor, batch: int) -> None:
        if not isinstance(coeffs, Tensor):
            raise TypeError(f"coeffs must be torch.Tensor, got {type(coeffs).__name__}")
        if coeffs.dim() != 2:
            raise ValueError(
                "coeffs must be 2-D (B, coefficient_dim); "
                f"got shape {tuple(coeffs.shape)}"
            )
        if coeffs.shape[0] != batch:
            raise ValueError(
                f"coeffs batch {coeffs.shape[0]} != trajectory batch {batch}"
            )
        if coeffs.shape[-1] != self.cfg.coefficient_dim:
            raise ValueError(
                f"coeffs last-dim must equal coefficient_dim={self.cfg.coefficient_dim}; "
                f"got {coeffs.shape[-1]}"
            )

    # ---------------- public API ----------------

    def forward(
        self,
        trajectory: Tensor,
        coeffs: Tensor | None = None,
        *,
        sample: bool | None = None,
    ) -> tuple[Tensor, EncoderOutput]:
        """Encode trajectory + (optional) coefficients, sample z, decode.

        When ``coeffs`` is given (training): z is drawn from the posterior
        ``q(z|x, c)``. When ``coeffs`` is None (inference): z is drawn from
        the prior ``p(z|c)`` and the posterior parameters are duplicated
        from the prior so downstream KL evaluation degenerates to zero.

        ``sample`` defaults to ``self.training`` so calls under ``model.eval()``
        return deterministic decodes (z = mu); explicitly pass ``sample=True``
        to override.
        """
        if sample is None:
            sample = self.training
        self._validate_trajectory(trajectory)
        context = self._encode_context(trajectory)
        mu_p, logvar_p = self._prior(context)

        if coeffs is None:
            mu_q, logvar_q = mu_p, logvar_p
        else:
            self._validate_coeffs(coeffs, trajectory.shape[0])
            mu_q, logvar_q = self._posterior(context, coeffs)

        z = _reparameterise(mu_q, logvar_q, sample=sample)
        coeff_pred = self._decode(z, context)
        return coeff_pred, EncoderOutput(
            mu_q=mu_q, logvar_q=logvar_q, mu_p=mu_p, logvar_p=logvar_p, z=z
        )

    @torch.no_grad()
    def sample(
        self,
        trajectory: Tensor,
        n_samples: int = 8,
        *,
        deterministic_mean: bool = False,
    ) -> Tensor:
        """Draw ``n_samples`` coefficient vectors from the prior conditioned on c.

        Returns a ``(B, n_samples, coefficient_dim)`` float32 tensor in
        physical units (already bounded by ``COEFFICIENT_LETTER_BOUNDS``).
        Set ``deterministic_mean=True`` to skip the eps draw and return the
        prior mean decoded; useful for unit tests / regression locks.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        self._validate_trajectory(trajectory)
        context = self._encode_context(trajectory)
        mu_p, logvar_p = self._prior(context)
        std = torch.exp(0.5 * logvar_p)

        batch = trajectory.shape[0]
        latent_dim = mu_p.shape[-1]
        mu_e = mu_p.unsqueeze(1).expand(batch, n_samples, latent_dim)
        std_e = std.unsqueeze(1).expand_as(mu_e)
        z = mu_e if deterministic_mean else mu_e + std_e * torch.randn_like(mu_e)

        ctx_e = context.unsqueeze(1).expand(batch, n_samples, context.shape[-1])
        flat_z = z.reshape(batch * n_samples, latent_dim)
        flat_ctx = ctx_e.reshape(batch * n_samples, context.shape[-1])
        decoded = self._decode(flat_z, flat_ctx)
        return decoded.reshape(batch, n_samples, self.cfg.coefficient_dim)

    # ---------------- (de)serialisation ----------------

    def state_payload(self) -> dict:
        """Return a checkpoint-ready dict bundling weights and config."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "config": _config_to_dict(self.cfg),
            "state_dict": self.state_dict(),
        }

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, map_location: str | torch.device | None = None
    ) -> SwingInverseCVAE:
        """Re-instantiate a model from a payload produced by ``state_payload``.

        Raises:
            FileNotFoundError: if ``path`` does not exist.
            ValueError: if the payload is missing keys or has an
                incompatible schema version.
        """
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        payload = load_checkpoint_dict(
            ckpt_path,
            map_location=map_location,
            required_keys=("state_dict", "config", "schema_version"),
            artifact_name="SwingInverseCVAE checkpoint",
        )
        require_schema_version(
            payload,
            cls.SCHEMA_VERSION,
            artifact_name="SwingInverseCVAE checkpoint",
        )
        cfg_dict = payload.get("config")
        if cfg_dict is None:
            raise ValueError("checkpoint missing 'config' entry")
        cfg = _config_from_dict(cfg_dict)
        model = cls(cfg)
        model.load_state_dict(payload["state_dict"])
        return model


def _config_to_dict(cfg: CVAEConfig) -> dict:
    return {
        "n_joints": cfg.n_joints,
        "coefficients_per_joint": cfg.coefficients_per_joint,
        "trajectory_channels": cfg.trajectory_channels,
        "latent_dim": cfg.latent_dim,
        "encoder_channels": list(cfg.encoder_channels),
        "encoder_kernel": cfg.encoder_kernel,
        "encoder_stride": cfg.encoder_stride,
        "decoder_hidden": cfg.decoder_hidden,
        "coeff_proj_dim": cfg.coeff_proj_dim,
        "dropout": cfg.dropout,
        "coefficient_bound_strategy": cfg.coefficient_bound_strategy,
    }


def _config_from_dict(d: dict) -> CVAEConfig:
    # ``coefficient_bound_strategy`` was added after the initial release; old
    # checkpoints don't carry it. Default to ``"spec"`` so they keep loading.
    raw_strategy = str(d.get("coefficient_bound_strategy", "spec"))
    if raw_strategy not in ("spec", "empirical"):
        raise ValueError(
            "checkpoint coefficient_bound_strategy must be 'spec' or 'empirical'; "
            f"got {raw_strategy!r}"
        )
    return CVAEConfig(
        n_joints=int(d["n_joints"]),
        coefficients_per_joint=int(d["coefficients_per_joint"]),
        trajectory_channels=int(d["trajectory_channels"]),
        latent_dim=int(d["latent_dim"]),
        encoder_channels=tuple(int(x) for x in d["encoder_channels"]),
        encoder_kernel=int(d["encoder_kernel"]),
        encoder_stride=int(d["encoder_stride"]),
        decoder_hidden=int(d["decoder_hidden"]),
        coeff_proj_dim=int(d["coeff_proj_dim"]),
        dropout=float(d["dropout"]),
        coefficient_bound_strategy=raw_strategy,  # type: ignore[arg-type]
    )


def parameter_count(model: nn.Module) -> int:
    """Sum of trainable parameter counts. Convenience for tests / reports."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def to_numpy(t: Tensor) -> np.ndarray:
    """Detach + CPU + numpy. Helper used by the predict surface."""
    return t.detach().cpu().numpy()
