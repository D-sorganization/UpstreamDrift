"""SwingInverseCVAE: conditional VAE for the inverse swing problem.

This module owns just the PyTorch class and its config. Training (#033) and
inference with rejection sampling (#034) live in sibling modules.

Architecture (per APPROACH.md §Architecture):

* Encoder: 1D-Transformer over the kinematic sequence -> per-timestep hidden
  states. Mean-pooled to produce a context vector ``h_x``. A small MLP head
  on ``h_x`` emits the diagonal-Gaussian posterior parameters
  ``(mu, log_var)``.
* Reparameterization: ``z = mu + exp(0.5 * log_var) * eps`` with
  ``eps ~ N(0, I)`` at train time; ``z = mu`` when ``sample=False``.
* Decoder: MLP on ``concat(z, h_x)`` emitting a flat ``(B, n_joints * 7)``
  coefficient vector. Output is pass-through; quaternion / bound clamping is
  layered in the post-processing stages of #033 / #034.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.shared.python.core.contracts import postcondition, precondition

from ._transformer import TransformerSequenceEncoder

# Number of coefficients per joint (A..G; see shared/README.md).
COEFFICIENTS_PER_JOINT = 7


@dataclass(frozen=True)
class CVAEConfig:
    """Hyperparameters for :class:`SwingInverseCVAE`.

    Defaults track the values listed in the issue body for #032; deviations
    here must also update INTERFACES.md.
    """

    n_joints: int
    n_timesteps: int = 300
    n_kinematic_channels: int = 12  # butt(3) + clubhead(3) + quat(4) + 2 reserved
    latent_dim: int = 16
    encoder_layers: int = 4
    encoder_heads: int = 4
    encoder_dim: int = 128
    decoder_hidden: int = 256
    dropout: float = 0.1


@dataclass(frozen=True)
class EncoderOutput:
    """Posterior parameters and the (re)sampled latent.

    All tensors share batch dim B and have feature dim ``CVAEConfig.latent_dim``.
    ``z`` equals ``mu`` when sampling is disabled (eval-time deterministic
    pass).
    """

    mu: torch.Tensor
    log_var: torch.Tensor
    z: torch.Tensor


def _validate_config(cfg: CVAEConfig) -> None:
    """Eager DbC-style precondition check on ``CVAEConfig`` values."""
    if cfg.n_joints <= 0:
        raise ValueError(f"n_joints must be positive; got {cfg.n_joints}")
    if cfg.n_timesteps <= 0:
        raise ValueError(f"n_timesteps must be positive; got {cfg.n_timesteps}")
    if cfg.n_kinematic_channels <= 0:
        raise ValueError(
            f"n_kinematic_channels must be positive; got {cfg.n_kinematic_channels}"
        )
    if cfg.latent_dim <= 0:
        raise ValueError(f"latent_dim must be positive; got {cfg.latent_dim}")
    if cfg.encoder_dim % cfg.encoder_heads != 0:
        raise ValueError(
            f"encoder_dim ({cfg.encoder_dim}) must be divisible by "
            f"encoder_heads ({cfg.encoder_heads})"
        )


class SwingInverseCVAE(nn.Module):
    """Conditional VAE: ``kinematics -> torque coefficients``.

    Inputs are ``(B, T, n_kinematic_channels)`` kinematic sequences; outputs
    are ``(B, n_joints * 7)`` flat coefficient vectors. The model owns the
    encoder, decoder, and reparameterization trick; the loss, training loop,
    and rejection-sampling inference live in their own issues (#033, #034).

    See APPROACH.md for the architecture rationale.
    """

    def __init__(self, cfg: CVAEConfig) -> None:
        super().__init__()
        _validate_config(cfg)
        self.cfg = cfg
        self._output_dim = cfg.n_joints * COEFFICIENTS_PER_JOINT

        self.encoder = TransformerSequenceEncoder(
            in_features=cfg.n_kinematic_channels,
            d_model=cfg.encoder_dim,
            n_heads=cfg.encoder_heads,
            n_layers=cfg.encoder_layers,
            dropout=cfg.dropout,
            max_seq_len=max(cfg.n_timesteps, 16),
        )

        # Posterior MLP: h_x -> (mu, log_var). 2*latent_dim head is the standard
        # split-output trick.
        self.posterior_head = nn.Sequential(
            nn.Linear(cfg.encoder_dim, cfg.encoder_dim),
            nn.GELU(),
            nn.Linear(cfg.encoder_dim, 2 * cfg.latent_dim),
        )

        # Decoder MLP: concat(z, h_x) -> coefficients.
        decoder_in = cfg.latent_dim + cfg.encoder_dim
        self.decoder_net = nn.Sequential(
            nn.Linear(decoder_in, cfg.decoder_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.decoder_hidden, cfg.decoder_hidden),
            nn.GELU(),
            nn.Linear(cfg.decoder_hidden, self._output_dim),
        )

    # ------------------------------------------------------------------
    # Helpers (kept method-level to satisfy LOD <= 2 in callers)
    # ------------------------------------------------------------------
    def _summarize(self, kinematics: torch.Tensor) -> torch.Tensor:
        """Encode a kinematic sequence and mean-pool to a context vector.

        Returns ``(B, encoder_dim)`` summary used by both the posterior head
        and the decoder.
        """
        hidden = self.encoder(kinematics)
        return hidden.mean(dim=1)

    def _split_posterior(self, h_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the posterior head and split into ``(mu, log_var)``."""
        params = self.posterior_head(h_x)
        mu, log_var = torch.chunk(params, 2, dim=-1)
        # Numerical-stability clamp on log_var. Matches typical VAE practice
        # (DKingma 2014 §2.3) and prevents NaNs from a runaway std.
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        return mu, log_var

    @staticmethod
    def _reparameterize(
        mu: torch.Tensor, log_var: torch.Tensor, *, sample: bool
    ) -> torch.Tensor:
        """Standard Gaussian reparameterization trick."""
        if not sample:
            return mu
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @precondition(
        lambda self, kinematics, *, sample=True: kinematics.dim() == 3,
        "kinematics must be 3D (B, T, n_kinematic_channels)",
    )
    @precondition(
        lambda self, kinematics, *, sample=True: (
            kinematics.shape[-1] == self.cfg.n_kinematic_channels
        ),
        "kinematics last-dim must equal cfg.n_kinematic_channels",
    )
    def encode(
        self,
        kinematics: torch.Tensor,
        *,
        sample: bool = True,
    ) -> EncoderOutput:
        """Encode kinematics into the posterior ``q(z | kinematics)``.

        Parameters
        ----------
        kinematics
            ``(B, T, n_kinematic_channels)`` float tensor.
        sample
            If ``True`` (default), draws ``z`` via the reparameterization
            trick. If ``False``, returns ``z = mu`` for deterministic
            decoding.

        Returns
        -------
        EncoderOutput
            ``mu``, ``log_var``, and the (re)sampled ``z``, each of shape
            ``(B, latent_dim)``.
        """
        h_x = self._summarize(kinematics)
        mu, log_var = self._split_posterior(h_x)
        z = self._reparameterize(mu, log_var, sample=sample)
        return EncoderOutput(mu=mu, log_var=log_var, z=z)

    @precondition(
        lambda self, z, kinematics=None, *, context=None: z.dim() == 2,
        "z must be 2D (B, latent_dim)",
    )
    @postcondition(
        lambda result: result.dim() == 2,
        "decode output must be 2D (B, n_joints*7)",
    )
    def decode(
        self,
        z: torch.Tensor,
        kinematics: torch.Tensor | None = None,
        *,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode ``z`` (and a kinematics summary) into a coefficient vector.

        Either ``kinematics`` or a precomputed ``context`` (``(B, encoder_dim)``)
        must be supplied; if both are given, ``context`` wins to avoid a
        redundant transformer pass during sampling.

        Returns
        -------
        torch.Tensor
            ``(B, n_joints * 7)`` raw coefficient vector. Bound enforcement
            (scaled tanh) and quaternion normalization are post-processing
            steps owned by the inference module (#034).
        """
        if context is None:
            if kinematics is None:
                raise ValueError("decode requires either `kinematics` or `context`")
            context = self._summarize(kinematics)
        if context.shape[0] != z.shape[0]:
            raise ValueError(
                f"batch mismatch: z={z.shape[0]} vs context={context.shape[0]}"
            )
        return self.decoder_net(torch.cat([z, context], dim=-1))

    def forward(
        self,
        kinematics: torch.Tensor,
        *,
        sample: bool = True,
    ) -> tuple[torch.Tensor, EncoderOutput]:
        """Encoder -> reparameterize -> decoder.

        Returns ``(coeffs, encoder_out)`` where ``coeffs`` has shape
        ``(B, n_joints * 7)`` and ``encoder_out`` carries the posterior for
        the loss in #033.
        """
        if kinematics.dim() != 3:
            raise ValueError(
                f"kinematics must be 3D (B, T, F); got {tuple(kinematics.shape)}"
            )
        if kinematics.shape[-1] != self.cfg.n_kinematic_channels:
            raise ValueError(
                f"kinematics last-dim must be {self.cfg.n_kinematic_channels}; "
                f"got {kinematics.shape[-1]}"
            )
        context = self._summarize(kinematics)
        mu, log_var = self._split_posterior(context)
        z = self._reparameterize(mu, log_var, sample=sample)
        coeffs = self.decode(z, context=context)
        return coeffs, EncoderOutput(mu=mu, log_var=log_var, z=z)

    @precondition(
        lambda self, kinematics, *, n_samples=1: n_samples >= 1,
        "n_samples must be at least 1",
    )
    @postcondition(
        lambda result: result.dim() == 3,
        "sample_coefficients output must be 3D (B, n_samples, n_joints*7)",
    )
    def sample_coefficients(
        self,
        kinematics: torch.Tensor,
        *,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Draw ``n_samples`` candidate coefficient vectors per input.

        Each draw uses an independent ``z ~ q(z | kinematics)``. The
        kinematic summary is computed once per input and reused for every
        sample.

        Returns
        -------
        torch.Tensor
            Shape ``(B, n_samples, n_joints * 7)``.
        """
        if kinematics.dim() != 3:
            raise ValueError(f"kinematics must be 3D; got {tuple(kinematics.shape)}")
        batch_size = kinematics.shape[0]
        context = self._summarize(kinematics)
        mu, log_var = self._split_posterior(context)
        std = torch.exp(0.5 * log_var)

        # Tile once across the sample dim so we issue a single decoder call
        # rather than a Python loop. Shapes: (B, S, latent_dim) and
        # (B, S, encoder_dim).
        mu_e = mu.unsqueeze(1).expand(batch_size, n_samples, mu.shape[-1])
        std_e = std.unsqueeze(1).expand_as(mu_e)
        eps = torch.randn_like(mu_e)
        z = mu_e + std_e * eps
        ctx_e = context.unsqueeze(1).expand(batch_size, n_samples, context.shape[-1])

        flat_z = z.reshape(batch_size * n_samples, -1)
        flat_ctx = ctx_e.reshape(batch_size * n_samples, -1)
        flat_out = self.decoder_net(torch.cat([flat_z, flat_ctx], dim=-1))
        return flat_out.reshape(batch_size, n_samples, self._output_dim)
