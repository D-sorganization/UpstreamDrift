"""Skeleton signature for the Option-3 inverse CVAE.

This file is **scaffold only**. Method bodies are intentionally
``raise NotImplementedError`` — Issue #032 fills them in. The single
purpose of this file at scaffold time is to lock the public API so
INTERFACES.md, TESTING.md, and downstream importers can be written
against it.

See:
    - APPROACH.md for architecture, loss, and inference protocol.
    - INTERFACES.md for the full set of contracts.
    - ASSUMPTIONS.md for the regimes in which this option is valid.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

# NOTE: Decorator imports kept here so the skeleton fails loudly if the
# shared contracts package moves. Implementation files re-import them.
from src.shared.python.core.contracts import postcondition, precondition  # noqa: F401


@dataclass(frozen=True)
class CVAEConfig:
    """Hyperparameters for the inverse CVAE.

    See APPROACH.md §Architecture for the rationale behind every default.
    """

    n_joints: int
    seq_len: int = 300
    d_model: int = 256
    encoder_layers: int = 4
    encoder_heads: int = 8
    d_ctx: int = 256
    d_z: int = 32
    decoder_hidden: tuple[int, ...] = (512, 512)
    coef_bounds: tuple[float, ...] = (1000.0, 1000.0, 500.0, 500.0, 100.0, 100.0, 25.0)
    dropout: float = 0.1


class SwingInverseCVAE(nn.Module):
    """Conditional VAE: club kinematic trajectory -> torque coefficients.

    Encoder consumes the club kinematic sequence (butt position, clubhead
    position, club orientation quaternion over time) and emits a context
    vector ``h_x``. A posterior MLP combines ``h_x`` with the embedded
    ground-truth coefficients to produce the parameters of a diagonal
    Gaussian ``q(z | x, theta)``. The decoder consumes ``(z, h_x)`` and
    emits a coefficient vector ``theta_hat`` whose components are bounded
    by ``config.coef_bounds`` via per-coefficient scaled tanh.

    See APPROACH.md for the full architecture diagram and ASSUMPTIONS.md
    for the regimes under which this is the right model.

    Notes
    -----
    Issue #032 implements the bodies. Until then, every method raises
    ``NotImplementedError`` — this is intentional and protected by
    ``test_cvae_overfits_single_trial`` once that test is wired up.
    """

    def __init__(self, config: CVAEConfig) -> None:
        super().__init__()
        self.config = config
        # Submodules (encoder, posterior MLP, decoder, theta_embed) are
        # constructed in Issue #032. No bodies here.
        raise NotImplementedError("Issue #032 implements the constructor.")

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a club kinematic sequence into a context vector ``h_x``.

        Parameters
        ----------
        x
            Tensor of shape ``(batch, seq_len, 12)``: butt(3) + clubhead(3)
            + quaternion(4) per timestep, with two channels reserved for
            future body-marker conditioning. Normalized via Option 2's
            ``NormalizationStats``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, d_ctx)``.
        """
        raise NotImplementedError("Issue #032 implements encode().")

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------
    def decode(self, z: torch.Tensor, h_x: torch.Tensor) -> torch.Tensor:
        """Decode latent ``z`` and context ``h_x`` into coefficients.

        Output shape: ``(batch, n_joints * 7)``. Each component is bounded
        by the corresponding entry in ``config.coef_bounds`` via scaled
        tanh — clipping is never required.
        """
        raise NotImplementedError("Issue #032 implements decode().")

    # ------------------------------------------------------------------
    # Training-time forward pass
    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, theta: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Training-time forward pass.

        Encodes ``x``, samples ``z`` from ``q(z | x, theta)`` via the
        reparameterization trick, decodes, and returns a diagnostic dict
        consumed by the loss function in APPROACH.md §Loss function.

        Returns
        -------
        dict
            Keys: ``theta_hat``, ``z``, ``mu``, ``log_sigma``, ``h_x``.
        """
        raise NotImplementedError("Issue #032 implements forward().")

    # ------------------------------------------------------------------
    # Inference-time sampling
    # ------------------------------------------------------------------
    def sample_coefficients(
        self, x: torch.Tensor, n_samples: int = 32
    ) -> torch.Tensor:
        """Draw ``n_samples`` coefficient vectors from ``p(theta | x)``.

        Each sample corresponds to one mode of the posterior — different
        latent draws give different valid coefficient vectors (subject to
        the rejection-sampling protocol in APPROACH.md §Inference).

        Returns
        -------
        torch.Tensor
            Shape ``(n_samples, batch, n_joints * 7)``.
        """
        raise NotImplementedError("Issue #032 implements sample_coefficients().")
