"""KL annealing schedule helpers for the inverse CVAE training loop.

The CVAE loss is

    L = recon_mse + beta(t) * KL(q(z|x) || N(0, I)) + lambda_W * work_estimate

with ``beta(t)`` ramped from 0.0 to 1.0 over the first ``warmup_epochs``
epochs to mitigate KL collapse (also called *posterior collapse*), a
well-known failure mode in which ``q(z|x) == N(0, I)`` so the latent
ignores the conditioning input and the decoder behaves like an
unconditional generator.

Mitigations documented here for reference (only linear warmup is
implemented in this module; the others are notes for future work):

* **Linear warmup (this module).** Bowman et al. 2016 — ramp ``beta``
  linearly from 0 to 1 so the encoder builds a non-trivial posterior
  before the KL term starts pushing it toward the prior.
* **Free bits.** Kingma et al. 2016 — clip the per-dimension KL at a
  floor ``lambda_free`` so a small KL is never penalised; encourages
  every latent dimension to encode at least ``lambda_free`` nats.
* **Capacity scheduling.** Burgess et al. 2018 (β-VAE) — penalise
  ``|KL - C(t)|`` instead of ``KL``, with ``C(t)`` ramping the *target*
  KL (capacity) up over training. Allows higher peak KL without
  collapse.
* **Cyclical annealing.** Fu et al. 2019 — repeat the linear warmup in
  cycles so the model alternates between a "fitting" regime (low beta)
  and a "regularising" regime (high beta).

For #033 we use plain linear warmup: empirically sufficient on the
random-sweep dataset and easy to reason about. The work-regularisation
term provides additional pressure that prevents the latent from being
ignored entirely even at full ``beta``.
"""

from __future__ import annotations

from src.shared.python.core.contracts import precondition


@precondition(
    lambda epoch, *, total_epochs, warmup_epochs, max_beta=1.0: epoch >= 0,
    "epoch must be non-negative",
)
@precondition(
    lambda epoch, *, total_epochs, warmup_epochs, max_beta=1.0: total_epochs >= 1,
    "total_epochs must be at least 1",
)
@precondition(
    lambda epoch, *, total_epochs, warmup_epochs, max_beta=1.0: warmup_epochs >= 0,
    "warmup_epochs must be non-negative",
)
@precondition(
    lambda epoch, *, total_epochs, warmup_epochs, max_beta=1.0: max_beta >= 0.0,
    "max_beta must be non-negative",
)
def linear_kl_beta(
    epoch: int,
    *,
    total_epochs: int,
    warmup_epochs: int,
    max_beta: float = 1.0,
) -> float:
    """Return ``beta`` for the given epoch under linear warmup.

    Parameters
    ----------
    epoch
        Zero-based current epoch index.
    total_epochs
        Total training epochs (used only for documentation / clamping).
    warmup_epochs
        Number of epochs over which ``beta`` ramps from 0 to ``max_beta``.
        ``0`` disables warmup (returns ``max_beta`` immediately).
    max_beta
        Plateau value; defaults to ``1.0``.

    Returns
    -------
    float
        Current ``beta(t)``. Monotonic non-decreasing in ``epoch``.
    """
    if warmup_epochs <= 0:
        return float(max_beta)
    if epoch >= warmup_epochs:
        return float(max_beta)
    fraction = (epoch + 1) / float(warmup_epochs)
    if fraction > 1.0:
        fraction = 1.0
    return float(max_beta * fraction)


@precondition(
    lambda total_epochs, *, fraction=0.2: total_epochs >= 1,
    "total_epochs must be at least 1",
)
@precondition(
    lambda total_epochs, *, fraction=0.2: 0.0 <= fraction <= 1.0,
    "fraction must lie in [0, 1]",
)
def default_warmup_epochs(total_epochs: int, *, fraction: float = 0.2) -> int:
    """Return the default warmup length: 20% of ``total_epochs`` (rounded).

    Always at least 1 to prevent a degenerate ``beta == max_beta`` from epoch 0.
    """
    return max(1, int(round(total_epochs * fraction)))
