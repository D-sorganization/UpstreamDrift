"""Framework-specific :class:`TrainingJobRunner` adapters.

Each adapter wires a real training loop (PyTorch / Gymnasium / …) into
the headless training-controller contract. Adapters are lazy by design:
importing this package does NOT pull in heavy ML frameworks. Concrete
adapters import their framework inside :meth:`prepare` / :meth:`run`
(or behind ``importlib.util.find_spec`` in :meth:`can_run`) so the
controller stays usable in environments where torch / gymnasium are
absent.
"""

from __future__ import annotations

from .pytorch_cvae import PyTorchCVAERunner

__all__ = ["PyTorchCVAERunner"]
