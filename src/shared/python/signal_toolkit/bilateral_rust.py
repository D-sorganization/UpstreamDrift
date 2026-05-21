"""Rust-accelerated bilateral filter facade.

This module exposes :func:`apply_bilateral_filter_rust` — a drop-in,
per-element-equivalent replacement for the pure-Python
:func:`signal_toolkit.filters.apply_bilateral_filter`. The eventual
migration is intended to be a one-line swap in callers::

    # before
    from signal_toolkit.filters import apply_bilateral_filter
    # after (when Rust wheel is available on the consumer's interpreter)
    from signal_toolkit.bilateral_rust import (
        apply_bilateral_filter_rust as apply_bilateral_filter,
    )

The Python signature, ``Signal``-in/``Signal``-out semantics, and metadata
keys are preserved exactly. If the ``tools_core`` Rust wheel is not
installed, importing the Rust function raises ``ImportError`` at call
time so that callers can fall back to the Python implementation
themselves; we deliberately do NOT silently substitute the slower
Python path here, because doing so would mask deployment misconfiguration
in performance-critical biomechanics pipelines.

Tracked task: GH issue #2569.
"""

from __future__ import annotations

import logging

import numpy as np

from .core import Signal

logger = logging.getLogger(__name__)

try:
    # `tools_core` exposes `signal` as a runtime PyO3 submodule (not a
    # filesystem-rooted module), so `import tools_core.signal` does not
    # work — we must reach in via attribute access on the parent.
    from tools_core import signal as _rust_signal  # type: ignore[attr-defined]

    _RUST_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on machines without wheel
    _rust_signal = None  # type: ignore[assignment]
    _RUST_AVAILABLE = False
    logger.warning(
        "bilateral_rust: tools_core wheel not available; using pure-Python path. "
        "See docs/development/rust_distribution.md"
    )


def apply_bilateral_filter_rust(
    signal: Signal,
    window_size: int = 5,
    sigma_space: float = 1.0,
    sigma_intensity: float = 0.1,
) -> Signal:
    """Apply a bilateral (edge-preserving) filter to a signal — Rust path.

    Args:
        signal: Input :class:`Signal`. Must have non-empty ``values``.
        window_size: Full window width. Half-window is ``window_size // 2``,
            matching the Python reference implementation.
        sigma_space: Spatial sigma (controls distance weighting). Must be > 0.
        sigma_intensity: Intensity sigma (controls value-similarity
            weighting). Must be > 0.

    Returns:
        New :class:`Signal` with filtered ``values``. Time, name suffix,
        units, and ``metadata`` keys mirror the Python facade.

    Raises:
        ImportError: If the ``tools_core`` Rust wheel is not installed.
        ValueError: If ``sigma_space`` or ``sigma_intensity`` are not > 0.
    """
    if not _RUST_AVAILABLE:
        raise ImportError(
            "tools_core Rust extension is not installed; "
            "fall back to signal_toolkit.filters.apply_bilateral_filter"
        )
    if sigma_space <= 0:
        raise ValueError(f"sigma_space must be > 0, got {sigma_space}")
    if sigma_intensity <= 0:
        raise ValueError(f"sigma_intensity must be > 0, got {sigma_intensity}")

    # Force contiguous f64 input — the PyO3 binding requires `as_slice`.
    values = np.ascontiguousarray(signal.values, dtype=np.float64)

    filtered = _rust_signal.bilateral_filter(
        values, int(window_size), float(sigma_space), float(sigma_intensity)
    )

    return Signal(
        time=signal.time,
        values=filtered,
        name=f"{signal.name}_bilateral",
        units=signal.units,
        metadata={
            **signal.metadata,
            "filter": "bilateral",
            "window": window_size,
            "sigma_space": sigma_space,
            "sigma_intensity": sigma_intensity,
            "backend": "rust",
        },
    )
