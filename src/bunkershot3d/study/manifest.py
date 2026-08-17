"""Run manifests: the record that makes a sweep replayable.

ADR-0032 consequence 6 requires every study artifact to carry enough
provenance to be re-run: the RNG entropy, the library versions whose numerical
streams it depends on, and the shape of the design that was sampled.

A manifest is deliberately *data*, not behaviour: it round-trips through JSON
so it can sit next to an HDF5 result file or in a git-tracked sweep
definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy

from .rng import SeedRecord, new_seed_record

__all__ = ["StudyManifest"]


@dataclass(frozen=True, slots=True)
class StudyManifest:
    """Provenance for one sampled design study.

    Attributes:
        seed: The recorded root entropy of the study.
        method: The sampling or analysis method (``"sobol"``, ``"morris"``...).
        parameter_names: Design-space parameter names, in column order.
        n_samples: Number of design points actually evaluated.
        numpy_version: NumPy version used.
        scipy_version: SciPy version used (``scipy.stats.qmc`` streams are
            version-dependent in the same way NumPy's are).
        extra: Method-specific scalars worth recording (``n_base``,
            ``n_levels``, ...). Values must be JSON-serialisable.
    """

    seed: SeedRecord
    method: str
    parameter_names: tuple[str, ...]
    n_samples: int
    numpy_version: str = field(default_factory=lambda: np.__version__)
    scipy_version: str = field(default_factory=lambda: scipy.__version__)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the manifest.

        Raises:
            ValueError: If the method is empty or ``n_samples`` is negative.
        """
        if not self.method:
            raise ValueError("method must be a non-empty string")
        if self.n_samples < 0:
            raise ValueError(f"n_samples must be non-negative, got {self.n_samples}")

    @property
    def dimension(self) -> int:
        """Number of design parameters.

        Returns:
            The length of :attr:`parameter_names`.
        """
        return len(self.parameter_names)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible mapping.

        Returns:
            A mapping suitable for ``json.dumps(..., allow_nan=False)``.
        """
        return {
            "seed": self.seed.to_dict(),
            "method": self.method,
            "parameter_names": list(self.parameter_names),
            "n_samples": self.n_samples,
            "numpy_version": self.numpy_version,
            "scipy_version": self.scipy_version,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StudyManifest:
        """Rebuild a manifest from :meth:`to_dict` output.

        Args:
            payload: Mapping produced by :meth:`to_dict`.

        Returns:
            The reconstructed manifest.

        Raises:
            KeyError: If a required key is missing.
        """
        return cls(
            seed=SeedRecord.from_dict(payload["seed"]),
            method=str(payload["method"]),
            parameter_names=tuple(payload["parameter_names"]),
            n_samples=int(payload["n_samples"]),
            numpy_version=str(payload["numpy_version"]),
            scipy_version=str(payload["scipy_version"]),
            extra=dict(payload.get("extra", {})),
        )

    @classmethod
    def create(
        cls,
        *,
        method: str,
        parameter_names: tuple[str, ...],
        n_samples: int,
        seed: int | SeedRecord | None = None,
        extra: dict[str, Any] | None = None,
    ) -> StudyManifest:
        """Build a manifest, drawing fresh entropy when ``seed`` is ``None``.

        Args:
            method: Sampling or analysis method name.
            parameter_names: Design-space parameter names in column order.
            n_samples: Number of design points evaluated.
            seed: Explicit entropy or seed record; ``None`` draws fresh.
            extra: Method-specific scalars to record.

        Returns:
            A populated manifest.
        """
        return cls(
            seed=new_seed_record(seed),
            method=method,
            parameter_names=tuple(parameter_names),
            n_samples=n_samples,
            extra=dict(extra or {}),
        )
