"""Reproducible random-number discipline for design studies.

Every study in this package draws randomness through a :class:`SeedRecord`.
The record stores the 128-bit entropy that seeded the study plus the NumPy
version that generated the stream, because NEP 19 permits bit-stream changes
across NumPy ``X.Y`` releases: a seed alone is not a reproduction recipe.

Rules enforced here (ADR-0032, research digest section 7):

- entropy comes from :func:`secrets.randbits` (128 bits), never from the clock;
- generators are :class:`numpy.random.Generator` over ``PCG64DXSM``;
- ``numpy.random.seed`` (the global legacy state) is never touched;
- parallel streams come from :meth:`numpy.random.SeedSequence.spawn`, never
  from ``root_seed + worker_id`` which produces overlapping streams.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import PCG64DXSM, Generator, SeedSequence

_ENTROPY_BITS = 128
_MAX_ENTROPY = (1 << _ENTROPY_BITS) - 1

__all__ = [
    "SeedRecord",
    "as_generator",
    "new_seed_record",
]


@dataclass(frozen=True, slots=True)
class SeedRecord:
    """A recorded, replayable random seed.

    Attributes:
        entropy: The 128-bit root entropy of the study.
        numpy_version: ``numpy.__version__`` at the time the stream was drawn.
    """

    entropy: int
    numpy_version: str

    def __post_init__(self) -> None:
        """Validate the recorded entropy.

        Raises:
            TypeError: If ``entropy`` is not an integer.
            ValueError: If ``entropy`` is out of the 128-bit range or the
                NumPy version string is empty.
        """
        if isinstance(self.entropy, bool) or not isinstance(self.entropy, int):
            raise TypeError(f"entropy must be an int, got {type(self.entropy)!r}")
        if not 0 <= self.entropy <= _MAX_ENTROPY:
            raise ValueError(
                f"entropy must fit in {_ENTROPY_BITS} bits, got {self.entropy}"
            )
        if not self.numpy_version:
            raise ValueError("numpy_version must be a non-empty string")

    def seed_sequence(self) -> SeedSequence:
        """Rebuild the root :class:`~numpy.random.SeedSequence`.

        Returns:
            A seed sequence carrying this record's entropy.
        """
        return SeedSequence(self.entropy)

    def generator(self) -> Generator:
        """Build the root generator for this record.

        Returns:
            A ``Generator(PCG64DXSM(...))`` seeded from :attr:`entropy`.
        """
        return Generator(PCG64DXSM(self.seed_sequence()))

    def spawn(self, count: int) -> list[Generator]:
        """Derive ``count`` statistically independent child generators.

        Uses :meth:`numpy.random.SeedSequence.spawn`; never arithmetic on the
        root seed, which can produce overlapping streams.

        Args:
            count: Number of independent generators to derive.

        Returns:
            A list of ``count`` generators.

        Raises:
            ValueError: If ``count`` is not positive.
        """
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        return [
            Generator(PCG64DXSM(child)) for child in self.seed_sequence().spawn(count)
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the record to a JSON-compatible mapping.

        Returns:
            A mapping with ``entropy`` and ``numpy_version`` keys.
        """
        return {"entropy": self.entropy, "numpy_version": self.numpy_version}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SeedRecord:
        """Rebuild a record from :meth:`to_dict` output.

        Args:
            payload: Mapping with ``entropy`` and ``numpy_version`` keys.

        Returns:
            The reconstructed seed record.

        Raises:
            KeyError: If a required key is missing.
        """
        return cls(
            entropy=int(payload["entropy"]),
            numpy_version=str(payload["numpy_version"]),
        )


def new_seed_record(entropy: int | SeedRecord | None = None) -> SeedRecord:
    """Create a seed record, drawing fresh entropy when none is supplied.

    Args:
        entropy: Explicit 128-bit entropy to replay a previous study, an
            existing :class:`SeedRecord` (returned unchanged), or ``None`` to
            draw fresh entropy from :func:`secrets.randbits`.

    Returns:
        A seed record pinned to the running NumPy version.
    """
    if isinstance(entropy, SeedRecord):
        return entropy
    root = secrets.randbits(_ENTROPY_BITS) if entropy is None else entropy
    return SeedRecord(entropy=root, numpy_version=np.__version__)


def as_generator(seed: int | SeedRecord | Generator | None) -> Generator:
    """Coerce a seed-like value into a generator.

    Intended for helpers that do not themselves record a manifest. Anything
    that publishes results must go through :func:`new_seed_record` so the
    entropy is recorded.

    Args:
        seed: A generator (returned unchanged), an integer entropy, a seed
            record, or ``None`` for fresh entropy.

    Returns:
        A NumPy generator.
    """
    if isinstance(seed, Generator):
        return seed
    return new_seed_record(seed).generator()
