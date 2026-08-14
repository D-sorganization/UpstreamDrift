"""Reproducible RNG discipline for BunkerShot3D (issue #8617).

The rules (research digest section 7) are:

* Draw 128 bits of entropy with :func:`secrets.randbits`, wrap it in a
  :class:`numpy.random.SeedSequence`, and build
  ``Generator(PCG64DXSM(seed_sequence))``.
* Record ``SeedSequence.entropy`` **and** ``numpy.__version__``: NEP 19 permits
  the bit stream of a given generator to change across numpy X.Y releases, so an
  entropy value alone does not pin the numbers.
* Never call :func:`numpy.random.seed` -- global state is not recordable.
* Never derive worker seeds as ``root_seed + worker_id``; those streams can
  overlap. Spawn children from the parent sequence instead.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.random import PCG64DXSM, Generator, SeedSequence

__all__ = [
    "ENTROPY_BITS",
    "GENERATOR_NAME",
    "SeedRecord",
    "make_generator",
    "new_entropy",
    "root_seed_sequence",
    "seed_record",
    "spawn_generators",
    "spawn_sequences",
]

#: Width of the entropy drawn for a root seed sequence.
ENTROPY_BITS = 128

#: The bit generator every BunkerShot3D stream uses.
GENERATOR_NAME = "PCG64DXSM"


def new_entropy() -> int:
    """Return a fresh 128-bit cryptographically strong seed value.

    Returns:
        An integer in ``[0, 2**128)``.
    """
    return secrets.randbits(ENTROPY_BITS)


def root_seed_sequence(entropy: int | None = None) -> SeedSequence:
    """Return the root :class:`~numpy.random.SeedSequence` for a run.

    Args:
        entropy: Explicit entropy to replay a previous run. ``None`` draws a
            fresh 128-bit value.

    Returns:
        A seed sequence whose ``entropy`` attribute is the recorded value.

    Raises:
        TypeError: If ``entropy`` is not an ``int``.
        ValueError: If ``entropy`` is negative.
    """
    if entropy is None:
        entropy = new_entropy()
    if isinstance(entropy, bool) or not isinstance(entropy, int):
        raise TypeError(f"entropy must be an int, got {type(entropy).__name__}")
    if entropy < 0:
        raise ValueError(f"entropy must be non-negative, got {entropy}")
    return SeedSequence(entropy)


def make_generator(seed_sequence: SeedSequence) -> Generator:
    """Return ``Generator(PCG64DXSM(seed_sequence))``.

    Args:
        seed_sequence: The recorded seed sequence for this stream.

    Returns:
        A numpy generator seeded from ``seed_sequence``.

    Raises:
        TypeError: If ``seed_sequence`` is not a
            :class:`~numpy.random.SeedSequence`. A bare integer is rejected on
            purpose: it bypasses the recorded provenance.
    """
    if not isinstance(seed_sequence, SeedSequence):
        raise TypeError(
            "make_generator requires a SeedSequence so the stream is recorded; "
            f"got {type(seed_sequence).__name__}. Use root_seed_sequence() or "
            "spawn_sequences()."
        )
    return Generator(PCG64DXSM(seed_sequence))


def spawn_sequences(parent: SeedSequence, count: int) -> list[SeedSequence]:
    """Spawn ``count`` independent child sequences from ``parent``.

    Args:
        parent: The parent seed sequence.
        count: Number of children to spawn.

    Returns:
        The spawned child sequences, in spawn order.

    Raises:
        TypeError: If ``parent`` is not a :class:`~numpy.random.SeedSequence`.
        ValueError: If ``count`` is not positive.
    """
    if not isinstance(parent, SeedSequence):
        raise TypeError(f"parent must be a SeedSequence, got {type(parent).__name__}")
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    return list(parent.spawn(count))


def spawn_generators(parent: SeedSequence, count: int) -> list[Generator]:
    """Spawn ``count`` independent generators from ``parent``.

    Args:
        parent: The parent seed sequence.
        count: Number of generators to create.

    Returns:
        One :class:`~numpy.random.Generator` per spawned child sequence.
    """
    return [make_generator(child) for child in spawn_sequences(parent, count)]


def _numpy_version() -> str:
    """Return the running numpy version string."""
    return str(np.__version__)


@dataclass(frozen=True, slots=True)
class SeedRecord:
    """Everything needed to reconstruct one RNG stream exactly."""

    name: str
    entropy: int
    spawn_key: tuple[int, ...] = ()
    pool_size: int = 4
    n_children_spawned: int = 0
    generator: str = GENERATOR_NAME
    numpy_version: str = field(default_factory=_numpy_version)

    def to_sequence(self) -> SeedSequence:
        """Return the :class:`~numpy.random.SeedSequence` this record describes.

        Returns:
            A seed sequence equivalent to the recorded one; re-spawning it
            reproduces the same children in the same order.
        """
        return SeedSequence(
            entropy=self.entropy,
            spawn_key=self.spawn_key,
            pool_size=self.pool_size,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of this record."""
        return {
            "name": self.name,
            "entropy": self.entropy,
            "spawn_key": list(self.spawn_key),
            "pool_size": self.pool_size,
            "n_children_spawned": self.n_children_spawned,
            "generator": self.generator,
            "numpy_version": self.numpy_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SeedRecord:
        """Rebuild a record from :meth:`to_dict` output.

        Args:
            payload: Mapping produced by :meth:`to_dict`.

        Returns:
            The reconstructed record.

        Raises:
            KeyError: If ``name`` or ``entropy`` is missing.
        """
        return cls(
            name=str(payload["name"]),
            entropy=int(payload["entropy"]),
            spawn_key=tuple(int(key) for key in payload.get("spawn_key", ())),
            pool_size=int(payload.get("pool_size", 4)),
            n_children_spawned=int(payload.get("n_children_spawned", 0)),
            generator=str(payload.get("generator", GENERATOR_NAME)),
            numpy_version=str(payload.get("numpy_version", "")),
        )


def seed_record(seed_sequence: SeedSequence, name: str) -> SeedRecord:
    """Capture ``seed_sequence`` as a :class:`SeedRecord`.

    Args:
        seed_sequence: The stream to record.
        name: Stable label for the stream (``"grains"``, ``"worker-3"``, ...).

    Returns:
        A record carrying the entropy, spawn key, pool size, spawn count,
        generator name and numpy version.

    Raises:
        TypeError: If ``seed_sequence`` is not a
            :class:`~numpy.random.SeedSequence`.
        ValueError: If ``name`` is empty.
    """
    if not isinstance(seed_sequence, SeedSequence):
        raise TypeError(
            f"seed_sequence must be a SeedSequence, got {type(seed_sequence).__name__}"
        )
    if not name:
        raise ValueError("name must be a non-empty label for the stream")
    entropy = seed_sequence.entropy
    if not isinstance(entropy, int):  # pragma: no cover - defensive
        raise TypeError(
            "only integer-entropy seed sequences are recordable, got "
            f"{type(entropy).__name__}"
        )
    return SeedRecord(
        name=name,
        entropy=entropy,
        spawn_key=tuple(int(key) for key in seed_sequence.spawn_key),
        pool_size=int(seed_sequence.pool_size),
        n_children_spawned=int(seed_sequence.n_children_spawned),
    )
