"""Non-overlapping grain placement shared by the DEM backends.

Issue #8612 (finding B16). The MuJoCo driver drew every grain from
``rng.uniform`` over the whole domain, so grains started life interpenetrating
and exploded on the first step — the 500-step "settle" phase only masked it.
The Chrono driver placed grains on ``np.linspace`` z-layers, whose spacing
collapses far below one grain diameter as soon as the population is realistic
(0.45 m / 50 000 grains = 9 um layers for 0.4 mm sand).

Both are replaced by a jittered cubic lattice filled from the floor upwards:
a bed, not a cloud, with a minimum separation guaranteed by construction.
"""

from __future__ import annotations

import numpy as np

from src.shared.python.core.contracts import ensure

#: Lattice pitch as a multiple of grain diameter. The excess over 1.0 is the
#: budget the random jitter is allowed to consume.
LATTICE_PITCH_FACTOR = 1.1


def lattice_capacity(
    extents: tuple[float, float, float],
    diameter: float,
    *,
    pitch_factor: float = LATTICE_PITCH_FACTOR,
) -> tuple[int, int, int]:
    """Number of lattice sites available along each axis."""
    if diameter <= 0.0:
        raise ValueError(f"diameter must be positive, got {diameter}")
    if pitch_factor <= 1.0:
        raise ValueError(f"pitch_factor must exceed 1.0, got {pitch_factor}")
    pitch = diameter * pitch_factor
    counts = []
    for extent in extents:
        if extent <= 0.0:
            raise ValueError(f"domain extents must be positive, got {extents}")
        usable = extent - diameter
        counts.append(0 if usable < 0.0 else int(usable / pitch) + 1)
    return (counts[0], counts[1], counts[2])


def lattice_positions(
    *,
    count: int,
    extents: tuple[float, float, float],
    diameter: float,
    rng: np.random.Generator,
    pitch_factor: float = LATTICE_PITCH_FACTOR,
) -> np.ndarray:
    """Place ``count`` grains on a jittered lattice, filling from the bottom up.

    The domain is the drivers' convention: ``x`` and ``y`` centred on the
    origin with extents ``lx``/``ly``, ``z`` running from the floor at 0 up to
    ``lz``.

    Jitter is bounded by ``(pitch - diameter) / 2``, so the minimum centre
    separation is at least ``diameter`` for every pair, whatever the seed.

    Args:
        count: Number of grains to place.
        extents: ``(lx, ly, lz)`` domain extents in metres.
        diameter: Largest grain diameter in metres (polydisperse populations
            must pass their upper bound so the guarantee still holds).
        rng: Seeded generator; placement is reproducible for a given seed.
        pitch_factor: Lattice pitch as a multiple of ``diameter``.

    Returns:
        ``(count, 3)`` array of grain centres.

    Raises:
        ValueError: The domain cannot hold ``count`` grains at this diameter.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")

    nx, ny, nz = lattice_capacity(extents, diameter, pitch_factor=pitch_factor)
    capacity = nx * ny * nz
    if capacity < count:
        raise ValueError(
            f"cannot place {count} grains of diameter {diameter:.4g} m in a "
            f"{extents[0]:.4g} x {extents[1]:.4g} x {extents[2]:.4g} m domain: "
            f"the lattice holds {capacity} grains "
            f"({nx} x {ny} x {nz}). Reduce grain_population.count, coarse-grain "
            "the grains, or enlarge the domain."
        )

    pitch = diameter * pitch_factor
    radius = diameter / 2.0
    lx, ly, _lz = extents

    # Centre the occupied lattice within the domain footprint.
    x0 = -(nx - 1) * pitch / 2.0
    y0 = -(ny - 1) * pitch / 2.0

    # Fill z-slowest so the lowest layers are complete first: a settled bed.
    layer, row, column = np.unravel_index(np.arange(count), (nz, ny, nx))
    sites = np.column_stack(
        (
            x0 + column * pitch,
            y0 + row * pitch,
            radius + layer * pitch,
        )
    )

    # A displacement of at most ``slack`` per grain keeps every pair separated
    # by at least ``diameter`` (two neighbours can each move by ``slack``).
    slack = (pitch - diameter) / 2.0
    jitter = rng.uniform(-1.0, 1.0, size=sites.shape) * (slack / np.sqrt(3.0))
    positions = sites + jitter

    # Keep every grain inside the walls.
    positions[:, 0] = np.clip(positions[:, 0], -lx / 2.0 + radius, lx / 2.0 - radius)
    positions[:, 1] = np.clip(positions[:, 1], -ly / 2.0 + radius, ly / 2.0 - radius)
    positions[:, 2] = np.maximum(positions[:, 2], radius)

    ensure(
        positions.shape == (count, 3),
        "lattice_positions must return one centre per grain",
        positions.shape,
    )
    return positions
