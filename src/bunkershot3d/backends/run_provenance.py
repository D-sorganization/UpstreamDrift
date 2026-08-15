"""Run provenance for the F3 grain-scale drivers (issue #8608, finding B18).

#8617 built :class:`~bunkershot3d.provenance.manifest.RunManifest` and taught
:class:`~bunkershot3d.io.schema.BunkerShotResultWriter` to persist it, but no
driver ever passed one, so every result file carried a schema version and no
provenance: no config hash, no seeds, no library versions, no fidelity tier and
no validity verdict. A result that cannot be traced back to its inputs is not
evidence.

Two honesty rules are enforced here rather than left to each driver:

**The recorded seeds are the seeds actually used.** The drivers seed
``np.random.default_rng`` with fixed literals, which is a PCG64 stream, not the
PCG64DXSM discipline of :mod:`bunkershot3d.provenance.rng`; LIGGGHTS decks seed
LAMMPS' own Park/Miller generator. Each record says which, because a manifest
that names the wrong generator is worse than no manifest.

**No F3 result is labelled valid.** ADR-0032 records the DEM tier as
intractable at true grain scale -- 2.1e8 grains and days per shot -- and keeps
these backends in-tree only as accepted debt. A file produced by one of them
carries a verdict that says so.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..provenance import RunManifest, SeedRecord, Validity, config_hash, physics_hash

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import BunkerShotConfig

__all__ = [
    "DEM_FIDELITY_TIER",
    "GRAIN_POSITION_SEED",
    "GRAIN_RADII_SEED",
    "LIGGGHTS_GENERATOR",
    "NUMPY_DEFAULT_GENERATOR",
    "dem_run_manifest",
    "fixed_seed_record",
]

#: Fidelity tier of every DEM backend in this package (ADR-0032).
DEM_FIDELITY_TIER = "F3"

#: Generator behind ``np.random.default_rng`` -- *not* the PCG64DXSM used by
#: :mod:`bunkershot3d.provenance.rng`.
NUMPY_DEFAULT_GENERATOR = "PCG64"

#: LAMMPS/LIGGGHTS seeds drive its internal Park/Miller generator.
LIGGGHTS_GENERATOR = "LIGGGHTS-RanPark"

#: Fixed seed the drivers use for the grain size draw.
GRAIN_RADII_SEED = 42

#: Fixed seed the drivers use for grain placement.
GRAIN_POSITION_SEED = 43


def fixed_seed_record(
    name: str, entropy: int, generator: str = NUMPY_DEFAULT_GENERATOR
) -> SeedRecord:
    """Record a hard-coded seed exactly as the driver uses it.

    Args:
        name: Stable label for the stream (``"grain-positions"``, ...).
        entropy: The literal seed value passed to the generator.
        generator: Name of the generator the seed drives.

    Returns:
        A seed record that replays the stream.
    """
    return SeedRecord(name=name, entropy=int(entropy), generator=generator)


def dem_run_manifest(
    config: BunkerShotConfig,
    *,
    solver: str,
    seeds: tuple[SeedRecord, ...],
    validity: Validity = Validity.OUT_OF_ENVELOPE,
    validity_reason: str = "",
) -> RunManifest:
    """Build the run manifest for one grain-scale (F3) backend run.

    Args:
        config: The configuration the run is driven by; both hashes are taken
            from it.
        solver: Solver identifier written into the file (``"chrono"``, ...).
        seeds: Every RNG stream the run depends on. At least one is required:
            an unrecorded stream makes the run unreproducible.
        validity: Verdict on the result. Defaults to out-of-envelope because
            no F3 backend reaches true grain scale.
        validity_reason: Justification for the verdict.

    Returns:
        A populated manifest ready to hand to the result writer.

    Raises:
        ValueError: ``seeds`` is empty.
    """
    return RunManifest.capture(
        config_hash=config_hash(config),
        physics_hash=physics_hash(config),
        seeds=tuple(seeds),
        solver=solver,
        fidelity_tier=DEM_FIDELITY_TIER,
        validity=validity,
        validity_reason=validity_reason,
    )
