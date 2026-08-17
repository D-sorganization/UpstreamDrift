"""Persisting a sand field through the existing ``io/`` layer (issue #8710).

This module is the only place that knows both what a
:class:`~bunkershot3d.fields.schema.SandFieldSeries` is and what
:mod:`bunkershot3d.io` stores.  The io layer moves named arrays plus a
metadata string; the schema layer knows what those arrays mean.  Keeping
the two apart is what lets the field schema gain a quantity without the
file format changing version, and vice versa.

The load path is the interesting half
-------------------------------------

:func:`load_field` recomputes the digest over the metadata and the arrays
it just read and compares it with the one on disk.  A mismatch raises
:class:`~bunkershot3d.fields.schema.FieldIntegrityError` and the field
does not load.  That is what makes issue #8710's non-negotiable
enforceable rather than aspirational:

* renaming the file changes nothing a reader consults, because the tier
  is read out of the file;
* editing the tier attribute inside the file breaks the digest;
* swapping the arrays under an honest label breaks it too.

There is no "load anyway" flag.  A field whose declared standing does not
match its contents has no standing, and the one thing a caller must not
be able to do is look at it and decide it seems fine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..io.schema import (
    BunkerShotResultReader,
    BunkerShotResultWriter,
    SandFieldPayload,
)
from ..provenance.hashing import canonical_json, config_hash, physics_hash
from ..provenance.manifest import RunManifest, Validity
from ..provenance.rng import SeedRecord, root_seed_sequence, seed_record
from ..solvers.envelope import EnvelopeStatus
from .schema import (
    FieldIntegrityError,
    FieldLayout,
    FieldProvenance,
    GridGeometry,
    OccupancyRule,
    RetentionRecord,
    SandFieldSeries,
    series_digest,
)

__all__ = [
    "DETERMINISTIC_SEED_NAME",
    "deterministic_seed_record",
    "field_manifest",
    "load_field",
    "save_field",
]

DETERMINISTIC_SEED_NAME = "f1_draws_no_random_numbers"
"""Name of the seed record a deterministic tier writes.

:class:`~bunkershot3d.provenance.manifest.RunManifest` refuses an empty
seed tuple, on the grounds that unrecorded seeds make a run
irreproducible.  For F1 the reproducibility fact *is* that there were no
random numbers, so the record says exactly that rather than inventing a
seed nothing consumed.  A grain tier writing the same schema records its
real seeds and this name never appears."""

_VALIDITY_BY_STATUS = {
    EnvelopeStatus.WITHIN: Validity.VALID,
    EnvelopeStatus.EXTRAPOLATED: Validity.OUT_OF_ENVELOPE,
    EnvelopeStatus.BEYOND_VALIDATION: Validity.OUT_OF_ENVELOPE,
    EnvelopeStatus.REFUSED: Validity.INVALID,
}


def deterministic_seed_record() -> SeedRecord:
    """The seed record of a tier that draws no random numbers."""
    return seed_record(root_seed_sequence(0), DETERMINISTIC_SEED_NAME)


def field_manifest(
    series: SandFieldSeries, *, seeds: tuple[SeedRecord, ...] = ()
) -> RunManifest:
    """Build the run manifest for a field, from the field's own provenance.

    The manifest's hashes are taken over the solver settings the field
    already carries, so a stored field and its manifest cannot disagree
    about what produced it.

    Args:
        series: The field.
        seeds: Seed records. Defaults to
            :func:`deterministic_seed_record` when the field's own
            provenance records none, which is the F1 case.

    Returns:
        The manifest, ready for :class:`BunkerShotResultWriter`.
    """
    provenance = series.provenance
    settings = dict(provenance.settings)
    chosen = seeds or provenance.seeds or (deterministic_seed_record(),)
    reason = "; ".join(provenance.reasons) or provenance.headline()
    return RunManifest.capture(
        config_hash=config_hash(settings),
        physics_hash=physics_hash(settings),
        seeds=chosen,
        solver=provenance.solver_name,
        fidelity_tier=provenance.fidelity_tier.value,
        validity=_VALIDITY_BY_STATUS[provenance.envelope_status],
        validity_reason=reason,
    )


def save_field(
    path: Path | str,
    series: SandFieldSeries,
    *,
    manifest: RunManifest | None = None,
) -> Path:
    """Write a field, its metadata and its digest to a result file.

    The arrays are cast to the retention policy's ``store_dtype`` first,
    and the digest is taken over the **cast** values, so a load returns
    bit-for-bit what a load returns -- rather than a digest of a
    precision the file does not hold.

    Args:
        path: Destination ``.h5`` path; an existing file is overwritten.
        series: The field to store.
        manifest: Run provenance. Defaults to :func:`field_manifest`, so
            a sidecar ``<file>.provenance.json`` is always written and a
            field is always traceable to its run.

    Returns:
        The path written.
    """
    record = field_manifest(series) if manifest is None else manifest
    stored = _stored_form(series)
    policy = stored.retention.policy
    payload = SandFieldPayload(
        time_s=np.asarray(stored.time_s, dtype=np.float64),
        arrays=_arrays(stored, policy.store_dtype),
        metadata=canonical_json(stored.metadata()),
        digest=series_digest(stored),
    )
    with BunkerShotResultWriter(path, manifest=record) as writer:
        writer.write_sand_field(
            payload,
            compression=policy.compression,
            compression_level=policy.compression_level,
        )
    return Path(path)


def _stored_form(series: SandFieldSeries) -> SandFieldSeries:
    """The series exactly as it will come back off disk.

    Casting down and straight back up is not a no-op, which is the
    point: the digest has to cover the numbers a reader will see, not
    the ones the solver had.
    """
    dtype = series.retention.policy.store_dtype
    if dtype == "float64":
        return series
    down = np.dtype(dtype)

    def cast(array: NDArray[np.float64]) -> NDArray[np.float64]:
        return array.astype(down).astype(np.float64)

    def cast_optional(
        array: NDArray[np.float64] | None,
    ) -> NDArray[np.float64] | None:
        return None if array is None else cast(array)

    return SandFieldSeries(
        time_s=series.time_s,
        velocity_m_s=cast(series.velocity_m_s),
        density_kg_m3=cast(series.density_kg_m3),
        shear_rate_1_s=cast_optional(series.shear_rate_1_s),
        positions_m=cast_optional(series.positions_m),
        layout=series.layout,
        geometry=series.geometry,
        provenance=series.provenance,
        retention=series.retention,
        occupancy=series.occupancy,
        body_outline_m=cast_optional(series.body_outline_m),
    )


def load_field(path: Path | str) -> SandFieldSeries:
    """Read a field back and refuse it if its digest does not match.

    Args:
        path: Source ``.h5`` path.

    Returns:
        The field, with the tier and validity status it was written with.

    Raises:
        ValueError: If the file records no field at all.
        FieldIntegrityError: If the recomputed digest disagrees with the
            stored one -- which is what an edited tier, an edited status
            or a swapped array looks like from here.
    """
    with BunkerShotResultReader(path) as reader:
        payload = reader.read_sand_field()
    if payload is None:
        raise ValueError(
            f"{path} records no sand field; it may be a schema v1/v2 result, or "
            "a run that stored only the clubhead and wrench streams"
        )
    series = _series_from(payload)
    recomputed = series_digest(series)
    if recomputed != payload.digest:
        raise FieldIntegrityError(
            f"the sand field in {path} does not match its recorded digest "
            f"(stored {payload.digest[:16]}..., recomputed {recomputed[:16]}...). "
            f"It declares tier {series.provenance.fidelity_tier.value} and status "
            f"{series.provenance.envelope_status.value}; one of those, or the "
            "arrays under them, has been changed since it was written."
        )
    return series


def _arrays(series: SandFieldSeries, store_dtype: str) -> dict[str, np.ndarray]:
    """The bulk arrays of a series, under the names the io layer stores."""
    dtype = np.dtype(store_dtype)
    arrays: dict[str, np.ndarray] = {
        "velocity": series.velocity_m_s.astype(dtype),
        "density": series.density_kg_m3.astype(dtype),
    }
    if series.shear_rate_1_s is not None:
        arrays["shear_rate"] = series.shear_rate_1_s.astype(dtype)
    if series.positions_m is not None:
        arrays["positions"] = series.positions_m.astype(dtype)
    if series.body_outline_m is not None:
        arrays["body_outline"] = series.body_outline_m.astype(dtype)
    return arrays


def _series_from(payload: SandFieldPayload) -> SandFieldSeries:
    """Rebuild a typed series from what the io layer handed back."""
    metadata = json.loads(payload.metadata)
    geometry = metadata.get("geometry")
    shear = payload.arrays.get("shear_rate")
    positions = payload.arrays.get("positions")
    outline = payload.arrays.get("body_outline")
    if metadata.get("has_shear_rate") and shear is None:
        raise FieldIntegrityError(
            "the field metadata declares a shear rate that is not in the file"
        )
    if metadata.get("has_body_outline") and outline is None:
        raise FieldIntegrityError(
            "the field metadata declares an intruder outline that is not in the "
            "file, so a slice could not tell sand ahead of the sole from sand "
            "riding up the face"
        )
    if metadata.get("has_positions") and positions is None:
        raise FieldIntegrityError(
            "the field metadata declares per-frame positions that are not in "
            "the file, so its samples have no location"
        )
    return SandFieldSeries(
        time_s=payload.time_s,
        velocity_m_s=payload.arrays["velocity"],
        density_kg_m3=payload.arrays["density"],
        shear_rate_1_s=shear,
        positions_m=positions,
        layout=FieldLayout(str(metadata["layout"])),
        geometry=None if geometry is None else GridGeometry.from_dict(geometry),
        provenance=FieldProvenance.from_dict(metadata["provenance"]),
        retention=RetentionRecord.from_dict(metadata["retention"]),
        occupancy=OccupancyRule.from_dict(metadata["occupancy"]),
        body_outline_m=outline,
    )
