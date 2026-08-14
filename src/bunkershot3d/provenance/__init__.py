"""Run provenance for BunkerShot3D: config hashing, RNG discipline, manifest.

Results that cannot be traced back to their inputs are not evidence
(finding B18). This package supplies the three pieces a reproducible run needs:

* :mod:`~bunkershot3d.provenance.hashing` -- canonical (RFC 8785 style)
  configuration hashing, emitting ``config_hash`` and ``physics_hash``.
* :mod:`~bunkershot3d.provenance.rng` -- ``SeedSequence`` based seeding whose
  streams are recordable and replayable.
* :mod:`~bunkershot3d.provenance.manifest` -- the :class:`RunManifest` written
  into every result file and its sibling JSON.
"""

from .hashing import (
    PHYSICS_EXCLUDED_FIELDS,
    FieldClass,
    canonical_json,
    classify_field,
    config_hash,
    leaf_field_paths,
    physics_hash,
    strip_excluded_fields,
)
from .manifest import (
    MANIFEST_ATTR_PREFIX,
    PROVENANCE_SUFFIX,
    RunManifest,
    Validity,
    library_versions,
)
from .rng import (
    ENTROPY_BITS,
    GENERATOR_NAME,
    SeedRecord,
    make_generator,
    new_entropy,
    root_seed_sequence,
    seed_record,
    spawn_generators,
    spawn_sequences,
)

__all__: list[str] = [
    "ENTROPY_BITS",
    "GENERATOR_NAME",
    "MANIFEST_ATTR_PREFIX",
    "PHYSICS_EXCLUDED_FIELDS",
    "PROVENANCE_SUFFIX",
    "FieldClass",
    "RunManifest",
    "SeedRecord",
    "Validity",
    "canonical_json",
    "classify_field",
    "config_hash",
    "leaf_field_paths",
    "library_versions",
    "make_generator",
    "new_entropy",
    "physics_hash",
    "root_seed_sequence",
    "seed_record",
    "spawn_generators",
    "spawn_sequences",
    "strip_excluded_fields",
]
