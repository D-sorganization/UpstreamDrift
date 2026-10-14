"""Sand velocity, density and shear fields, extracted and persisted.

Issue #8710, epic #8699.  ADR-0033 chose the F1 2-D plane-strain MPM
tier, so this package is what turns a march of that solver into
something a view can animate and a reviewer can trace back to the run
that made it.

Four modules, four jobs, plus a persistent particle trajectory:

* :mod:`.schema` -- what a field *is*, tier-neutrally.  A continuum tier
  and a grain tier write the same containers, so switching tiers does
  not invalidate stored results.
* :mod:`.capture` -- reading the field out of an F1 march using the
  solver's own transfer operators, at a stride and a crop the caller
  chose deliberately.
* :mod:`.standing` -- what a field *claims*: tier, validity, where it
  says there is sand, and what was dropped. Separate from the container
  because contents and claims change for different reasons.
* :mod:`.store` -- writing and reading it through
  :mod:`bunkershot3d.io`, with the digest that makes the recorded tier
  and validity status checkable rather than merely present.
* :mod:`.trajectory` / :mod:`.trajectory_reader` -- the versioned 3-D
  particle-trajectory and contact-event artifact (issue #9545, folded-in
  #9692): stable grain ids, radii and masks, digests over every array,
  bounded streaming and an atomic finalisation marker.

The non-negotiable, in one sentence: the stored field carries its tier
and validity status as data covered by a content digest, so an
illustrative field cannot be relabelled by copying a file.
"""

from .capture import (
    F1_KINEMATICS_NOTE,
    GridFieldSample,
    capture_f1_field,
    sample_grid_field,
)
from .shotcapture import (
    WHOLE_SHOT_KINEMATICS_NOTE,
    WholeShotRecorder,
    capture_f1_shot_field,
)
from .schema import (
    FieldLayout,
    FieldQuantity,
    GrainRoster,
    GridGeometry,
    ParticleKind,
    SandFieldFrame,
    SandFieldSeries,
    series_digest,
)
from .standing import (
    DEFAULT_OCCUPANCY_FLOOR_FRACTION,
    FIELD_SCHEMA_VERSION,
    FieldIntegrityError,
    FieldProvenance,
    OccupancyRule,
    RetentionPolicy,
    RetentionRecord,
)
from .store import (
    DETERMINISTIC_SEED_NAME,
    deterministic_seed_record,
    field_manifest,
    load_field,
    save_field,
)
from .trajectory import (
    TRAJECTORY_SCHEMA_VERSION,
    ContactEvent,
    ParticleTrajectoryWriter,
    TrajectoryIncompleteError,
    TrajectoryIntegrityError,
    TrajectoryVersionError,
    final_marker_path,
)
from .trajectory_reader import ParticleTrajectoryReader
from .units import DENSITY_UNIT, SHEAR_RATE_UNIT, TIME_UNIT, VELOCITY_UNIT

__all__ = [
    "DEFAULT_OCCUPANCY_FLOOR_FRACTION",
    "DENSITY_UNIT",
    "DETERMINISTIC_SEED_NAME",
    "F1_KINEMATICS_NOTE",
    "FIELD_SCHEMA_VERSION",
    "SHEAR_RATE_UNIT",
    "TIME_UNIT",
    "TRAJECTORY_SCHEMA_VERSION",
    "VELOCITY_UNIT",
    "ContactEvent",
    "FieldIntegrityError",
    "FieldLayout",
    "FieldProvenance",
    "FieldQuantity",
    "GrainRoster",
    "GridFieldSample",
    "GridGeometry",
    "OccupancyRule",
    "ParticleKind",
    "ParticleTrajectoryReader",
    "ParticleTrajectoryWriter",
    "RetentionPolicy",
    "RetentionRecord",
    "SandFieldFrame",
    "SandFieldSeries",
    "TrajectoryIncompleteError",
    "TrajectoryIntegrityError",
    "TrajectoryVersionError",
    "WHOLE_SHOT_KINEMATICS_NOTE",
    "WholeShotRecorder",
    "capture_f1_field",
    "capture_f1_shot_field",
    "deterministic_seed_record",
    "field_manifest",
    "final_marker_path",
    "load_field",
    "sample_grid_field",
    "save_field",
    "series_digest",
]
