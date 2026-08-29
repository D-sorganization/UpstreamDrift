"""Sand velocity, density and shear fields, extracted and persisted.

Issue #8710, epic #8699.  ADR-0033 chose the F1 2-D plane-strain MPM
tier, so this package is what turns a march of that solver into
something a view can animate and a reviewer can trace back to the run
that made it.

Four modules, four jobs:

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
    GridGeometry,
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
from .units import DENSITY_UNIT, SHEAR_RATE_UNIT, TIME_UNIT, VELOCITY_UNIT

__all__ = [
    "DEFAULT_OCCUPANCY_FLOOR_FRACTION",
    "DENSITY_UNIT",
    "DETERMINISTIC_SEED_NAME",
    "F1_KINEMATICS_NOTE",
    "FIELD_SCHEMA_VERSION",
    "SHEAR_RATE_UNIT",
    "TIME_UNIT",
    "VELOCITY_UNIT",
    "FieldIntegrityError",
    "FieldLayout",
    "FieldProvenance",
    "FieldQuantity",
    "GridFieldSample",
    "GridGeometry",
    "OccupancyRule",
    "RetentionPolicy",
    "RetentionRecord",
    "SandFieldFrame",
    "SandFieldSeries",
    "WHOLE_SHOT_KINEMATICS_NOTE",
    "WholeShotRecorder",
    "capture_f1_field",
    "capture_f1_shot_field",
    "deterministic_seed_record",
    "field_manifest",
    "load_field",
    "sample_grid_field",
    "save_field",
    "series_digest",
]
