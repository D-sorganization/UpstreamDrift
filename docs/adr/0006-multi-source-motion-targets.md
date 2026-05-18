# ADR 0006: Multi-Source Motion Targets

- Status: Accepted
- Date: 2026-05-08
- Related issues: #4487

## Context

The motion-matching pipeline in `src/shared/python/motion_matching/` was
originally built around a single frozen dataclass, `ClubTarget`, that
captured the time-aligned six-degree-of-freedom trajectory of a club
during a swing. All cost terms, validators, alignment helpers, and
visualisations consumed `ClubTarget` directly, and the Excel and C3D
loaders both produced one.

That design is no longer sufficient. Two adjacent capture modalities
are entering the pipeline:

1. **Ball-aware club captures** — some sources record both the club
   trajectory _and_ the launch-frame ball state on the same clock
   (impact-pinned). Cost terms that score launch quality need both
   tracks aligned to a shared time grid, but the ball track is
   optional and not present in legacy xlsx workbooks.
2. **Full-body motion capture** — C3D files from full-body marker sets
   carry a heterogeneous bag of segment trajectories (pelvis, spine,
   shoulders, elbows, wrists, hands, feet) that the pipeline wants to
   match against forward-kinematic predictions from a physics engine.
   These segments are independent of the club track and can come from
   a separate file.

Trying to bolt these onto `ClubTarget` creates a validation explosion:
optional ball fields, optional segment dictionaries, optional segment
labels, optional shared-clock metadata. Every cost term then has to
re-check which optional attributes are populated, and the post-init
validator becomes a thicket of cross-field guards.

The cost-function surface also needs to dispatch on what is actually
available in a given run. A "club only" run, a "club + ball" run, a
"club + body" run, and a "club + ball + body" run share large amounts
of code but score different terms.

A separate forcing function is naming. Several of the historical
identifiers leaked source-specific terminology into otherwise generic
code (the matcher tile is still labelled "Starting Pose Matcher", the
xlsx variant carried a vendor-specific label). Any restructuring is
also an opportunity to clear those out so file-on-disk names can stay
specific while everything in code, directories, and UI stays
source-agnostic.

## Decision

Introduce three frozen dataclasses and one aggregator, plus a pair of
format-agnostic dispatchers, and treat them as the canonical surface
for downstream cost code:

- `ClubTarget` — unchanged. Six-DoF club trajectory on a sim timegrid.
- `ClubBallTarget` — frozen dataclass adding launch-frame ball state on
  the same impact-pinned clock as the club track. Validates that ball
  samples and club samples share the same time vector.
- `BodyTarget` — frozen dataclass holding a labelled bag of segment
  trajectories from a full-body capture. Each segment carries its own
  position (and where available, orientation) array on the shared time
  vector. A `default_body_segments` helper returns the canonical
  segment labels for the supported marker set.
- `MultiSourceTarget` — aggregator that holds at most one `ClubTarget`,
  at most one `ClubBallTarget`, and at most one `BodyTarget`, plus the
  shared time grid. Exposes `has_club()`, `has_ball()`, and
  `has_body()` accessors that cost-function code dispatches on.

Loading is split into two format-agnostic dispatchers:

- `load_club_target(path, ...)` — already present; routes on extension
  (xlsx / xlsm / xls / c3d) to the per-format loader.
- `load_body_target(path, ...)` — new dispatcher for full-body marker
  sets. Currently routes only `.c3d` to the C3D body-segment loader,
  with the same dispatch shape as `load_club_target` so further
  formats can plug in without changing call sites.

A `MultiSourceTarget` is composed by the caller from one or more
loader outputs; loaders themselves stay independent, so a script that
only needs the club track does not pay the cost of opening a body
file.

### Generic-naming policy

- File-on-disk names (vendor-specific xlsx workbooks, named C3D files)
  remain whatever the source publishes them as.
- Everything in code, directories, and UI stays source-agnostic:
  `BodyTarget` not `MarkerSetTarget`, `load_body_target` not
  `load_<vendor>_body`, `motion_target_preview` not
  `<vendor>_starting_pose_matcher`.
- This keeps the public surface stable when new vendors or capture
  systems land — only the per-format loader inside `loaders/` changes.

## Alternatives Considered

### A. Single super-dataclass with optional fields

Add ball, segment-bag, and shared-clock fields to `ClubTarget` itself
and make them all `None`-able.

Rejected. The post-init validator becomes a combinatorial mess (ball
implies club, body may or may not share a clock with club, segment
labels must match a registered set when present, time vectors must
agree across whichever subset is populated). Cost-function code has
to keep guarding on `is None` for every term it touches. Worse, the
"is this a body target?" question stops being answerable by type and
becomes an attribute probe.

### B. Abstract base class + Target subclasses

Define a `Target` ABC and make `ClubTarget`, `ClubBallTarget`, and
`BodyTarget` subclasses; have cost terms pattern-match on `isinstance`.

Rejected. Cost terms are not polymorphic over a single target — most
of them want both a club track _and_ a body track at once, so an ABC
hierarchy implies dispatching N ways and re-assembling state on every
call. The flat frozen-dataclass + aggregator shape matches the actual
data flow more honestly: cost terms receive one `MultiSourceTarget`
and dispatch on `has_*()` for the terms they need.

## Consequences

### Positive

- Cost-function code reads a single `MultiSourceTarget` and dispatches
  on `has_club()`, `has_ball()`, `has_body()`. No more `is None`
  guards on optional attributes of the same dataclass.
- Loaders stay independent and composable. A caller that only needs
  the club track does not pay the cost of opening a body file.
- The dispatcher pattern (`load_club_target`, `load_body_target`)
  scales: adding a new file format means adding one entry to the
  routing table, not changing every call site.
- Naming cleanup clears the source-revealing identifiers out of code
  and UI without forcing renames on the underlying capture files.

### Negative

- One additional aggregation step at call sites that previously
  consumed a `ClubTarget` directly. A thin shim lets legacy code keep
  passing a bare `ClubTarget` for one release while migrating.
- Three dataclasses and one aggregator is more surface than one
  dataclass. The trade-off is paid back in cost-term clarity.

## Validation Strategy

- Every dataclass runs post-init validation:
  - `ClubBallTarget` checks that ball samples share the club time
    vector and that quaternions are unit-normalised within
    `QUAT_NORM_TOL`.
  - `BodyTarget` checks that every segment trajectory shares the
    declared time vector and that segment labels come from the
    registered set returned by `default_body_segments`.
  - `MultiSourceTarget` checks that whatever subset of
    `ClubTarget` / `ClubBallTarget` / `BodyTarget` it holds shares a
    time grid (when the caller has declared `impact_source`).
- Dispatchers refuse unsupported extensions with a `ValueError` that
  lists the supported set, matching the existing `load_club_target`
  error shape.
- Cost terms that require a particular track call the matching
  `has_*()` accessor and raise `ValueError` with a descriptive message
  if the required track is absent.

## Migration

- Existing `load_club_target` callers continue to work unchanged.
- Cost terms that consumed a bare `ClubTarget` accept either a
  `ClubTarget` or a `MultiSourceTarget` for one release; the bare
  form is wrapped internally and a `DeprecationWarning` points at the
  new aggregator.
- No on-disk format changes. Existing xlsx / C3D files are still
  loaded by the same per-format loaders.

## References

- `src/shared/python/motion_matching/club_target.py` — current
  `ClubTarget` definition.
- `src/shared/python/motion_matching/load_club_target.py` — existing
  dispatcher whose shape `load_body_target` mirrors.
- `src/shared/python/motion_matching/loaders/` — per-format loaders.
- `docs/user_guide/motion_matching/loading_targets.md` — companion
  user-guide for the new surface.
