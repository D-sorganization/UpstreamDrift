# docs(motion-matching): ADR + user guide for multi-source motion targets

## Why

This effort introduces three target types (`ClubTarget`, `ClubBallTarget`, `BodyTarget`), a multi-source aggregator (`MultiSourceTarget`), three loader formats (xlsx / c3d / mat), a new dispatcher, a renamed module surface, and a substantially expanded matcher GUI. That's the kind of architectural shift that needs a documented decision record so future contributors don't redesign it from scratch.

## What to write

### A. Architecture Decision Record

`docs/adr/00<next>-multi-source-motion-targets.md` covering:

- **Context**: existing `ClubTarget`-only pipeline; user requirement for full-body and ball-aware matching.
- **Decision**: three frozen dataclasses + `MultiSourceTarget` aggregator + format-agnostic dispatchers.
- **Alternatives considered**:
  1. Single super-dataclass with optional fields — rejected: validation-rule explosion, encourages partial loads.
  2. ABC + multiple Target subclasses — rejected: forces inheritance on dataclasses that should stay flat & frozen.
- **Consequences**: cost-function code now reads `MultiSourceTarget` and dispatches on `has_*()`; loaders are independent and composable; renames cleared the source-revealing identifier surface.
- **Validation strategy**: post-init validation in every dataclass; dispatcher refuses unsupported extensions.
- **Generic-naming policy**: file-on-disk names can be specific; identifiers in code, directories under `src/`, and user-visible UI strings must not name a specific source/lab/person.

### B. User guide

`docs/user_guide/motion_matching/loading_targets.md`:

- How to load club data from xlsx (`load_club_target(path, sheet=...)`) / `.mat` (auto-dispatch) / `.c3d` (auto-dispatch).
- How to load body markers from `.c3d` (`load_body_target(path, opts=opts)`).
- How to share a clock between body and club (pass `impact_source`).
- Worked example using `data/C3D_TA_Driver.c3d` end-to-end with code excerpts.

### C. Update the motion-matching index

`docs/motion_matching/README.md` (or create if missing): table of contents linking to:

- The loaders ADR (above).
- The CLUB_IK_SPEC.
- The starting-pose-matcher README.
- The new user guide.

### D. Update AGENTS.md

`AGENTS.md` "shared infrastructure" list — add:

- `BodyTarget`, `ClubBallTarget`, `MultiSourceTarget` dataclasses location.
- `load_body_target`, `load_club_target` dispatchers.
- `default_body_segments` helper.

So future agents know to **reuse** these instead of reinventing.

## Generic naming

Every doc string and example uses generic terminology. The user-guide example file path is a concrete C3D file from the repo, but the explanatory text frames it as "a representative full-body mocap capture", not "the X dataset".

## Acceptance criteria

- [ ] ADR in `docs/adr/` with the next sequential number, following the existing ADR template style (see `docs/adr/0004-launcher-provider-migration.md`).
- [ ] User guide present, code examples copy-paste-runnable from a fresh checkout.
- [ ] Motion-matching index links the new pages.
- [ ] AGENTS.md updated with the new shared-infrastructure entries.
- [ ] No source names in any doc.

## Files touched

- New: `docs/adr/00<next>-multi-source-motion-targets.md`
- New: `docs/user_guide/motion_matching/loading_targets.md`
- Edit: `docs/motion_matching/README.md` (or new)
- Edit: `AGENTS.md`

## Sequencing

Lands last — after the implementation issues so the docs reflect what shipped, not what was planned.
