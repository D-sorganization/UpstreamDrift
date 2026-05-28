# AGENTS.md — sg_optimizer (shared)

Before touching code here, re-read the four binding pitfalls from
`docs/sg_optimizer/STROKES_GAINED_OPTIMIZER_SPEC.md` §6:

- **#1** Don't initialise V from baseline data; iterate via Bellman.
- **#6** Putting model stays separate from the swing model.
- **#11** Keep `bellman_backup_scalar` and `HoleMDP.bellman_backup` in sync; property tests assert this on small grids.
- **#14** Conditions are configurable, not part of the lie raster. Do not introduce `rough_light` / `rough_heavy` lie codes.

## Things to leave alone

- `LIE_CODES` integer values — public contract; downstream solvers rely on them.
- `_condition_modifiers` signature — `HoleMDP._modifier_table` indexes into it.
- Scalar Bellman implementation — it is the reference, not legacy.

## What's safe to extend

- New presets in `CourseConditions` / `RoughModel` / `GreenModel`.
- New baseline YAMLs in `data/sg_optimizer/baselines/`.
- Additional Hypothesis property tests under `tests/property/sg_optimizer/`.

## Coordination

Lease the parent issue (#6270 for Phase 1) before editing. Cross-phase changes
should touch the epic #6269 with a comment summarising the impact.
