# Golfer Model Versions

`src/config/golfer_model_versions.json` names the current golfer model and the
parts it deprecates. Read it through
`src/shared/python/model_appearance/golfer_model_versions.py`
(`current_golfer_model()`, `deprecated_parts(spec)`), never by memory or by
copying an older spec.

## Current Model: `full_body_skinned_v1`

| Part | Spec Field | Current | Deprecated |
|---|---|---|---|
| Hands | `subject.hand_visual_backend` | `skinned_human` | `capsule` |
| Body | `subject.body_visual_backend` | `skinned_cc0` | `loft` |

The skinned hands carry the owner-approved grip: trail palm over the lead
thumb, ring finger beside the lead forefinger and a 30° trail trigger spread.

## Putting a Spec on the Current Model

1. Apply `apply_skinned_hand_wrists(spec)`
   (`motion_matching/execution/skinned_hand_wrists.py`). It moves the wrist
   joints to the skinned hands' wrists, sets `hand_visual_backend` and records
   a `spec_revision`. It is right-handed only.
2. Set `subject.body_visual_backend` to `skinned_cc0`.
3. Refit any pose fitted before the revision with `refit_arms_to_club`
   (`engines/physics_engines/mujoco/python/skinned_arm_refit.py`). It holds the
   club where the old pose had it and closes the trail hand on the grip.

Drawing an unrevised spec with skinned hands misplaces them. That is why a spec
without the selectors still falls back to the deprecated capsule hands and loft
body at draw time. Each fallback logs a deprecation warning once per process,
naming the registry.

## What Is Still Migrating (#15648)

- **Builders:** the spec builder, character builder and API still emit
  unrevised specs. They are next to move to the current model by default.
- **Committed specs:** every committed full-body spec, including
  `full_body_spec_anthro_driver.json` and `full_body_spec_anthro_iron7.json`,
  is legacy. Current-model reference specs will be committed beside them, under
  versioned names, and listed here.
- **Model Explorer demo:** the demo bundle is legacy and is being replaced.

Until a part has migrated, prefer a current-model source, such as a matched
swing on the skinned model, for renders and screenshots. When only legacy
sources exist, say so in the PR.

## Ratchet

`src/config/golfer_model_legacy_assets.json` lists every committed spec and
bundle still on a deprecated part. `tests/config/golfer_model_ratchet` fails
when a new spec or bundle uses a deprecated part, and when a listed file was
migrated or removed without deleting its entry. The list only shrinks.
