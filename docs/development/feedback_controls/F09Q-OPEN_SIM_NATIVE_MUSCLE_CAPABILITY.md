# F09 OpenSim Variant Capability Turnover (#12148)

## Change

The legacy `create_muscle_model_variant` factory previously presented eight
target names as if they were executable actuators and activation states. It now
retains those names only as declarations and lazily reads the exact source model
through OpenSim when the SDK is installed. Native actuator/state names and
muscle-force, activation-dynamics and tendon-dynamics support come from the
initialized model inventory. Missing SDK is `unknown`; a successful load with
zero muscles is `observed_empty`. The shipped golf source currently reads back
zero muscles under OpenSim 4.6.

The viewer no longer enables a muscle layer from the declared target list.
`replay_controls` now reports `Controls_Validated_Not_Replayed`; this adapter
validates values and does not execute forward dynamics. The required inventory
row remains present and unsupported for the shipped zero-muscle model, keeping
the full parity denominator honest.

## TDD and Validation

The existing behavior was red in three ways: the variant omitted its declared
target count, a target control was accepted against an empty native muscle set,
and a validation-only call reported `Succeeded`. Regressions now check the
declaration/native distinction, reject zero-muscle controls, keep the view layer
hidden, and require the validation-only status. An independent one-muscle
OpenSim fixture proves that a real native muscle and its loaded state names are
exposed. A source-bound native test confirms that the shipped golf model reads
back zero muscles.

Focused suite results after reusing `_native_model_inventory.observe_native_model`:

- Default Python 3.13: 45 passed, 2 OpenSim-dependent skips.
- OpenSim 4.6 environment: 47 passed, including zero-muscle shipped model and
  one-muscle native fixture.

The run required a temporary local test-time junction for the uninitialized
`vendor/ud-tools` gitlink; the worktree gitlink was restored immediately after
the tests. This does not change dependency pins or claim the feature is merged.

## Scope Boundary

The observed source contains no native muscles. This correction does not create
muscles or qualify full-body anatomy, actuation physiology, resource closure,
contact, grip, or full-horizon native replay. The OpenSim runtime readback and
synthetic fixture are structural capability evidence only. Six-engine parity
and the all-model required denominator remain open. See canonical chapter 41,
the updated chapter 18 admission boundary, and chapter 13's uncertainty
section.
