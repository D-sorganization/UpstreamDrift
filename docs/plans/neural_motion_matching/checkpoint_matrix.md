# NM-09: Checkpoint Matrix and Qualification

Governing Issue: #10624 (epic #10603).\
Schema: `neural-checkpoint-matrix/1.0.0`\
Card Schema: `neural-model-checkpoint/1.0.0`\
Receipt: [nm09_checkpoint_matrix_receipt.json](evidence/nm09_checkpoint_matrix_receipt.json)

> **DIAGNOSTIC ONLY.** The receipt has `status: DIAGNOSTIC` and
> `roster_coverage.qualified_native: []`. Its audit note says prior native
> replay and qualification statuses were simulated or lookup-derived without
> physical trained checkpoints or ODE rollouts. This record establishes no
> native qualification.

## Recorded Software Scope

The receipt records a software matrix schema, model roster coverage, and
validation command. Those records describe software-level artifacts only;
they do not establish trained-checkpoint replay or measured native dynamics.
The receipt classifies the driven pendulums and constrained golfer as
`unqualified`, reconstruction models as kinematic proposals, and models with
missing engine prerequisites as blocked.

## Evidence and Limitations

- Receipt: [nm09_checkpoint_matrix_receipt.json](evidence/nm09_checkpoint_matrix_receipt.json).
- The recorded matrix digest identifies the historical diagnostic artifact; it
  is not qualification evidence.
- Native replay remains unverified for the models listed in the receipt.
- Reconstruction models are kinematic proposals and do not produce physical
  joint torques.
- Missing optional engine runtimes remain explicit blockers.

## Next Action

NM-10 (#10625) has a separate historical diagnostic receipt. Its reported
benchmark estimates are not qualified results.
