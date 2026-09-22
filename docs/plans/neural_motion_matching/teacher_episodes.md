# NM-04 Teacher Episodes and Active-Learning Candidates

Governing issue: [#10619](https://github.com/D-sorganization/UpstreamDrift/issues/10619)
(epic [#10603](https://github.com/D-sorganization/UpstreamDrift/issues/10603)).

Schemas: `neural-teacher-episodes/1.0.0`, `neural-acquisition-log/1.0.0`

## What Landed

Feasible teacher generation and active acquisition under
`src/shared/python/neural_motion/teachers/`:

- `TeacherSpec` / `TeacherOutcome` with DbC (finite duration, allow-listed
  channels, non-empty identity strings)
- `TeacherEpisodeGenerator` — near-baseline, stratified, low-discrepancy and
  random-torque comparison paths over a qualified baseline `EpisodeRecord`;
  native reproducibility via content hash; zero requested channel cannot pass
  as available
- `RejectionLedger` and separate quarantine vs reject roles (outliers retained,
  not clipped into the accepted corpus)
- `NestedTeacherCorpus` — nested stage sizes from NM-01, compute caps,
  checkpointed resume, duplicate-seed avoidance, nested reuse across stages,
  learning-curve subsets with ancestry
- `ActiveLearningAcquirer` — uncertainty / coverage / random-control strategies;
  fails closed when the candidate pool is entirely in the test split

Reuses NM-03 `EpisodeStore` / `EpisodeRecord` / `FamilySplitPlan` and NM-01
`NESTED_EPISODE_STAGES` contracts. No parallel episode store or trainer.

## Evidence

[`evidence/nm04_teacher_episodes_receipt.json`](evidence/nm04_teacher_episodes_receipt.json)

## Limitations

Software-contract fixtures only. Synthetic rollouts exercise contracts; they
are not native teacher qualification or training evidence. No NM-05+ training
or speed claim.
