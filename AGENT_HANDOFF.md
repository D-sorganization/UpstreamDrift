# Agent Handoff: Proximal–Distal Research Program

Updated: 2026-08-27

Epic #8557 is the canonical completion authority. This worktree is the active
integration vehicle for issues #9124, #9125, #9126, and #9128. Do not infer
completion from local files or an open pull request.

## Protected Authority

- UpstreamDrift owns scientific sources, evidence, and the paper. Tools owns
  reusable mechanics and UI-neutral provider contracts. AffineDrift publishes
  immutable source-pinned projections.
- Protected PR #9144 completed #9123 at squash
  `5a330ed9b9f74c77a540d37beb90d2df622e719d`, verified on remote `main`.
- The current #9125-integrated paper is 250 pages and 1,958,661 bytes at
  SHA-256 `6ca47ab88331cbb728a0f464f1a1200cf16553328148f40e42192f66d56a1647`.
- It contains 322 adjudicated claims, 141 numeric contracts, 482/482 verified
  numeric literals, 2,426 evidence references, and 684 release artifacts.
- The PDF is computationally qualified but not archival-accessibility
  qualified: it is untagged and retains Type 3 and two unembedded resources.

## Active Integration

- Worktree: `UpstreamDrift-worktrees/9128-consolidated-research`.
- Branch: `feat/9128-swing-objective-web-parity-and-research`.
- Existing protected-flow PR: #9136. Preserve its squash auto-merge request;
  do not merge until every scientific and software gate below is satisfied.
- PR #9136 is being reconciled with protected #9123. Protected #9123
  trajectory-authority sources and tests take precedence over its earlier,
  incorrect fixed-horizon implementation.
- A clean prior #9124 worktree at
  `UpstreamDrift-worktrees/9124-bounded-event-reachability` contains the more
  complete event-replay and multiple-shooting implementation. Reuse only the
  #9124-owned files, then rebind them to protected #9123 source identities.

## Open Correctness Findings

- #9124 source, governed evidence, figure, claims, paper, and release bundle
  now enforce amplitude/rate limits, typed guard outcomes, and independent
  protected-RK4 replay. The study retains 32/38 feasible cases and a maximum
  feasible residual of `8.82244e-11`, but its 24.9517% multistart spread fails
  the 5% optimality gate; every channel/controller ranking remains suppressed.
- #9125 now has regenerated Phase A/B/C JSON and NPZ evidence, a reviewer
  figure, explicit global topology types, antithetic common-random-number
  perturbations, a fixed stress-to-failure ladder, four channel masks, and
  step/horizon controls. Its governed claims and release registration are
  locally complete; protected delivery remains pending in PR #9136.
- #9126 now rejects every solver identity except the implemented projected
  iLQR kernel; collocation NMPC is explicitly unavailable. Plant transport is
  rebound to the protected public backend. Full comparative evaluation,
  governed evidence, and every solver/controller ranking remain unavailable.
- #9128 now uses the mounted `/api/tools/...` frontend contract, generated
  OpenAPI types, and the current `WorkspaceShell` interface. The vendored Tools
  pin is advanced to protected provider `3dfbd32cc`; focused API and React
  parity tests pass, but full web/build and provider-consumer gates remain.
- The current consolidated branch has complete local paper/release integration
  for #9124 and #9125, but not #9126. The handoff, issue closures, and `parity`
  status must not overstate implementation, protected delivery, or validation.

## Required Order

1. Finish the merge from protected `main`, retaining protected #9123 sources.
2. Preserve #9124's governed multiple-shooting evidence and verified release;
   do not infer optimality or ranking from its local feasibility result.
3. Correct and qualify #9125 event-topology robustness, including delays,
   channel coverage, common-random-number perturbations, and typed failures.
4. Correct #9126 solver identity and keep prospective registrations, human
   limits, and rankings unavailable until the registered comparisons exist.
5. Correct #9128 API routing/types/shell contracts and test desktop/web parity
   without moving research authority into the UI.
6. Regenerate claims, numeric contracts, release artifacts, TeX, PDF, and
   reviewer figures only after the source results are final. Visually inspect
   affected pages and re-run the computational publication profile.
7. Shepherd #9136 through protected CI, verify its squash on remote `main`,
   close only actually completed issues, and update epic #8557.

## Scientific Boundaries

- Keep energy transfer, momentum redistribution, joint work, interaction-force
  power, event time, and clubhead speed distinct.
- Torque and torque-rate bounds are declared model scenarios, not measured
  human capacity. Synthetic traces do not establish participant behavior.
- Local linear authority, bounded nonlinear feasibility, topology robustness,
  controller qualification, and UI parity are separate evidence tiers.
- No current result establishes passive negative torque, muscle strategy,
  fatigue resistance, controller superiority, coaching efficacy, or a
  universal clubhead-speed prescription.

## Validation

Use Python 3.12 with
`PYTHONPATH=C:/Users/diete/AppData/Local/Temp/codex-precommit-wmi;<worktree>/src`.
Run Python tests serially (`-n 0`) and web tests with at most two workers.

```powershell
python -m scripts.research.proximal_distal_energy.run_trajectory_control_authority validate
python -m scripts.research.proximal_distal_energy.run_bounded_event_reachability validate
python -m scripts.research.proximal_distal_energy.claim_audit validate
python -m scripts.research.proximal_distal_energy.qualify_open_release validate
python -m pytest -n 0 -q tests/research/test_trajectory_control_authority.py tests/research/test_trajectory_control_authority_evidence.py
python -m pytest -n 0 -q tests/research/test_bounded_event_multiple_shooting.py tests/research/test_bounded_event_reachability.py tests/research/test_bounded_event_reachability_evidence.py
python scripts/check_document_title_case.py --changed-from origin/main
python scripts/ci/check_file_size_budget.py
python scripts/ci/check_architecture_budget.py
```

Also run Ruff, MyPy, API type generation/freshness, affected frontend tests,
provider-consumer contracts, release qualification, and responsive/rendered
inspection. Use the GitHub App setup script immediately before every GitHub
operation. Never force-push, bypass checks, alter protection, or add quarantine.

## Frozen External Boundary

- #8800 remains frozen at source
  `1bd4d57da7bd257b76b42b3cc19524b283b5f748`: 93/830 checkpoints exist.
- ControlTower's WSL VHDX remains unreadable (`0x80070570`). Do not retry WSL,
  repair/mount/copy/mutate the VHDX, restart services, or replace the campaign
  without explicit user approval and a recoverability plan.
- DeskComputer remains runner-drained. Do not run large parallel campaigns or
  reactivate its runners.
