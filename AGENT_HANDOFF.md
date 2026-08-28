# Agent Handoff: Proximal–Distal Research Program

Updated: 2026-08-28

Epic #8557 is the completion authority. Local artifacts and partial campaigns
are not completion evidence.

## Protected Authority

- UpstreamDrift protected `main` was `e732757c90538acae7d7c4531dc1a05dc321b94f`
  when #9153 began. It contains the #9152/#9151 authority.
- The qualified paper has 252 pages and SHA-256
  `0527465cd0bf6b69c4ae4c541986b0aecad28d5c05dfdabdbda0d493433e19ec`.
  Its computational evidence gates pass, but tagged-PDF and font-resource gates
  remain open, so it is not an archival accessibility release.
- AffineDrift #3993 pins #9152 at protected squash `9b9cbcc2199f1fbf8cd281beb08c57d543b552b1`;
  handoff correction #3995 merged as `6cc909273d63147392b17078a35c6c4da034e1da`.
- Tools #4669 merged as `f9730033fd279ba8b4abe03bab2aadd950400b47`;
  UpstreamDrift #8358 is closed. Tools #4142 remains the reusable-variation
  authority. R14.6 merged through `b2d7f721`, `d7a95e2a`, and `da0759c7`.

## Active Scientific Slice: #9153

- Lease: `codex`, session `019fe886-6614-70a2-a596-e5b0dea725d0`.
  Worktree: `UpstreamDrift-worktrees/9153-forward-impulse-work`; branch:
  `feat/9153-forward-impulse-work`. No PR exists yet.
- The event-aligned kernel separates continuous force impulse/work, independently
  evaluated `Mdot v` momentum transport, kinetic geometry work
  `0.5 v^T Mdot v`, and registered event impulse/work. It handles duplicate
  event times, coordinate scaling, cancellation, and denominator suppression.
- Distributed traces retain signed station gaps. Opening and reattachment roots
  are located on a declared linear state interpolant to the frozen gap/time
  tolerances. Stick/slip surfaces and distributed forward execution remain next.
- Plan revision 5 freezes source `9a91a957f793a0f2b1e891b7367624544ab33b88`,
  data SHA-256 `9fa4364571ba5535995c63226289c0711ee1ebf37c58b7a3b4e4d14a98561779`,
  42 rigid-smoke cases, the later 18-state screen, contact/model/frame/unit
  conventions, and separate descriptive-attribution/counterfactual estimands.
  Raw file SHA-256 is `2cd22750542d4cc5360ed37b5c4fe88a88e1e679dcd5c5c29d9739eaff253f1b`;
  canonical JSON checkpoint identity is
  `1b1601bc93596ed85d13eb6a310001a70551a084f05a372d9f4ad03a0e0ea466`.
- The diagnostic execution `0ba50aee3ab1fe1d445cd003e2428e048685d4f0`
  is preserved at `C:/Users/diete/Campaigns/UpstreamDrift-9153-rigid-smoke-0ba50aee3`.
  It exposed the omitted kinetic geometry work term; do not delete or relabel it.
- Corrected execution `65939421194a59db12d926537a687ab922891c61` is
  preserved locally and now published under
  `docs/research/proximal_distal_energy_transfer/data/articulated_forward_attribution_smoke/`.
  The directory contains all 42 exact atomic checkpoints plus a deterministic
  summary and SHA-256 inventory (`checkpoint_set_sha256`
  `1e2dfc947d63b4ef143602c67505b37fc2de2e2b7ea745c5cdb1a9b61096796c`).
- Corrected results: 21 MuJoCo cases complete, 21 Pinocchio cases retain typed
  native unavailability, zero execution failures. Every MuJoCo case passes its
  individual closure tolerances. Momentum refines in all seven variants. Work
  refinement fails nominal (`0.829 > 0.8`) and high damping (`1.235 > 0.8`).
  Promotion is false for refinement and unavailable native parity.
- The aggregator validates complete plan/source/data/execution/case/schema
  identities, retains every failure, inventories exact bytes, and refuses
  promotion if parity is unavailable or not yet evaluated. A freshness test
  regenerates the committed summary from the published checkpoints.
- Focused aggregation/runner tests, Ruff, format, and file-size gates pass.
  Existing focused MyPy behavior is not green: inherited scientific-script
  errors and an installed-MyPy internal crash were previously observed.

## Immediate Order

1. Commit and push the governed aggregator and complete smoke evidence.
2. Diagnose the two work-refinement failures without weakening frozen gates;
   add distributed stick/slip events and the distributed forward evaluator.
3. Add matched rigid/shaft/base branches and causal killswitch runs, keeping
   counterfactual outcomes distinct from same-trajectory attribution.
4. Execute refinement/parity/adverse cases serially, then generate governed
   figures, paper/claim/reviewer integrations, and protected delivery.

## Scientific Boundaries

- #9151 is pointwise and does not establish forward persistence, biological
  passivity, participant behavior, safety, timing economy, or coaching advice.
- Energy transfer, momentum redistribution, joint work, contact power, event
  timing, and clubhead speed are distinct estimands.
- Engine agreement would verify declared operators, not calibrate anatomy,
  grip, shaft, or ground. Synthetic evidence cannot replace #8556/#9004 human
  bilateral six-axis grip-wrench acquisition.
- Signed shares may be negative or exceed one under cancellation. Ratios below
  the registered denominator floor are suppressed, not reported as zero.

## Frozen External Boundary

- #8800 remains frozen at `1bd4d57da7bd257b76b42b3cc19524b283b5f748`;
  only 93/830 checkpoints exist. ControlTower ground stopped at 45/48 and shaft
  at 48/48. Its WSL VHDX is unreadable (`0x80070570`). Do not start WSL,
  repair/mount/copy/mutate the VHDX, restart services, or launch a replacement
  without explicit approval and a recoverability plan.
- DeskComputer remains runner-drained. Use serial local tests; web tests use at
  most two workers.

## Validation

Use Python 3.12 and serial pytest. Current focused commands:

```powershell
python -m pytest -n 0 -q tests/research/test_articulated_forward_attribution.py tests/research/test_articulated_forward_attribution_study.py tests/research/test_articulated_forward_attribution_runner.py tests/research/test_articulated_forward_attribution_summary.py
python -m pytest -n 0 -q tests/research/test_articulated_contact_events.py tests/research/test_articulated_distributed_grip.py tests/research/test_articulated_distributed_forward.py
python -m ruff check scripts/research/proximal_distal_energy tests/research
python -m ruff format --check scripts/research/proximal_distal_energy tests/research
python scripts/check_document_title_case.py --changed-from origin/main
python scripts/ci/check_file_size_budget.py
python scripts/ci/check_architecture_budget.py
```

Run the GitHub App setup script immediately before every GitHub operation.
Never force-push, bypass protection/review, relax tolerances after inspecting
results, or create redundant CI reruns.
