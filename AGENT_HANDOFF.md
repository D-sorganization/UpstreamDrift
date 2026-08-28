# Agent Handoff: Proximal–Distal Research Program

Updated: 2026-08-28

Epic #8557 is the canonical completion authority. Issue state, local files, and
partial campaign checkpoints are not completion evidence.

## Protected Authority

- UpstreamDrift remote `main` is
  `e732757c90538acae7d7c4531dc1a05dc321b94f`, the protected squash of
  #9147 for #9142. It contains the #9152/#9151 authority.
- The qualified paper has 252 pages and SHA-256
  `0527465cd0bf6b69c4ae4c541986b0aecad28d5c05dfdabdbda0d493433e19ec`.
- Its 328 claims, 498/498 governed literals, 702 release artifacts, 2,495
  evidence references, and 419 local evidence artifacts are computationally
  qualified. The PDF remains non-archival because tagged-PDF and font-resource
  gates are still open.
- AffineDrift PR #3993 pins the exact #9152 authority as protected squash
  `9b9cbcc2199f1fbf8cd281beb08c57d543b552b1`; handoff correction #3995
  merged as `6cc909273d63147392b17078a35c6c4da034e1da`. All hosted checks passed.
- Tools PR #4669 merged as `f9730033fd279ba8b4abe03bab2aadd950400b47`;
  UpstreamDrift #8358 is closed after protected consumer integration. Tools
  #4142 remains the broader reusable-variation completion authority.
- Tools R14.6 registration/acceptance, calibrated renderer, and extension map
  merged through protected squashes `b2d7f721`, `d7a95e2a`, and `da0759c7`.
  The trusted rendered-evidence run remains capacity-pending; do not rerun it.

## Active Scientific Slice: #9153

- #9153 is the leased #8557 child for event-aligned forward impulse/work
  attribution. Lease owner: `codex`, session
  `019fe886-6614-70a2-a596-e5b0dea725d0`.
- Worktree: `UpstreamDrift-worktrees/9153-forward-impulse-work`.
- Branch: `feat/9153-forward-impulse-work`, created cleanly from protected
  `e732757c9`; three local commits are saved through `e41105424`.
- The pure kernel separates continuous generalized-force impulse/work,
  independently evaluated `Mdot v` momentum transport, and registered event
  impulse/work. Duplicate event times are integrated as separate segments.
- Eight manufactured tests cover constant force/work, variable mass transport,
  event separation, coordinate scaling, malformed topology, mass-rate
  differentiation, and planted-force corruption. A ninth MuJoCo test replays a
  five-sample rigid articulated contact trace into configuration, velocity,
  contact, and zero-active contributions with exact pointwise force closure.
- No branch has been pushed and no PR exists yet. The current work is an
  implementation foundation, not completion of #9153.

## Immediate Order

1. Add denominator adequacy and cancellation reporting without inventing an
   invariant impulse fraction; retain coordinate-wise units/scaling.
2. Add exact event records and active-set segmentation adapters for the
   distributed-grip opening/reattachment and stick/slip authorities.
3. Add matched rigid/shaft/base branches, causal forward killswitch runs kept
   separate from same-trajectory attribution, refinement/parity/adverse cases,
   and a serial smoke manifest.
4. Generate JSON/NPZ evidence, figures, paper/claim/reviewer integrations, and
   release manifests; then run full governed gates and protected delivery.

## Scientific Boundaries

- The #9151 decomposition is pointwise. It does not establish forward
  persistence, impulse/work attribution, ZVCF, biological passivity, muscle
  action, participant behavior, timing economy, safety, or coaching strategy.
- Energy transfer, momentum redistribution, joint work, contact power, event
  timing, and clubhead speed remain distinct estimands.
- Native-engine agreement verifies the declared operators and common contact
  law; it does not independently calibrate anatomy, grip, shaft, or ground.
- Ratios below the registered denominator floor are suppressed, not reported as
  zero. Signed shares may be negative or exceed one under cancellation.
- The next gate is matched forward impulse/work attribution through contact
  transitions, shaft/base coupling, uncertainty, and adverse loads.
- #8556/#9004 remain governed human-data boundaries. Synthetic evidence cannot
  substitute for bilateral six-axis participant grip wrenches.

## Frozen External Boundary

- #8800 remains frozen at source
  `1bd4d57da7bd257b76b42b3cc19524b283b5f748`; only 93/830 checkpoints exist.
- ControlTower ground stopped at 45/48 and shaft at 48/48. Its WSL VHDX is
  unreadable (`0x80070570`). Do not retry WSL, repair/mount/copy/mutate the
  VHDX, restart services, or launch a replacement without explicit approval
  and a recoverability plan.
- DeskComputer remains runner-drained. Keep local tests serial and web tests at
  no more than two workers.

## Validation

Use `C:\Users\diete\AppData\Local\Programs\Python\Python312\python.exe` for
portable tests. Run Python tests with `-n 0`.

```powershell
python -m pytest -n 0 -q tests/research/test_articulated_forward_attribution.py
python -m ruff check scripts/research/proximal_distal_energy/articulated_forward_attribution.py scripts/research/proximal_distal_energy/articulated_rigid_forward_attribution.py tests/research/test_articulated_forward_attribution.py
python -m ruff format --check scripts/research/proximal_distal_energy/articulated_forward_attribution.py scripts/research/proximal_distal_energy/articulated_rigid_forward_attribution.py tests/research/test_articulated_forward_attribution.py
python scripts/check_document_title_case.py --changed-from origin/main
python scripts/ci/check_file_size_budget.py
python scripts/ci/check_architecture_budget.py
```

Also run claim/evidence integrity, release qualification, PDF build/inspection,
and affected full gates after publication artifacts are added. Run the GitHub
App setup script immediately before every GitHub operation. Never force-push,
bypass protection/review, relax scientific tolerances after inspecting results,
or create redundant CI reruns.

## Other Repository Programs

- UP-D0 (#9066) establishes `manuals/upstreamdrift` QMD as the sole editable
  engineering-manual authority; generated manual formats are non-editable and
  unapproved. UP-D1 (#9067) must inventory and classify every in-scope
  calculation and module before any manual coverage or release claim.
- Markerless mocap still follows Tools provider -> Upstream orchestration ->
  Affine sanitized projection. Camera candidates are not procurement or lab
  qualification.
