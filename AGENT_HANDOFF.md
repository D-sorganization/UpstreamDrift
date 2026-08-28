# Agent Handoff: Proximal–Distal Research Program

Updated: 2026-08-28

Epic #8557 is the canonical completion authority. Issue state, local files, and
partial campaign checkpoints are not completion evidence.

## Protected Authority

- UpstreamDrift remote `main` is
  `85cce4d3307bb7ad3953d9fc6e583e370803515c`, the protected squash of
  #9152 for #9151.
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

## Active Publication Integrity Slice: #9142 / PR #9147

- #9142 prevents the generated claim-adjudication chapter from emitting
  repository-relative `data/` links that break when AffineDrift republishes the
  chapter. PR #9147 owns the implementation; do not create a duplicate PR.
- Worktree: `UpstreamDrift-worktrees/9142-portable-links`.
- Local branch: `fix/9142-portable-links`, tracking the PR branch through a
  merge-only reconciliation with current `origin/main`; never force-push.
- The source now emits portable UpstreamDrift `blob/main` links, while the
  AffineDrift publication boundary remains responsible for immutable SHA
  rewriting. The focused five-test generator/committed-artifact suite passes.
- Release manifests were regenerated after reconciling #9151. Complete serial
  validation, commit the merge, push to the existing PR branch, preserve squash
  auto-merge, inspect actionable hosted failures, and verify its squash on
  remote `main` before closing this slice.

## Immediate Order

1. Finish and protected-merge #9147 without changing scientific content.
2. Register the next #8557 child for matched forward impulse/work attribution
   through contact transitions. Its contract must freeze event surfaces,
   contact-state matching, impulse/work estimands, shaft/base coupling,
   uncertainty, adverse loads, killswitches, and evidence promotion rules.
3. Implement that child from a clean leased worktree only after its issue and
   dependency order are visible in #8557.

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
python -m pytest -n 0 -q tests/research/test_articulated_drift_contact_attribution.py
python -m ruff check scripts/research/proximal_distal_energy/articulated_drift_contact_attribution.py scripts/research/proximal_distal_energy/run_articulated_drift_contact_attribution.py tests/research/test_articulated_drift_contact_attribution.py
python -m ruff format --check scripts/research/proximal_distal_energy/articulated_drift_contact_attribution.py scripts/research/proximal_distal_energy/run_articulated_drift_contact_attribution.py tests/research/test_articulated_drift_contact_attribution.py
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
