# Agent Handoff: Proximal–Distal Research Program

Updated: 2026-08-27

Epic #8557 is the canonical completion authority. Issue state, local files, and
partial campaign checkpoints are not completion evidence.

## Protected Authority

- UpstreamDrift remote `main` is `10a247ba23df55537933f072d2e9adddae08b7a4`.
- PR #9136 merged the nonlinear reaching/control and Swing Objective Lab stack
  as `1b867e9da38dc6c3a321e0f7c199586fff5cf7be`.
- The qualified paper has 251 pages and SHA-256
  `92bfaca850ac459cc431e573be8c0288af51ceab4d28759d02c67c602274ee8b`.
- Its 325 claims, 144 numeric contracts, 498/498 governed literals, and 693
  release artifacts are computationally qualified. The PDF is still untagged
  and retains Type 3 and two unembedded resources, so it is not archival.
- AffineDrift PR #3991 pins the exact #9136 authority; protected squash/current
  remote main is `17a6a65b8c95145fc56d90af8c139aa3b049c5b6`.
- Tools R14.6 registration/acceptance, calibrated renderer, and extension map
  merged through protected squashes `b2d7f721`, `d7a95e2a`, and `da0759c7`.
  The trusted rendered-evidence run remains capacity-pending; do not rerun it.

## Active Scientific Slice: #9151

- Issue: #9151, `Research: Qualify Articulated Drift and Contact Attribution`.
- Worktree: `UpstreamDrift-worktrees/9151-articulated-drift-attribution`.
- Branch: `feat/9151-articulated-drift-attribution` from current remote main.
- Objective: carry the formal same-state configuration, velocity, contact, and
  applied-input decomposition into all 234 subject-scaled articulated states.
- Pure Python 3.12 contracts are implemented test-first and pass eight focused
  tests. They close acceleration and power exactly, provide coordinate-invariant
  mass-metric shares, report cancellation, and suppress inadequate ratios.
- The native atlas binds those contracts to MuJoCo and robotics Pinocchio, zero
  applied input, contact/velocity/gravity killswitches, coordinate scaling,
  coincident/reversed geometry, and a corrupted-force sentinel.
- Native Windows reproduction is not qualified: PyPI `pin` lacks a usable
  Windows wheel in the project environment, and an isolated conda-forge attempt
  exposed binary incompatibility. Do not commit or treat those local runtime
  attempts as evidence.
- PR #9152 is open with squash auto-merge enabled. Resolve its current exact
  head from `git rev-parse origin/feat/9151-articulated-drift-attribution`
  before acting. Ubuntu native-evidence run `33139951819`, job
  `98748338285`, passed from exact head `cb8af2cd846da351fdfa5c4bbee15f9ee6e62ad9`.
  Its downloaded JSON/NPZ reports MuJoCo 3.12.0, Pinocchio 4.1.0, 234 states,
  zero failed engine-states, and all registered gates passed. Independent NPZ
  inspection found every numeric array finite and every per-state gate true.
- The transient path-scoped generation workflow was removed after that
  successful run because repository policy prohibits active-workflow growth and
  direct hosted-runner routing. The stable public runner, figure, registration,
  README commands, committed evidence, and standard-CI evidence tests remain;
  do not recreate a one-off workflow for the next tier.
- The evidence, four-panel PDF/SVG figure, chapter, claims PD-CLAIM-327--329,
  data dictionary, release preset/claim, release review, and evidence/release
  tests are now integrated locally. The optimized paper has 252 pages and the
  new section/figure were visually inspected on pages 148--149 without clipping
  or an intervening float. The final 702-artifact release, checksum, and
  2,495-reference/419-local-artifact claim-evidence manifests validate locally;
  computational PDF
  qualification renders all 252 pages and passes, while tagged-PDF and font
  resources remain explicitly reported archival gaps.
- Architecture remains green through a narrow #9151 exception for the
  hash-bound ten-parameter native control evaluator. It expires 2026-09-30 and
  must be removed when the next forward-attribution slice can regenerate native
  evidence after introducing a cohesive evaluator context.
- The four tests in `test_articulated_drift_contact_attribution.py` remain in
  the suite-marker baseline because that file is hash-bound to the successful
  native evidence. Mark them `scientific` and remove those baseline entries only
  with the next native regeneration; do not rewrite the recorded source hash.
- Do not close #9151 until this complete publication slice passes protected CI,
  squash-merges, and the merge commit is verified on remote main.

## Immediate Order

1. Inspect exact protected checks on PR #9152's current remote head, fix only
   actionable source failures, preserve
   squash auto-merge, and verify the squash commit on remote main.
2. Refresh AffineDrift only from that immutable qualified UpstreamDrift squash,
   then verify its protected merge and public paper links.

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
