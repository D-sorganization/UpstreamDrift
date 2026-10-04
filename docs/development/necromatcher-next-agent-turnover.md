# Necromatcher Next-Agent Turnover

## Stop Condition and Checkpoint

The user requested a committed handoff checkpoint and instructed this agent to
stop pursuing the goal afterward. Do not resume automatically. The full goal is
unfinished; resume only when the user authorizes continued work.

Implementation checkpoint: `f0e149443ad4b4f58c17dce44accd233ca42cdca` on
`feat/necromatcher-native-fit-11235`, in
`C:/Users/diete/Repositories/Worktrees/UpstreamDrift-necromatcher-native-fit`.
The documentation commit containing this guide follows that checkpoint.
[PR #11359](https://github.com/D-sorganization/UpstreamDrift/pull/11359) is open
against `main` and was observed as **draft**. Do not merge, mark ready or claim
deployment from this handoff. No native export, fit, transfer or worker was
started during preparation of this guide.

The governing epic is [#11232](https://github.com/D-sorganization/UpstreamDrift/issues/11232).
Current local research integration is [#11493](https://github.com/D-sorganization/UpstreamDrift/issues/11493).
Overall interface parity remains [#11234](https://github.com/D-sorganization/UpstreamDrift/issues/11234);
historical fitting remains [#11235](https://github.com/D-sorganization/UpstreamDrift/issues/11235).
Scope/restriction evidence is recorded under #11414 and #11450.

## Read Before Making Changes

Read repository `AGENTS.md`, `CLAUDE.md`, nearest directory instructions,
[Agent Context Usage](../agent_context/USAGE.md) and the
[Shared Infrastructure Directory](../agents/shared-infrastructure.md).
Read the relevant provider, public facade, consumer and tests before coding.
Do not rely on a stale generated graph or renew an old boundary review merely
because tests pass. Newly added seams may require direct source inspection.

Before an issue fix, check its claim, post an individual lease and register an
individual session through Repository_Management. Run its coordination inbox
before scope expansion, commit and handoff. Recent inbox calls returned exit 2
with incomplete/malformed board evidence; that is **not** proof of no conflicts.
Retain the fleet fail-open policy and inspect issue/PR ownership. Do not reuse
this agent's session identity or delete another agent's claim.

Use topic branches and normal commit/push hooks. Preserve all user and peer
changes. All new behavior needs meaningful RED → GREEN tests, explicit boundary
contracts, public owners and small functions. Do not duplicate native FK,
projection, flight, contact conversion, storage or session lifecycles.
Repository budgets are 80 lines per function, five arguments, 1,200 production
lines and 400 test lines unless an applicable documented exception exists.

## What Exists and What Is Still Missing

| Requirement          | Current Evidence                                                                                        | Remaining Work                                                                |
| -------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Historical Workspace | Native tile and React route share a persistent player/swing/version Library                             | Final actual-player navigation/parity acceptance                              |
| Footage Review       | Original-frame review, skeleton and translucent model-proxy overlays                                    | Improve the actual matches; proxy overlays are not anatomical qualification   |
| Historical Fits      | Stored Tiger/Hogan research trajectories and explicit scope/camera/model declarations                   | Camera/anatomy/contact/continuous ROM/dynamics qualification                  |
| Profiles and Replay  | Source-bound authored replay and effort-profile import/recall                                           | Validate historical driving profiles and their physical interpretation        |
| Impact               | Authenticated saved replay, bounded clean worker, real impact/Rust flight, receipt and four-file export | Historical contact and physical source-time qualification                     |
| Local Golf           | Explicit authenticated MODEL_CONTACT preparation, arm/submit and retained samples in both hosts         | Restart-persistent local results, avatar animation and course integration     |
| External Golf        | Existing qualified-contact and destination gates remain intact                                          | No unverified research delivery to external destinations                      |
| AffineDrift          | Earlier handoff/deployment cohort is separate                                                           | New sanitized publication must be separately verified; no new deployment here |
| Reports and Visuals  | Compiled reviewed LaTeX supplements, synthetic evidence and historical research stills/videos           | Latest optimized Tiger full video has not been exported                       |
| ControlTower         | Earlier deliveries are historical evidence only                                                         | No new delivery; resolve existing access/approval constraints with the user   |

## Exact Evidence Boundaries

Tiger matching is original frames **0–190**, half-open `[0,191)`. Exclude the
uncertain release transition and released-right-hand follow-through from frame
191 onward. Full source browsing may preserve excluded footage; a recalled fit
or export may not project excluded frames. The exact physical release time was
not measured. Hogan remains `[0,750)`.

The local historical Library exists at:

```text
C:/Users/diete/AppData/Local/upstream-drift/upstream-drift/launcher/necromatcher
```

Its index is **`project.json`**, not `index.json`. Confirm the configured store
on the next host, then use the public `NecromatcherLibrary` for admission and
mutation. Never retarget a stale helper or create a replacement index by hand.
The following IDs were located in the current index during this handoff:

- `tiger-both-hands-lossless-fit-v22`
- `hogan-forearm-analytic-control-fit-v17`
- `hogan-forearm-analytic-variant-fit-v17`

Tiger V22 was produced at `263b478989d96e395a061cf34c23b1a791b81087`.
Its weighted dense body RMS decreased from 24.082673308003265 to
23.57224248027222 px on the same 191-frame domain. It reached 120 evaluations,
did not converge and remains **REJECTED research**. The three reviewed stills
are frames 0/150/190; frame 150 has substantial mismatch. The existing complete
191-frame video is the **unoptimized restricted seed**, not this optimized fit.
Hogan V17 also remains rejected; its shaft mismatch cannot be treated as
anatomical calibration. Preserve earlier cohorts and metric definitions.

Authored replay seconds are not calibrated historical physical time. All current
impact/local research qualifications remain unverified. A successful worker or
delivery receipt proves execution, not a qualified Tiger/Hogan model. Saved
impact environment and new local-default simulation environment are distinct.
The manual governance check passes structurally with two QMD sources and zero
registered calculations, while release stays `blocked-inventory-required`.

## Reusable Owners

| Task                                 | Public Owner or Entry Point                                                               |
| ------------------------------------ | ----------------------------------------------------------------------------------------- |
| Stored Fit Admission                 | `workspace.load_native_fit_binding`                                                       |
| Scoped Still/Video Composition       | `workspace.export_fit_stills`, `workspace.export_fit_video`                               |
| Bounded Video Jobs and Recall        | `workspace.NativeVideoSession.submit`, `stored_runs`, `view_for_fit`, `download`, `close` |
| Authored Replay Impact               | `workspace.NativeImpactSession`                                                           |
| Portable State/Trajectory Receipt    | `workspace.load_replay_impact_receipt`                                                    |
| Library-Authenticated Golf Admission | `workspace.load_research_impact_shot`, immutable `ResearchImpactShot`                     |
| Explicit Local Research Lifecycle    | `GolfSessionService.prepare_research_shot`, ordinary arm/cancel/submit                    |
| Detached Confirmed Local Samples     | `GolfSessionService.get_local_trajectory_record`                                          |
| Proper Aim Transform                 | Existing spatial transformation and post-impact launch conversion                         |
| Web Flight View                      | Existing `BallFlightScene3D`                                                              |

The bridge authenticates the four saved files and full parent chain, preserves
complete velocity/spin and applies a declared proper rotation. The API checks
unused shot identity both before and immediately after awaited admission.
No event-loop yield occurs between the second guard and preparation/context
publication. Do not remove this guard, replace MODEL_CONTACT with manual input,
promote qualification, infer launch speed from plotted samples or use demo IDs.
Current local golf context is bounded session data, not restart persistence.

## Bounded Work Packets for the Next Authorized Agent

### A. Complete the Current Optimized Video Evidence

This is the next suggested bounded packet; it was **not executed** here.
Read the [Camera and Scope Procedure](necromatcher-conditional-camera-and-scope.md)
and latest V22 still receipts first. Admit V22 through the public Library owner,
confirm its exact scope and stored curve, then use the existing public video
owner. Retain opacity 0.35 proxy shapes, skeleton, original footage, observation
roles and ordinary rejected-research captions. Do not display a seed caption
for the optimized fit or imply convergence.

Capture fresh source/Library/runtime maps, use a new output directory, and keep
the existing bounded worker budget. Poll the same confirmed-live handle;
timeouts in observation do not authorize a second operation. Verify all 191
decoded frames, source sizes, ordered identities, rational PTS and output hashes;
compare selected PNGs against the accepted still layout. Review representative
original/overlay pairs and report the visual-review scope explicitly. Inspect
the full residual record to identify where the match fails; do not substitute
display-marker RMS for weighted fit RMS. Save Desktop copies and a provenance
receipt. Add Hogan's latest videos only as a separate, independently admitted
packet after confirming its current fit/domain and budget.

Old helper configurations in `Repositories/Temp` pin older source/runtime
cohorts. They are reference material, **not runnable authorization or fresh
baselines**. Do not flip a disabled/ready flag or silently change their pins.
Existing render/verification tests should precede the actual historical run.

### B. Improve Matching From the New Evidence

Read [Camera/Morphology/Shutter Review](necromatcher-camera-morphology-shutter-review.md),
[Arm Morphology Admission](necromatcher-arm-morphology-admission.md),
[Shaft Trials](necromatcher-shaft-trials.md) and
[Fragment Diagnostics](necromatcher-shaft-fragment-diagnostics.md).
Choose one explicit hypothesis and its falsifying holdout test. Keep source
clock, camera, anatomy, observed shaft and assumed contact distinctions intact.
Use existing analytic marker/projection owners and stored exact spline starts.
Do not compensate for unexplained shaft pixels by adopting new body dimensions.
Consult the existing primary references before designing shutter measurements;
readout distortion, shaft deformation and transcoding must remain distinguishable
uncertainties. Require same-domain comparisons, sampled/continuous constraint
reports and selected original/overlay evidence. No automatic retry, budget
increase or silent replacement of a rejected baseline.

### C. Persist and Recall Local Research Golf Results

Define a source-bound schema and public Library owner for the new local result,
its simulator settings/environment, source impact hashes, aim and exact SI
samples. This owner does not exist yet. First write tests for restart recall,
tampering, foreign parents, detached data, cancellation/incomplete publication
and exclusive output. Reuse canonical atomic storage and admission patterns;
do not persist the API's weak context map as an authoritative result. Saved
source impact and newly simulated local flight must remain separate objects.
Integrate both hosts only after the public owner passes those contracts.

### D. Add Actual Avatar/Course Integration

Determine the real supported local destination and its public pose/course
contracts before implementation. Current `LocalReferenceAdapter` declares no
native avatar or course feedback. A plotted ball flight or trajectory table is
not completion of this packet. Preserve qualification and explicit controls;
implement source-bound pose/clock/profile handoff and actual playback/course
acceptance with real behavior. Do not change capability flags to pass a test.
Track a dedicated child issue with concrete acceptance before starting.

### E. Final Integration and Publication Acceptance

Exercise actual stored historical records through native and web navigation,
fit recall, variable-opacity overlays, profiles/replay, checked export, impact
and the eventual golf handoff. Capture stale/foreign/cancel/restart cases and
actual-player form behavior. Update #11234 only when that full scope passes.
Sanitized AffineDrift publication and ControlTower delivery remain distinct
actions with their own receipts. Do not retry previously rejected transfer or
agy actions through alternate routes without renewed human authorization.

## Validation Commands and Known Results

Run from the worktree, with `PYTHONPATH` containing the root and `src`.
Use `QT_QPA_PLATFORM=offscreen` for the Qt test lane. On this host the installed
Python is `C:/Users/diete/AppData/Local/Programs/Python/Python313/python.exe`.
The repository hooks use Ruff 0.15.17 and mypy 1.13 with:

```text
mypy --config-file pyproject.toml --follow-imports=silent <changed production owners>
pytest tests/unit/api/test_golf_research_routes.py -q
pytest tests/unit/workspace/test_necromatcher_golf.py tests/unit/golf_simulator/test_research_session.py -q
pytest tests/unit/tools/golf_simulator/test_research_dialog.py -q
pytest tests/unit/workspace/test_necromatcher_video_scope.py tests/unit/workspace/test_necromatcher_video_shape_jobs.py tests/unit/workspace/test_necromatcher_caption_export.py -q
```

Retain repo marker/config options; ordinary defaults can deselect integration
tests. The opt-in `test_necromatcher_golf_native_handoff.py` must run with MuJoCo
imported first, actual installed Rust and a fresh temporary canonical Library.
Its API override substitutes only the temporary Library and local-client gate;
physics, worker, parser, stamps and transport remain real. Existing OS font
bootstrap is test-only and does not install fonts. Whole-test cap 120 s, phase
waits 90 s and native worker budget 60 s are separate limits. Do not rerun it
unless a relevant change or unresolved concern justifies the operation.

Evidence already obtained, without adding overlapping counts:

- Integration checkpoint: 209 focused Python cases, 106 React cases,
  configured seven-owner mypy, 13-file pinned Ruff/budgets and 11 atlas cases.
- First real API/Qt acceptance: one pass, 23.07 s, seven retained samples.
- Subsequent identity-race regression: all 10 API cases pass, configured
  changed-owner mypy passes; fresh real acceptance passes in 24.56 s.
- Latest native before/after maps match over 6,697 src/native-test files;
  this is not a whole-repository baseline. Both screenshots were reviewed.
- A bare mypy attempt exposed 179 unrelated transitive errors; it is preserved
  as failed evidence. Configured checks and normal production push hooks pass.
- Exact-head remote snapshot at f0e14944 has zero checks, status contexts and
  Actions runs. Remote CI is unregistered/unverified, not green. Do not poll an
  unchanged snapshot or claim that local hooks replace remote acceptance.

## Evidence and Handoff Locations

The tracked starting points are the
[Local Research Procedure](necromatcher-local-research-golf.md),
[Integration Review](historical_capture/local-research-golf-review.json),
[Admission Race Review](historical_capture/local-research-golf-admission-race-review.json),
[Effort Profiles](necromatcher-effort-profiles.md),
[Replay Procedure](necromatcher-replay.md),
[Impact Procedure](necromatcher-replay-impact.md) and the chronological
[Turnover Record](necromatcher-turnover.md). Read this focused guide before the
long historical turnover; retrieve only the packet's relevant sections.

Local Desktop artifacts are under
`C:/Users/diete/Desktop/Necromatcher Review 2026-10-01`.
`Local Research Golf Supplement 1` contains the reviewed three-page LaTeX/PDF,
raw synthetic evidence and `publication.json`; `Admission Race Follow-Up`
preserves the second run separately. These screenshots are not historical
Tiger/Hogan matches. Historical V22 originals/optimized stills are in
`Tiger Both-Hands Strict Lossless Fit Stills V22`. Keep earlier reports and
unoptimized videos intact. The final documentation publication receipt will be
saved beside the supplement, identifying the handoff commit and goal pause.

## Economical Agent Workflow and Turnover Contract

Use one bounded packet per agent, a disjoint ownership list and one governing
issue. Assign lower-cost agents source inventory, deterministic test fixtures,
documentation, serialized evidence checks and packaging. Reserve native runs
and scientific acceptance for a reviewed, explicit protocol. Parallelize only
independent read/test work; do not run competing Library writers or shared
source edits. This turn's agents exhausted usage; no new agent dispatch is
necessary to read this handoff. Do not retry agy through a workaround.

Every packet should leave: the exact problem/acceptance, meaningful RED and
GREEN logs, source/parent/runtime pins, actual process terminal status, artifact
hashes and clearly bounded review scope, issue/PR links, normal committed/pushed
code, updated procedure/SPEC/turnover, and the next unresolved requirement.
Compile and inspect affected LaTeX report pages only when their source changes.
Do not redefine the full goal around a passing software subset or label an
unqualified historical hypothesis complete. Stop at the user-requested handoff
when all owned work is committed and the guide is current.
