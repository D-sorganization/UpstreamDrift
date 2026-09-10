# Capture Product Turnover

## Start Here

The user requested a handoff to a lower-cost agent. Stop expanding scope during
turnover. The overall product goal is **not complete**. Read this file before the
older chronological records in `HANDOFF.md`; GitHub and current Git state remain
authoritative. Do not recreate epics or replace existing implementations.

## Active Branches and PRs

| Work                                  | Location and Branch                                                                   | State at Turnover                                                                                                                                                                                                                                                        |
| ------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Installed runtime repairs             | `Worktrees/UpstreamDrift-parameter-bounds`, `fix/9949-installed-capture`              | PR #9950 merged as `9c8afeaabf60f2751ebbd61b32dac98d32546c3e`; local branch is retained.                                                                                                                                                                                 |
| Camera setup and guided help          | `feat/9952-camera-setup` at `7c7950574e47abcc42bad432c267d03031e62aa5`                | PR #9954 has a merge conflict after #9950 landed. Its required quality gate passed. Resolve the merge normally, preserve both sets of records, and rerun required checks.                                                                                                |
| Calibration and model revision status | `Worktrees/UpstreamDrift-common-calibration`, `feat/9899-calibration-revision-status` | PR #9959, based on #9954. Initial implementation `1229dd7aa` is published; final turnover commit adds model-lineage validation and CI size fixes. Use current HEAD for the final revision.                                                                               |
| Canonical ruler scale                 | `Worktrees/Tools-calibration-numerics`, `feat/5168-linear-reference-scale`            | Tools PR #5169 merged as `d4ab52a926cbd74d10b881a700c0c4f12f89728f`; implementation and API repair `3d7beb203a71dcfa47d77ebe9fa318d181924867`, followed by turnover documentation. The private Gasification checkout check still fails; merge alone does not qualify it. |

All paths above are relative to `C:/Users/diete/Repositories`. PR #9955's Bioptim
scalar-bound fix is already merged as `32410babfd1e4741fa0c53bf05dd8403a51bf233`.
Never force-push or push directly to main. Do not confuse the common-calibration
worktree's current branch with the retained camera-setup branch.

## Ordered Continuation

1. Inspect PR #9954's merge conflict against current main, resolve in a clean
   checkout of its branch, preserve both runtime and camera-setup changes, then
   publish normally. Merge only after current-head required checks pass.
2. Integrate that main into #9959 and qualify it. The previous CI failure was
   `reconstruct_session` exceeding 100 lines and the development log exceeding
   51,200 bytes. Local fixes extract measurement/lens helpers and shorten only
   this task's log entry; both budgets now pass. Do not increase the budgets.
3. Tools #5169 is merged, but still resolve its private consumer checkout access with its existing
   workflow owner. Job `102883109039`, run `34479121462`, failed while retrieving
   the Gasification repository's default branch with HTTP Not Found, before
   consumer tests. This is not a ruler numerical failure. Do not disable the
   consumer gate, expose tokens, or treat skipped tests as qualification.
4. Coordinate the merged Tools #5169 provider update: gitlink,
   `requirements-tools.txt`, Rust revision and generated source/context evidence
   must agree. Never edit `vendor/ud-tools` source as a substitute for the provider.
5. Complete the ruler consumer under #9899. Reuse the existing isolated
   `reference_calibration/worker.py`, client, saved revision, result review and
   source-evidence contracts. Add a distinct scale-estimation action starting
   from an established, reviewed layout. Show per-placement lengths, residuals,
   holdouts, repeatability and limitations; explicitly review and save a new
   revision. Preserve old results and invalidate downstream reconstruction/fit.
6. Test that full sequence in the installed application, then audit the open
   calibration, club and wizard epics requirement by requirement before closure.

## Ruler Method and Limits

`sidekick.lab.mocap.reference_scale.estimate_reference_scale` estimates one
positive scale from known two-endpoint lengths, using existing camera geometry
and matching pinhole lens/zoom profiles. Rotations and source records stay intact;
camera centers scale about an explicit anchor. Independent holdouts never enter
the fit. Rulers cannot initialize camera orientation. The existing four-point
pose-initialization guard must remain intact. Full provider details are in Tools
`docs/development/REFERENCE_SCALE.md`.

Paper, yardstick and metre-stick selection/repeated-placement infrastructure
already exists. The missing work is the applied ruler-scale consumer, not another
target catalog. Optical zoom requires matching lens calibration and renewed
review. Synthetic accuracy does not establish physical camera accuracy.

## Status Fix and Validation

PR #9959 records calibration source SHA-256 in reconstruction summaries, checks
source stability before reconstruction and publication, and compares that value
with the player's current review. Legacy or changed association yields
**Reconstruct Again** through existing wizard links. Triangulated model fits also
check their existing summary-input fingerprint, so old fits do not become current
merely because a reconstruction was rerun. Image-space fits retain their separate
observation/camera contract and need coverage in the remaining end-to-end audit.

- 26 focused camera-source, real reconstruction command, wizard and model-lineage
  tests pass on Python 3.12. The last extraction/model subset passes six tests.
- Initial source and helper tests failed before implementation. Same-path changes,
  missing sources, legacy associations, matching results, and actual wizard
  refresh after a new calibration review are covered.
- Required normal hooks passed on the published initial #9959 source. The final
  turnover source must retain its own successful push receipt; inspect GitHub.
- Tools: 27 independent OpenCV geometry/placement controls, existing API stability,
  root Ruff/format, changed-module mypy and all nine manual/inventory gates passed.
  The API repair adds exactly three entries; all existing entries are unchanged.
- Earlier candidate acceptance: 566 Capture Rig tests, 30 isolated provider
  checks, and five real reconstruction/model/variant controls passed. They do not
  qualify later source changes automatically or prove physical-camera accuracy.

## Full Goal Ledger

GitHub currently records product epic #9849, architecture map epic #9850 and
advanced expert overlay epic #9863 closed. Their existing implementations include
trim/crop, library notes/management, coaching drawings, reference import/sync,
generated maps and responsive UI work; preserve them during integration.
Everyday calibration #9897, clubs #9902, wizard #9906 and acceptance #9909 remain
open. In particular, club snapshots are model context, not a claim that a club
segment is physically fitted. Audit those requirements rather than closing the
parents based on the current narrow tests. Hardware qualification #9554 remains
separate and unproven.

Fleet communication was delivered through Repository_Management #1576/#1579;
the adoption receipt at `071fedc` covers 41 repositories. See its
`docs/development/agent_communication_remote_adoption.json`. Use the existing
presence/mailbox/lease system; do not create another message store. Gasification
mapping planning is already recorded under Gasification #4944 and intentionally
left to future cheaper agents at the user's request.

## Runtime and Coordination

The installed candidate is **Capture Rig — Candidate 7c7950574**, previously
verified responding as PID 61800 (launcher 37212). The final turnover lookup no
longer found that PID; do not assume it is running. Its dedicated runtime is
`%TEMP%/capture-reference-py312-runtime`; do not reinstall it while the app runs.
Camera preview is paused. Installed wheel SHA-256 is
`493f8e0cd19974313cf501dbf728dd937e62b4fe4f3fd52f8ceb6994f53109ad`.
Older apps and atlas port 2963 are preserved. Recheck process identity before any
lifecycle action. User video sessions and browser probe files are local artifacts,
not source to add to Git.

The Heavy Hit and Acoustics task owns impact/golf-club code and confirmed no
mocap overlap. Its task ID is `01a07d8a-869e-7540-a36b-bec998601c08`.
Current capture presence is `capture-product-01a08427-scale-consumer`; Tools uses
`capture-product-01a08427-linear-scale`. At handoff renew or replace leases only
after checking ownership. The newer CLI is available in
`Worktrees/Repository_Management-agent-console` (the default central checkout is
older). Use `python3 -m scripts.agent_communicate --repo REPO --session ID inbox`.
The historical rejected identity-change warning is not a current path conflict.

The system drive previously filled; free space later recovered externally.
Automatic approval rejected deleting task caches/runtimes. No deletion workaround
was used; do not retry those rejected targets. Local logs and the supplementary
`%TEMP%/capture-product-resume-checkpoint.json` aid recovery, but this committed
turnover and current external state are authoritative.

The status/lineage source and first complete turnover are committed and pushed
as `c63fa425880756d50cacf25aacdcbb25d9ab0dbb`; subsequent changes are handoff
clarifications. Tools has a concurrent scheduled-agent main merge at `4c5010e55`;
preserve it together with this task's turnover commit `62073ac6d`. Regenerate the
handoff manifest if resolving that merge changes its recorded evidence.
