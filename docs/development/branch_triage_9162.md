# Local Branch Triage (#9162)

Disposition ledger for every local branch in the primary UpstreamDrift
checkout (`C:/Users/diete/Repositories/UpstreamDrift`), resolving issue #9162.
The fleet development-log rollout (Repository_Management#1460) found 250
local branches ahead of `main`; by the time of this triage (2026-09-15) the
checkout carried **386** local heads. The log was bootstrapped with
`--no-seed` (#9161) precisely so that this wall of stubs would not land in
`DEVELOPMENT_LOG.md`; this ledger is where the triage lives instead, and the
development log carries exactly one entry for the triage itself (DL-#9162).

The triage is decided here; the deletions are executed by the runbook below
from the primary checkout, never from an issue worktree. Nothing in this
document was produced by mutating git state.

## Evidence Used

| Source                                       | What it establishes                                                                        |
| -------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `.git/refs/heads/**` and `.git/packed-refs`  | the 386 local heads                                                                        |
| `.git/logs/refs/heads/<branch>` (last entry) | who last moved the branch and when; 322 reflogs are fully expired (no entry in 30/90 days) |
| `.git/worktrees/*/HEAD`                      | which branches are checked out in a linked worktree (git refuses to delete those)          |
| `refs/remotes/origin/<branch>`               | whether a copy exists on `origin`, so a local delete loses nothing                         |
| `SPEC.md` Section 12                         | PR/issue numbers that reached `main` (rows are keyed by pull request)                      |
| `DEVELOPMENT_LOG.md`                         | which issues already have a live or shipped entry                                          |

Merged-into-`main` status is **not** decided from the evidence above; the
runbook resolves it mechanically with `git branch --merged origin/main` and
the `[gone]` upstream marker (squash merges do not register as merged).

## Disposition Rules

| Disposition        | Rule                                                                                                                                                                                                 | Count |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| **protected**      | `main`, the branch checked out in the primary checkout, and this issue's branch. Never deleted.                                                                                                      | 3     |
| **defer-worktree** | Checked out in a linked worktree. Out of scope for branch deletion; the fleet worktree-pruning pass (`prune_stale_worktrees.py`) owns the worktree, and the branch follows it.                       | 27    |
| **live**           | Reflog moved on or after 2026-09-01 and the name is a topic branch. Run the merged sweep first; every survivor is live work and gets a development-log entry under the reference shown.              | 34    |
| **stale**          | Topic branch whose reflog is expired (or last moved in August). Merged sweep first; survivors are abandoned: bundle-preserved, then deleted.                                                         | 178   |
| **abandoned**      | Merge-prep, conflict-resolution, backup, consolidation, PR-mirror, `tmp`/`test-pr`, `*-local` and empty `conductor/` scratch branches. Deleted after the bundle snapshot regardless of merge status. | 144   |

Every commit on a branch marked `local only` exists nowhere else. The bundle
snapshot in step 2 of the runbook is therefore mandatory before any `-D`.

## Coordination

- `codex/*`, `codex-*`, `bolt*`, `agent-*`, `consolidate*` and `combined/*`
  branches belong to concurrent agents. Every one of them classified
  `abandoned` or `stale` has an expired reflog (last moved before mid-June
  2026 or earlier), which is months past the 2 h lease TTL and the 24 h
  open-PR extension of the agent-lease protocol.
- Before step 4, post one inbox notice from the Repository_Management
  checkout listing the branches to be deleted and wait one business day:
  `python -m scripts.agent_communicate --repo UpstreamDrift --session <id> send --to all --text-file <list>`.
  For any branch whose name carries an issue number, also run
  `python -m scripts.check_agent_claim --repo UpstreamDrift --issue <N>`;
  a held claim by another agent removes that branch from the deletion set.
- Worktree-attached branches are never touched here, matching the 2026-07-22
  pruning pass that preserved agent worktree infrastructure.
- Do not modify or delete lease comments left by other agents.

## Runbook

Run from the primary checkout, on a day with no active agent sessions in the
repo (`python -m scripts.agent_communicate --repo UpstreamDrift --session <id> list`).
`TRIAGE=docs/development/branch_triage_9162.md` is this file.

1. Refresh remote state: `git fetch --prune origin`.
2. Snapshot every local head before deleting anything, then record the
   bundle path and SHA-256 under **Outcome** below:

   ```bash
   git bundle create ../UpstreamDrift-heads-9162-$(date +%Y%m%d).bundle --branches
   git bundle verify ../UpstreamDrift-heads-9162-*.bundle
   sha256sum ../UpstreamDrift-heads-9162-*.bundle
   ```

   Keep the bundle outside any git worktree. It restores any branch with
   `git fetch <bundle> refs/heads/<branch>:refs/heads/<branch>`.

3. Merged sweep (`live` and `stale` sets). `git branch -d` refuses
   unmerged and worktree-attached branches, so this step is safe to run over
   the whole set:

   ```bash
   git branch --merged origin/main | grep -v -E '^\*|^\+| main$' | tr -d ' ' > /tmp/merged
   git for-each-ref --format='%(refname:short) %(upstream:track)' refs/heads | awk '$2=="[gone]"{print $1}' > /tmp/gone
   xargs -r git branch -d < /tmp/merged
   xargs -r git branch -D < /tmp/gone   # squash-merged: PR merged and origin branch deleted
   ```

4. Delete the `abandoned` set after the coordination notice has aged:

   ```bash
   sed -n '/^## Abandoned/,/^## /p' "$TRIAGE" | sed -n -E 's/^- `([^`]+)`.*//p' | xargs -r git branch -D
   ```

5. Delete `stale` survivors the same way (`/^## Stale/`). Anything git
   refuses because a worktree holds it stays and is re-listed under
   **defer-worktree**.
6. For each `live` survivor, confirm an open PR exists with one REST call
   (`gh api "repos/D-sorganization/UpstreamDrift/pulls?head=D-sorganization:<branch>&state=open"`)
   and add a `DL-#<issue>` entry to `docs/development/DEVELOPMENT_LOG.md`
   under the governing reference in the table; a survivor with no open PR
   and no held claim is stale and follows step 5.
7. Fill in **Outcome**, set DL-#9162 to `shipped`, and `git worktree prune`.

## Outcome

Not yet executed. Record here: execution date, bundle path and SHA-256,
branches deleted per step, `live` survivors and the DL entries they received,
and any branch refused by git and why.

## Protected

- `conductor/issue-9162`: this issue's branch; local only
- `feat/10155-motion-matching-package`: checked out in the primary checkout; origin copy; #10155
- `main`

## Defer-Worktree

Out of scope for deletion until the fleet worktree pass removes the worktree.

| Branch                                               | Last reflog | `origin/` copy | Worktree                             |
| ---------------------------------------------------- | ----------- | -------------- | ------------------------------------ |
| `agent/fix-launcher-qa-validation-8071-8072`         | expired     | yes            | `UpstreamDrift-fix-8071-8072`        |
| `agent/fix-npm-advisories-7988`                      | expired     | yes            | `UpstreamDrift-7988-security-npm`    |
| `bolt-mesh-norm-opt-4257202569685040652`             | 2026-08-20  | no             | `pr-8798-format`                     |
| `bolt-optimize-funnel-benchmark-5950430926761491703` | 2026-08-20  | no             | `pr-8799-conflict`                   |
| `codex-pr8354-conflict`                              | expired     | no             | `UpstreamDrift-pr8354-conflict`      |
| `codex/pr-8355-dispersion-hypot`                     | expired     | no             | `pr-8355-dispersion-hypot`           |
| `codex/pr-8356-contact-hypot`                        | expired     | no             | `pr-8356-contact-hypot`              |
| `codex/pr-8357-mask-sum`                             | expired     | no             | `pr-8357-mask-sum`                   |
| `codex/rebuild-upstream-7890`                        | expired     | no             | `upstream-7890-rebuild`              |
| `codex/rebuild-upstream-7902`                        | expired     | no             | `upstream-7902-rebuild`              |
| `codex/rebuild-upstream-7966`                        | expired     | no             | `upstream-7966-rebuild`              |
| `codex/tools-3316-import-canonicalization`           | expired     | no             | `upstreamdrift-issue-3316`           |
| `conductor/issue-7736`                               | expired     | no             | `UpstreamDrift-conductor-issue-7736` |
| `consolidate/open-prs-20260727`                      | expired     | yes            | `UpstreamDrift-consolidate`          |
| `dep-fix-9156`                                       | 2026-08-29  | no             | `UD-dep-9156`                        |
| `feat/9934-biomechanical-analysis`                   | 2026-09-09  | yes            | `UpstreamDrift-biomechanics`         |
| `feat/golf-sim-pro-graphics`                         | expired     | yes            | `_wt_golf_sim`                       |
| `fix/9286-validation-admission`                      | 2026-09-08  | no             | `upstreamdrift-9286`                 |
| `fix/9690-club-shaft-coupling`                       | 2026-09-08  | no             | `upstreamdrift-9690`                 |
| `fix/9691-spherical-ball-dynamics`                   | 2026-09-08  | no             | `upstreamdrift-9691`                 |
| `fix/portable-artifact-links-9142`                   | 2026-08-27  | yes            | `UpstreamDrift-pr9147`               |
| `rebase-9240`                                        | 2026-08-29  | no             | `UD-rebase-9240`                     |
| `ud-9234-resolve`                                    | 2026-08-29  | no             | `UD-9234`                            |
| `ud-9252-resolve`                                    | 2026-08-29  | no             | `UD-9252`                            |
| `ud-9253-resolve`                                    | 2026-08-29  | no             | `UD-9253`                            |
| `ud-9254-resolve`                                    | 2026-08-29  | no             | `UD-9254`                            |
| `ud-9255-resolve`                                    | 2026-08-29  | no             | `UD-9255`                            |

## Live

Sweep first; survivors get a development-log entry under the governing reference.

| Branch                                                      | Last reflog | `origin/` copy | Governing reference                                                      |
| ----------------------------------------------------------- | ----------- | -------------- | ------------------------------------------------------------------------ |
| `bolt-optimize-list-summation-8418805788140045299`          | 2026-09-02  | yes            | bolt PR (perf lane)                                                      |
| `bolt-optimize-norm-retargeting-17531268052453484555`       | 2026-09-15  | yes            | bolt PR (perf lane)                                                      |
| `bolt-vector-magnitude-optimization-12415252143447788248`   | 2026-09-15  | yes            | bolt PR (perf lane)                                                      |
| `bolt/optimize-capture-rig-norm-11276562552528656285`       | 2026-09-13  | yes            | bolt PR (perf lane)                                                      |
| `bolt/optimize-linalg-norm-gui-5818749643169844677`         | 2026-09-15  | yes            | bolt PR (perf lane)                                                      |
| `bolt/optimize-max-magnitude-collision-1023459975214310183` | 2026-09-02  | yes            | bolt PR (perf lane)                                                      |
| `bolt/optimize-vector-norm-13737489114899385214`            | 2026-09-15  | yes            | PR #10187 (perf lane)                                                    |
| `docs/adr-0044-fix-forces-field-name`                       | 2026-09-02  | yes            | ADR-0044 (#9184)                                                         |
| `feat/10062-visual-skeleton-layer`                          | 2026-09-15  | yes            | #10062 HO-0 (#10186 landed)                                              |
| `feat/9762-bioptim-parameter-ocp`                           | 2026-09-09  | yes            | #9762                                                                    |
| `feat/9862-coaching-drawings`                               | 2026-09-09  | yes            | #9862                                                                    |
| `feat/9864-reference-assets`                                | 2026-09-09  | yes            | #9864 (DL-#9864 shipped)                                                 |
| `feat/9865-scene-registration`                              | 2026-09-09  | yes            | #9865 (DL-#9865 shipped)                                                 |
| `feat/9866-reference-comparison-workspace`                  | 2026-09-09  | yes            | #9866                                                                    |
| `feat/9881-reference-sync-calibration`                      | 2026-09-09  | yes            | #9881 (DL-#9881 shipped)                                                 |
| `feat/9882-preview-export-parity`                           | 2026-09-09  | yes            | #9882 (DL-#9882 shipped)                                                 |
| `feat/9967-native-simscape-pinocchio`                       | 2026-09-15  | yes            | #9967 (DL-#9967 in_progress) — keep                                      |
| `feat/adr0045-f1-roll-model-provenance`                     | 2026-09-02  | yes            | ADR-0045 F1                                                              |
| `feat/adr0047-h2-shot-tracer-import`                        | 2026-09-02  | yes            | ADR-0047 H2 (#9351)                                                      |
| `feat/impact-explorer-web-route`                            | 2026-09-02  | yes            | Impact Explorer web route (cherry-pick of #9184 follow-up)               |
| `feat/issue-10194-session-service`                          | 2026-09-15  | yes            | #10194 GS-05 (PR #10217 merged)                                          |
| `feat/segment-force-colors`                                 | 2026-09-08  | yes            | segment-force colour epic (docs/development/segment_force_color_epic.md) |
| `feat/shadow-tracker-video-decoding-10168`                  | 2026-09-15  | yes            | #10168 (PR #10234 merged)                                                |
| `feat/tour-matching-viewer-tile`                            | 2026-09-13  | yes            | #9921 tour-matching viewer                                               |
| `feat/uncertainty-propagation`                              | 2026-09-02  | yes            | uncertainty epic (see `goal-8752-uncertainty` worktree, #8752)           |
| `feat/validation-roadmap`                                   | 2026-09-02  | yes            | validation roadmap (#9286 lineage)                                       |
| `feat/visuals-step4-cross-engine-replays`                   | 2026-09-15  | yes            | PR #10203 merged                                                         |
| `fix/9165-launcher-exec-ownership`                          | 2026-09-02  | yes            | #9165                                                                    |
| `fix/9236-hybrid-authority-red`                             | 2026-09-02  | yes            | #9236                                                                    |
| `fix/9385-quality-gate-drift-contracts`                     | 2026-09-02  | yes            | #9385                                                                    |
| `fix/adr0046-g0-gates-actually-run`                         | 2026-09-02  | yes            | ADR-0046 G0 (#9348)                                                      |
| `fix/proximal-distal-solve-hang`                            | 2026-09-02  | yes            | proximal-distal program (`ud9234` lineage, #9234)                        |
| `fix/validate-suite-pre-migration-layout`                   | 2026-09-02  | yes            | resolve at sweep                                                         |
| `perf/9851-capture-responsiveness`                          | 2026-09-08  | yes            | #9851 (DL-#9851)                                                         |

## Stale

Sweep first; survivors are deleted after the bundle snapshot.

- `bolt-flight-models-hypot-2484070034629757581` — origin copy
- `bolt-math-hypot-opt-1541850552757815465` — origin copy
- `bolt-math-hypot-opt-1980069339518730802` — origin copy
- `bolt-np-einsum-deformable-17813726378030727` — local only
- `bolt-np-linalg-norm-opt-11381327926626002659` — origin copy
- `bolt-optimize-friction-laws-12744114279475527414` — origin copy
- `bolt-performance-improvement-13320917364305045593` — origin copy
- `bolt/optimize-norm-starting-pose-6444100604616567514` — origin copy
- `chore/an-review-followups` — origin copy
- `chore/rebaseline-module-size-budget` — origin copy
- `claude/error-handling-hardening-5911` — origin copy; #5911
- `claude/fix-ci-runner-label` — origin copy
- `claude/fix-ci-standard-ascii` — local only
- `claude/test-coverage-improvements` — origin copy
- `codex/7281-modelpanel-facade` — origin copy; #7281
- `codex/7283-ws-engine-manager-boundary` — origin copy; #7283
- `codex/7297-model-library-source-url-validation` — origin copy; #7297
- `codex/7300-cloud-client-blank-token` — origin copy; #7300
- `codex/7314-pr-unit-gate-hardening` — origin copy; #7314
- `codex/7328-ball-flight-spin-force` — local only; #7328
- `codex/7333-mujoco-matching` — local only; #7333
- `codex/7384-dry-followup` — local only; #7384
- `codex/benchmark-precision-search-7556-7561` — local only; #7556, #7561
- `codex/coriolis-cache-7558` — local only; #7558
- `codex/data-explorer-model-library-contracts` — origin copy
- `codex/deformable-object-vectorization` — local only
- `codex/export-write-failure-7722` — origin copy; #7722
- `codex/fd-central-differentiable-engine` — local only
- `codex/finite-diff-contiguous-7575` — local only; #7575
- `codex/fix-7037-spec` — local only; #7037
- `codex/fix-analysis-metrics-errors-1041` — origin copy
- `codex/fix-manipulation-env-pos` — local only
- `codex/fix-realtime-zero-torque-7685` — origin copy; #7685
- `codex/fix-torque-polynomial-order-7688` — origin copy; #7688
- `codex/ilqr-spd-solve-7570` — local only; #7570
- `codex/imitation-transitions-vectorize-7563` — local only; #7563
- `codex/issue-7217-launcher-frameless-extract` — origin copy; #7217
- `codex/issue-7562-jax-vmap` — origin copy; #7562
- `codex/issue-7706-quality-shim` — origin copy; #7706
- `codex/lower-body-ik-solve-7556` — local only; #7556
- `codex/medfilt-select-nth-7574` — local only; #7574
- `codex/mocap-filter-lane-7573` — local only; #7573
- `codex/pr-7258-mocap-path-errors` — local only; #7258
- `codex/pr-7473-ui-foundation` — local only; #7473
- `codex/pr-7748-spec` — local only; #7748
- `codex/pr7021-vector-magnitude` — local only; #7021
- `codex/pr7261-launcher-split` — local only; #7261
- `codex/pr7265-doc-index-fix` — local only; #7265
- `codex/qp-vector-constraints-7568` — local only; #7568
- `codex/retargeter-allocation-cache` — local only
- `codex/robotics-precision-vectorization-7556-7559-7563` — local only; #7556, #7559, #7563
- `codex/rrt-nearest-propagation-7564` — local only; #7564
- `codex/symlink-path-policy-20260611` — origin copy
- `codex/ud-6181-api-security-realtime-slice` — origin copy; #6181
- `codex/ud-6181-docs-container-slice` — origin copy; #6181
- `codex/ud-6181-myosuite-pose-slice` — origin copy; #6181
- `codex/ud-7819-grip-rebase-20260621` — local only; #7819
- `codex/upstream-launcher-embed-contracts` — origin copy
- `codex/upstream-plot-contracts-coverage` — origin copy
- `codex/upstream-theme-colors-coverage` — origin copy
- `codex/upstream-theme-icons-zoom-coverage` — origin copy
- `codex/upstream-theme-integration-coverage` — origin copy
- `codex/upstream-theme-manager-coverage` — origin copy
- `codex/upstream-theme-style-coverage` — origin copy
- `codex/upstream-theme-stylesheets-coverage` — origin copy
- `codex/upstream-theme-typography-coverage` — origin copy
- `codex/upstreamdrift-7034` — local only; #7034
- `codex/upstreamdrift-7471-ci-import-fix` — local only; #7471
- `data/cmu-subject-64-golf` — origin copy
- `dependabot/npm_and_yarn/ui/terser-5.48.0` — local only
- `dependabot/npm_and_yarn/ui/typescript-eslint-8.60.0` — local only
- `docs/6804-adapter-authoring-guide` — origin copy; #6804
- `docs/9066-design-manual-authority` — origin copy; reflog 2026-08-27; #9066
- `feat/5909-adr-numbering-ci` — origin copy; #5909
- `feat/5968-ux-wrappers` — origin copy; #5968
- `feat/5969-standalone-sidekick` — origin copy; #5969
- `feat/6084-myosuite-pose-interchange` — origin copy; #6084
- `feat/6774-canonical-state` — origin copy; #6774
- `feat/6775-canonical-model-core` — origin copy; #6775
- `feat/6777-capability-taxonomy` — origin copy; #6777
- `feat/6779-conformance-harness` — origin copy; #6779
- `feat/6780-ci-wiring` — origin copy; #6780
- `feat/6782-pinocchio-canonical-v2` — origin copy; #6782
- `feat/6783-mujoco-canonical-v2` — origin copy; #6783
- `feat/6784-differential-report` — origin copy; #6784
- `feat/6785-observation-schema` — origin copy; #6785
- `feat/6786-pose2sim-integration` — origin copy; #6786
- `feat/6787-opencap-integration` — origin copy; #6787
- `feat/6790-synthetic-groundtruth-rig` — origin copy; #6790
- `feat/6791-cc18-residuals` — origin copy; #6791
- `feat/6792-map-estimator` — origin copy; #6792
- `feat/6794-addbiomechanics-inertia-priors` — origin copy; #6794
- `feat/6794-addbiomechanics-inertia-priors-quality-fix` — local only; #6794
- `feat/6795-nimble-gradient-oracle` — origin copy; #6795
- `feat/6796-moving-horizon-estimator` — origin copy; #6796
- `feat/6797-ztcf-zvcf-canonical` — origin copy; #6797
- `feat/6798-wrench-extractor` — origin copy; #6798
- `feat/6801-drake-canonical-core` — origin copy; #6801
- `feat/6803-myosuite-canonical-core` — origin copy; #6803
- `feat/6805-app-shell-tool-registry` — origin copy; #6805
- `feat/6807-engine-selector-comparison-views` — origin copy; #6807
- `feat/6808-session-results-browser` — origin copy; #6808
- `feat/6809-config-setup-wizard` — origin copy; #6809
- `feat/6810-sidekick-retrieval-qa` — origin copy; #6810
- `feat/6811-sidekick-canonical-tools` — origin copy; #6811
- `feat/9128-swing-objective-web-parity-and-research` — origin copy; reflog 2026-08-27; #9128
- `feat/cc15-keypoint-offset` — origin copy
- `feat/jaxsim-cross-engine-gate` — origin copy
- `feat/jaxsim-parameter-gradients` — local only
- `feat/mjwarp-backend` — origin copy
- `feat/parity-about-7459` — origin copy; #7459
- `fix/6013-background-tabs-followup` — origin copy; #6013
- `fix/6658-jaxsim-selector` — origin copy; #6658
- `fix/6659-jaxsim-convention-annotations` — origin copy; #6659
- `fix/6766-pose-subscriber-background-target` — local only; #6766
- `fix/6862-docs-governance` — local only; #6862
- `fix/7208-mjcf-fixed-joint-roundtrip` — origin copy; #7208
- `fix/build-on-install-rust-7600` — origin copy; #7600
- `fix/ci-broaden-runner-label-20260527` — origin copy
- `fix/ci-security-test-and-docker-docs` — local only
- `fix/ci-security-test-docker-docs-main` — origin copy
- `fix/concurrency-leaks-7148` — origin copy; #7148
- `fix/current-main-ci-after-7632` — local only; #7632
- `fix/embed-adapter-token-stale` — origin copy; reflog 2026-08-29
- `fix/file-size-budget-diagnostics` — origin copy
- `fix/frontend-7165` — origin copy; #7165
- `fix/human-review-toolcache` — local only
- `fix/issue-6494-redact-comma` — origin copy; #6494
- `fix/issue-6515-qapp-ref` — origin copy; #6515
- `fix/issue-6637-tauri-api-base` — local only; #6637
- `fix/issue-6638-engine-checkpointable` — local only; #6638
- `fix/issue-6638-mujoco-state-gravity-name` — local only; #6638
- `fix/issue-6638-physics-engines-checkpointable` — local only; #6638
- `fix/issue-6639-motion-pipeline` — local only; #6639
- `fix/issue-6640-safety-monitor` — local only; #6640
- `fix/issue-6642-frontend-errors` — local only; #6642
- `fix/issue-6679-grid-wrapping-sidebar` — local only; #6679
- `fix/issue-7129` — origin copy; #7129
- `fix/issue-7688-mujoco-coeff-order` — origin copy; #7688
- `fix/issue-7716` — origin copy; #7716
- `fix/issue-7719-mypy-ws` — origin copy; #7719
- `fix/issue-7719-ws-set-speed-dedup` — local only; #7719
- `fix/launcher-sidekick-review-fixes` — origin copy
- `fix/launcher-startup-tiles-and-deps` — origin copy
- `fix/main-linux-mypy-baseline` — origin copy
- `fix/main-mypy-baseline-after-7641` — local only; #7641
- `fix/main-strict-mypy-and-benchmark-gates` — local only
- `fix/main-tauri-build-contract` — local only
- `fix/main-tauri-version-alignment` — local only
- `fix/main-tauri-windows-app-control-gate` — local only
- `fix/main-ui-audit-and-color-guard` — local only
- `fix/main-workflow-run-trust-boundary` — local only
- `fix/modal-a11y-focus-trap-7438` — origin copy; #7438
- `fix/mypy-fixes-forgejo-20260607` — local only
- `fix/physics-correctness-7144` — origin copy; #7144
- `fix/physics-engine-checkpointable-conformance` — local only
- `fix/plot-identity-export-metadata-8828` — origin copy; reflog 2026-08-27; #8828
- `fix/remedy-open-review-issues` — origin copy
- `fix/review-7017-7015-6954` — origin copy; #7017, #7015, #6954
- `fix/review-feedback-6816-6827` — origin copy; #6816, #6827
- `fix/root-clutter-keystone` — origin copy
- `fix/rust-audit-hardening` — origin copy
- `fix/security-auth-7139` — origin copy; #7139
- `fix/spec-tauri-toolchain-7616` — origin copy; #7616
- `fix/suppression-ratchet-7625` — local only; #7625
- `fix/tauri-toolchain-assert-7616` — origin copy; #7616
- `fix/test-api-extended-stale-import` — local only
- `fix/toolcache-shared-race` — origin copy
- `fix/upstreamdrift-adversarial-review-4-5-6-8-9` — local only
- `fix/workflow-gitconfig-resilient` — local only
- `port/test-suite-oom` — local only
- `refactor/6563-contracts-consolidation` — origin copy; #6563
- `refactor/6564-remove-deprecated-pkg` — origin copy; #6564
- `refactor/6565-pydantic-settings` — origin copy; #6565
- `refactor/6566-theme-unification` — origin copy; #6566
- `refactor/6568-data-io-schemas` — origin copy; #6568
- `refactor/unify-c3d-readers` — local only
- `test/coverage-boost-forgejo-20260607` — local only

## Abandoned

Deleted after the bundle snapshot and the coordination notice.

- `9865-reference-overlay-calibrated-scene-registration-and-event-synchronization` — origin copy; reflog 2026-09-09; #9865
- `agent-b-consolidate-prs-v3` — local only
- `agent-e-consolidate-prs-v3` — local only
- `agent-f-fold-upstreamdrift-perf-dupes` — local only
- `assessment/ao-2026-06-18` — origin copy
- `auto/backlog-6816` — origin copy; #6816
- `backup/7333-amended-with-spec-20260610-2315` — local only; #7333
- `backup/local-7373-pre-reset-20260610-223950` — local only; #7373
- `backup/local-chat-ws-test-harness-9950` — local only; #9950
- `backup/main-ci-fix-local-amend-20260611` — local only
- `backup/pr7396-pre-main-rebase-90fbe892` — local only; #7396
- `backup/stale-worktree/upstreamdrift-pr5541-20260616` — local only; #5541
- `backup/worktree-prune-20260615/UpstreamDrift-7295-refresh` — local only; #7295
- `backup/worktree-prune-20260615/UpstreamDrift-7310-refresh` — local only; #7310
- `backup/worktree-prune-20260615/UpstreamDrift-7324-refresh2` — local only; #7324
- `backup/worktree-prune-20260615/UpstreamDrift-7324-update` — local only; #7324
- `bolt-hypot-3d-12235444657656018243-local` — local only
- `claude/sci-accuracy-7411-7412-7413-local` — local only; #7411, #7412, #7413
- `codex-pr-6920` — local only; #6920
- `codex-pr-7803-conflict-20260620-074908` — local only; #7803
- `codex-pr7263-doc-governance-fix` — local only; #7263
- `codex-resolve-7320-20260610-195427` — local only; #7320
- `codex-upstreamdrift-7606-fix` — local only; #7606
- `codex-upstreamdrift-7798-merge-main-20260620` — local only; #7798
- `codex-verify-7320-20260610-200342` — local only; #7320
- `codex/7300-cloudclient-blank-token-local` — local only; #7300
- `codex/7334-pinocchio-model-fidelity-local` — local only; #7334
- `codex/issue-7217-launcher-entrypoint-split-fresh` — origin copy; #7217
- `codex/local-upstream-7793-20260620-090158` — local only; #7793
- `codex/local-upstream-7795-20260620-090551` — local only; #7795
- `codex/local-upstream-7798-20260620-090413` — local only; #7798
- `codex/local-upstream-7799-worker` — local only; #7799
- `codex/local-upstream-7800-worker` — local only; #7800
- `codex/local-upstream-7812-20260620-094206` — local only; #7812
- `codex/local-upstream-7813-20260620-100147` — local only; #7813
- `codex/main-ci-fix-20260611b` — origin copy
- `codex/main-hotfix-20260611` — origin copy
- `codex/mergeprep-upstreamdrift-pr7759-20260620-023501` — local only; #7759
- `codex/mergeprep-upstreamdrift-pr7778-20260620-023501` — local only; #7778
- `codex/mergeprep-upstreamdrift-pr7788-20260620-023501` — local only; #7788
- `codex/mergeprep-upstreamdrift-pr7790-20260620-023501` — local only; #7790
- `codex/mergeprep-upstreamdrift-pr7793-20260620-023501` — local only; #7793
- `codex/mergeprep-upstreamdrift-pr7794-20260620-023501` — local only; #7794
- `codex/mergeprep-upstreamdrift-pr7795-20260620-023501` — local only; #7795
- `codex/mergeprep-upstreamdrift-pr7796-20260620-023501` — local only; #7796
- `codex/mergeprep-upstreamdrift-pr7797-20260620-023501` — local only; #7797
- `codex/mergeprep-upstreamdrift-pr7798-20260620-023501` — local only; #7798
- `codex/mergeprep-upstreamdrift-pr7799-20260620-023501` — local only; #7799
- `codex/mergeprep-upstreamdrift-pr7800-20260620-023501` — local only; #7800
- `codex/mergeprep-upstreamdrift-pr7802-20260620-023501` — local only; #7802
- `codex/mergeprep-upstreamdrift-pr7803-20260620-023501` — local only; #7803
- `codex/pr-7237-startup` — local only; #7237
- `codex/pr-7248` — local only; #7248
- `codex/pr-7469-spec-fix` — local only; #7469
- `codex/pr-7470-spec-fix` — local only; #7470
- `codex/pr-7473-ui-foundation-merge` — local only; #7473
- `codex/pr-7477-manifest-web-conflict` — local only; #7477
- `codex/pr-7484-settings-conflict` — local only; #7484
- `codex/pr-7490-shottracer-conflict` — local only; #7490
- `codex/pr-7491-plots-conflict` — local only; #7491
- `codex/pr-7492-mocap-conflict` — local only; #7492
- `codex/pr-7494-crossengine-conflict` — local only; #7494
- `codex/pr-7495-counterfactual-conflict` — local only; #7495
- `codex/pr-7496-consolidate` — local only; #7496
- `codex/pr-7788-merge-main-20260620` — local only; #7788
- `codex/pr-7790-merge-main-20260620` — local only; #7790
- `codex/pr-7799-merge-main-20260620` — local only; #7799
- `codex/pr-7800-merge-main-20260620` — local only; #7800
- `codex/pr7254-resolve` — local only; #7254
- `codex/pr7296-main-merge` — local only; #7296
- `codex/pr7889-conflict-20260725` — local only; #7889
- `codex/pr7890-conflict-20260724` — local only; #7890
- `codex/pr7902-conflict-20260725` — local only; #7902
- `codex/pr7967-conflict-20260724` — local only; #7967
- `codex/pr7968-conflict-20260724` — local only; #7968
- `codex/pr8044-update-20260725` — local only; #8044
- `codex/pr8059-conflict-20260725` — local only; #8059
- `codex/rebase-7039-perf` — local only; #7039
- `codex/tmp-pr-6187` — local only; #6187
- `codex/tmp-pr-6199` — local only; #6199
- `codex/tmp-pr-6200` — local only; #6200
- `codex/tmp-pr-6211` — local only; #6211
- `codex/tmp-pr-6212` — local only; #6212
- `codex/tmp-pr-6213` — local only; #6213
- `codex/tmp-pr-6214` — local only; #6214
- `codex/upstream-pr-7790-conflict-20260620-1355` — local only; #7790
- `codex/upstream-pr-7793-conflict-20260620-1355` — local only; #7793
- `codex/upstream-pr-7795-conflict-20260620-1355` — local only; #7795
- `codex/upstream-pr-7798-conflict-20260620-1355` — local only; #7798
- `codex/upstream-pr-7799-conflict-20260620-1355` — local only; #7799
- `codex/upstream-pr-7800-conflict-20260620-1355` — local only; #7800
- `codex/upstream-pr-7803-conflict-20260620-1355` — local only; #7803
- `codex/upstreamdrift-7808-merge-main` — local only; #7808
- `codex/upstreamdrift-consolidated-7468-7470` — origin copy; #7468, #7470
- `codex/upstreamdrift-ui-deps-consolidated` — origin copy
- `combined/all-prs-20260522` — origin copy
- `combined/consolidated-backlog` — local only
- `combined/upstream-drift-final-20260525` — origin copy
- `combined/upstream-drift-q1-20260525` — origin copy
- `combined/upstream-drift-q2-20260525` — origin copy
- `combined/upstream-drift-q3-20260525` — origin copy
- `combined/upstream-drift-q4-20260525` — origin copy
- `conductor/issue-8359` — local only; reflog 2026-09-10; #8359
- `conductor/issue-9387` — origin copy; reflog 2026-09-06; #9387
- `consolidate-work` — local only
- `feat/parity-recordings-7451-local` — local only; #7451
- `feat/parity-services-7446-local` — local only; #7446
- `feat/parity-stubs-7448-local` — local only; #7448
- `feat/parity-tstypes-7447-local` — local only; #7447
- `feat/segment-force-colors-final-integration` — local only; reflog 2026-09-08
- `feature/coverage-wave-backup` — local only
- `fix/ci-main-green-pyarrow-mypy-fresh` — local only
- `fix/ci-main-green-pyarrow-mypy-local` — local only
- `fix/issue-7694-safety-tests-local` — local only; #7694
- `fix/issue-7697-realtime-estop-test-local` — local only; #7697
- `fix/issue-7713-opensim-hoist-integrator-mergefix` — local only; #7713
- `fix/issue-7714-vectorize-clubhead-traj-mergefix` — local only; #7714
- `fix/pr-7538-spec-refresh` — local only; #7538
- `local-8083-rerun` — local only; #8083
- `pr-5944` — local only; #5944
- `pr-6145` — local only; #6145
- `pr-6146` — local only; #6146
- `pr-6147` — local only; #6147
- `pr-6148` — local only; #6148
- `pr-6149` — local only; #6149
- `pr-6150` — local only; #6150
- `pr-6151` — local only; #6151
- `pr-6152` — local only; #6152
- `pr-6153` — local only; #6153
- `pr-6154` — local only; #6154
- `pr-6553` — local only; #6553
- `pr-6819-provenance` — local only; #6819
- `pr-6824-capability-taxonomy` — local only; #6824
- `pr-7074-fix-api-contract-reds` — local only; #7074
- `pr-7513-lifecycle-fix` — local only; #7513
- `pr-7519-current` — local only; #7519
- `pr-7665-complexity-fix` — local only; #7665
- `sentinel-fix-drake-pickle-4567978792504172650-local` — local only
- `test-pr-10087` — local only; reflog 2026-09-13; #10087
- `test-pr-10089` — local only; reflog 2026-09-13; #10089
- `tmp-6854-retrigger` — local only; #6854
- `ud9157` — local only; reflog 2026-08-29; #9157
- `ud9234` — local only; reflog 2026-08-29; #9234
- `ud9237` — local only; reflog 2026-08-29; #9237
