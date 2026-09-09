# Agent Handoff: Proximal–Distal Research Program

## Segment Force Colors: Epic #9833 in Progress

Branch `feat/segment-force-colors` lives in `_codex_worktrees/segment-force-colors`.
See `docs/development/segment_force_color_epic.md` for current scope and evidence.
Shared Python/Three.js policies, native pendulum and MuJoCo reaction sources,
renderer adapters, desktop controls and WebSocket/Scene3D wiring are implemented.
Epic #9833 and children #9834–#9837 are published. Remote main `ff0effa5a` was merged
into the branch. Git works by clearing the stale `http.https://github.com/.extraheader`
per invocation and using `gh auth git-credential`; global settings are unchanged.
Remaining gates: wider interface adapters, native render evidence, protected CI
and merge. Native MuJoCo tests on this Windows host must import mujoco before
pytest/Qt to avoid a loader-order DLL failure. Do not claim universal rollout.

Updated: 2026-09-08 02:55 PDT
Updated: 2026-09-08 03:10 UTC
Updated: 2026-09-08 09:30 UTC (PR backlog catch-up sweep)

## Capture Rig Multiview Epic #9818: 2026-09-08 (Agent `claude`)

The Capture Rig tile went from "no live view at all" to a recording station.
Merged in order: #9809 (live preview, movable/scrollable panes with saved dock
layouts, transport-style recording), #9819 (#9816 theme and layout standards:
`styling.py`, `header.py`, `action_grid.py`, `playback.py` split out of
`gui.py`), #9820 (#9810/#9811 `layout_model.py` compositor + `layout_presets.py`
store), #9821 (#9815 `mosaic.py` + `rig multipicture`), #9823 (#9813/#9814 live
preview and playback rendered through a `LayoutSpec`). PR #9822 was closed as
superseded: it was #9823's base, so its files landed byte-identically with that
squash, and merging it would have reverted the multiview panes. Only #9817
(this documentation pass) remained.

Hardware facts worth keeping, all measured on the three-camera rig and written
up in `docs/motion_capture/evidence/capture_rig_multiview.md`:

- A DirectShow camera opens once. During a take the recorder tees its own
  preview (`--live-preview DIR`); the tile shows those snapshots.
- That tee must decode cheaply. At full resolution it starved the stream copy
  (95/393/74 frames in ~7 s); with `-lowres:v 2` and a 256 MB real-time buffer
  all three cameras hold 60 fps (493/494/462 frames in 8 s).
- Enumeration costs ~30 s; `--camera VIEW=INSTANCE_ID` reuses what the preview
  already bound.
- OpenCV cannot drive these cameras: Media Foundation hangs on the third unit
  and DirectShow-by-index refuses 1920x1200@60.

Two defects were found while integrating the parallel branches, not by CI:
`workflow.py` briefly held two `ACTION_HELP` tables where the second silently
won, and three new entry points each took nine parameters against a budget of
eight (now `MosaicOptions` / `MultipictureArgs`). A wall-clock assertion in
`test_layout_model.py` that flaked under load now takes the best of several
rounds.

## PR Backlog Catch-Up Sweep: 2026-09-08 (Agent `claude`, Session UpstreamPRs)

Disposition of the 38-PR open backlog (REST-verified states at sweep start):

- **Merged (18):** #9513, #9715, #9716, #9717, #9718, #9719, #9721, #9722, #9728, #9734, #9736, #9738, #9739, #9741, #9742, #9743, #9744, #9745. Every branch was brought current with `main` (REST update-branch, never force-push) and its AGENT_HANDOFF/SPEC/DEVELOPMENT_LOG conflicts resolved (union for SPEC rows, main-wins for handoff stamps, DL entries re-added into the Active table).
- **Closed as redundant (3):** #9433 (duplicate of #9723 for #9409), #9437 (superseded by the merged #9412 registry work, `1b53a9bf5`), #9737 (superseded by merged #9740 for #9699). Explanatory comments posted on each.
- **Armed for auto-merge (9):** #9434, #9465, #9471, #9720, #9723, #9724, #9725, #9726, #9729. Branches are conflict-free and current; auto-merge (merge/squash per repo allowance) merges each as `quality-gate` passes under strict up-to-date. No action needed; they drain serially.
- **Blocked (2):**
  - **#9440** — the `split` ruling deletes UD `theme/__init__.py`+`palette.py`, but main landed UD-only palette/typography extensions (#8972/#9037, `ThemePalette`/`get_current_colors`, ~217 lines) that the pinned Tools tree (`eab74a901a`) does not contain; 10+ launchers/API modules consume them. Next steps: land #8972/#9037 in D-sorganization/Tools tools-canonical side, bump the `vendor/ud-tools` pin, re-merge main, regenerate `docs/shared_tools/divergence_inventory{.md,.v1.json}` via `scripts/shared_tools/divergence_inventory.py`, then re-run `tests/unit/shared_python/test_seam_redirect.py` + quality-gate.
  - **#9442** — stacked on #9440's base branch (retarget to `main` only after #9440 lands); its own CI failures overlap open #9607 / PR #9726 (pinocchio authority lock drift) and a self-hosted authority-runner artifact path. Do not merge in isolation.
- **Skipped drafts (4):** #9610, #9618, #9633, #9636 (conductor research drafts, not trivially completable).

No issue was closed in this sweep (redundant-PR closures do not close issues). Fleet-wide handoff/lease state at sweep start: `C:/tmp/backlog/UpstreamDrift.md`.

## Impact Dynamics and Acoustics: #9700

- Theory AffineDrift #4258/#4282 and integration plan #9706 merged. Tools T1 #5077 merged; T2 #5082 remains open.
- Tools T3 rotating loaded roots published at `f47f64acf` (560 golf/API passes, two optional CAD skips). Full stability, work, contact/acoustics and empirical gates remain open.
- Tools early T5 #5084 and T6 #5083 merged; source audits and remaining qualification work are on #5074/#5075. Metadata labels alone do not validate measurements.
- Provider prerequisite #9735 / PR #9745 merged at `1b48707d54fb47655e43eaaffaad7b1739445e40`. Renderer PR #9784 merged at `9aa26e4f8`; #9787 repairs its observed research companion failures.
- #9787 / PR #9804: synchronized with main `563e7aaa9`, including native bootstrap #9726. Actual regenerated native record has SHA `0c0f3395`; only governed source hashes change. All 122 combined authority/provenance/bootstrap tests and three distinct rolling tests pass. Reviewer summary and inspected 253-page PDF are current; incoming mocap aliases are explicitly typed and the actual mypy hook passes. See `docs/development/manufactured_authority_9787_turnover.md`; protected CI remains required.
- Preserve #8557 protected authority, exact Tools pin, manufactured-data limits and workstation recovery restrictions; no calibrated acoustic solver is established.

- Import/provider history: `docs/development/impact_provider_import_turnover.md`; reviewed renderer evidence: `docs/development/renderer_reference_9783_turnover.md`; program design: `docs/development/impact_acoustics_program.md`.

Epic #8557 is canonical; issue state, local files, and checkpoints are not completion
evidence. UP-D0 (#9066) and UP-D1 (#9067) remain a separate design-manual program.

Detailed takeover context, the merge-versus-quarantine boundary, exact smoke contract,
recovery constraints, and next commands are in `docs/development/proximal_distal_program_turnover.md`.

## Import Bootstrap Fail-Fast: #9733

- Open PR fixes the pytest livelock in fresh worktrees with `vendor/ud-tools`
  uninitialized: `src/__init__.py` now raises an actionable ImportError naming
  `git submodule update --init vendor/ud-tools` instead of installing the
  fallback finder into an unbounded `find_spec` recursion. Regression tests:
  `tests/unit/repo_hygiene/test_src_fallback_fail_fast_9733.py` (probe simulated
  by monkeypatch; never touches the real submodule).

Epic #8557 is canonical; issue state, local files, and checkpoints are not
completion evidence. UP-D0 (#9066) and UP-D1 (#9067) remain a separate
design-manual program.

Detailed takeover context (merge-versus-quarantine boundary, exact smoke contract, recovery constraints, next commands): `docs/development/proximal_distal_program_turnover.md`.

Seam (#9406) and failure triage (#9474): see
`docs/development/readiness_seam_handoff.md` before retiring a shared cluster.
UD #9492 (branch `claude/issue-9492-decompose-timer`): `_on_timer` and
`_add_live_kinematics_overlays` are decomposed into focused helpers, both dated
`architecture_budget.json` exceptions are removed, behavior pinned by tests.

## In-Flight Tool Migration: #9470 (Launch-Monitor Async Analytics)

- Branch `claude/issue-9470-async-analytics` migrates the seven launch-monitor
  analysis handlers onto the #8880 `async_action` worker: one shared
  `AsyncActionBar` for all trigger buttons, widget-free `_compute_*` halves
  with cancellation checkpoints, synchronous `present(compute())` paths kept,
  and the embed adapter `cleanup()` cancelling and joining the worker.
- The branch is stacked on PR #9472 (the #8880 mechanism, `readiness/p2-8880-async-action-worker`);
  merge #9472 first, then this PR applies cleanly.
- Remaining #9470 checklist follow-ups, in pain order: `bunker_shot_gui/gui.py`
  (`_guarded` wait cursor), `putting_green_gui`, `ball_flight_gui`,
  `swing_flight_pipeline`, `terrain_engine`, then the audit-only tools.
- Gate: `python -m pytest -q tests/ui/tools/launch_monitor tests/tools/test_async_action.py`

## Unit-Gate `src`-Identity Sentinel: #9387 (Merged)

- The worker-corruption class from #9099 (a test mutating
  `sys.modules['src']`/`src.*` and corrupting later tests on the same
  xdist worker) is covered by a runtime sentinel,
  `tests/unit/repo_hygiene/test_src_identity_sentinel.py`: each of the
  four documented victim files runs in its own serial subprocess
  (`-p no:xdist`) and the sentinel asserts `sys.modules['src']`
  identity plus the `src.*` namespace snapshot are unchanged. RED was
  demonstrated with a scratch pivot module (removed before commit).
- The four judgment-call leak sites from the audit are explicitly
  snapshot/restored (`test_ux_enhancements.py`, pinocchio
  `test_tasks.py`, `test_gui_import_boundaries.py`,
  `test_golf_launcher_integration.py`); the full audit table lives in
  the PR body. `tests/unit/test_ux_enhancements.py` collects 0 test
  functions (dead fixture file since #5753) — deletion candidate for a
  follow-up PR; do not delete it here.
- Known environment limitation: in `git worktree` checkouts (`.git` is
  a pointer file), `tests/scripts/test_validate_suite.py` fails two
  `.git`-inspecting tests; the sentinel tolerates exactly those two ids
  there and requires them to pass in normal checkouts and CI.

## Pre-Commit on Windows — Resolved (#9494)

- The hook environment works on Windows; no `--no-verify` exception exists
  or is needed. `default_language_version: python: python` resolves to the
  PATH interpreter (3.13.3 here); the old `python3.11` pin is gone since
  #1792/#2720. Do not pin a minor version — workstations without it cannot
  build hook virtualenvs.
- Verified 2026-09-08 (pre-commit 4.6.2, from-scratch env build): all
  commit-stage hooks pass; pre-push `mypy`/`bandit` pass on scoped files;
  `pytest-unit` is slow locally (CI owns the full suite). CLAUDE.md
  "Hook bypass policy" documents this resolution.

## `bioptim` Optimal-Control Layer and the Swing-Dynamics Fixes (#9762)

- Branch `claude/fixes-epic-implementation-x2bu36`, PR #9768 (open). Epic doc:
  `docs/issues/EPIC_BIOPTIM_OCP_INTEGRATION.md`; decision: ADR-0050.
- Prerequisite issues #9755-#9761 are filed; #9755-#9760 are implemented on
  this branch, #9761 (upstream PR to pyomeca/bioptim) is external and open.
- Epic phases 0-3 are implemented and tested; phases 4 (parameter block) and
  5 (moving-horizon wrapper) are not started. `ocp/tracking_ocp` already
  accepts a `parameters` list, which is the seam phase 4 builds on.
- **Do not** import `bioptim` outside `src/shared/python/optimization/ocp/`:
  `tests/architecture/test_bioptim_isolation.py` fails on it. bioptim is
  git-pinned to `Release_3.4.0` (SHA `fdafe4d9`) in the `[bioptim]` extra;
  re-pinning is a ticket that re-runs the phase 0-3 tests.
- Two findings that constrain how results may be read:
  maximising terminal clubhead speed is a concave objective and converges in
  no backend once the dynamics are enforced, so the OCP defaults to a convex
  target-speed objective; and the six-marker set cannot observe the full
  seven-DOF chain (`hip_rotation` and `trunk_rotation` are an exact null
  direction), so tracking results report their own identifiability.
- Gate commands: `MPLBACKEND=Agg pytest tests/integration/optimization/ocp
tests/architecture/test_bioptim_isolation.py -m "not slow"`;
  `pytest tests/unit/optimization tests/unit/estimation`;
  `MPLBACKEND=Agg PYTHONPATH=src python -m benchmarks.bioptim_parity --nodes 8
--duration 0.6`. The benchmark needs `PYTHONPATH=src` (pytest's conftest adds
  it, `python -m` does not) or it dies importing `bunkershot3d`.
- The ocp tests live in `tests/integration/optimization/ocp/`, NOT under
  `src/`. `scripts/check_test_layout.py` (the Test Layout Guard inside
  `repo-structure-gates`) rejects any new `tests` directory under `src/`
  because root pytest does not collect it -- its `LEGACY_SRC_TEST_DIRS`
  allowlist is grandfathered debt, so do not add to it. They sit beside
  `test_casadi_swing_live.py`, the other suite that needs the real optional
  stack rather than the mocks `tests/unit/conftest.py` installs.
- The lane that really exercises them is the `bioptim OCP Tests` step in
  `ci-optional-stack.yml`, which installs the extra and runs only
  `tests/integration/optimization/ocp` plus the isolation test under
  `-m "requires_bioptim or integration"`. It is NOT fail-soft. Locally that
  exact selection is 26 passed in about 150 s; a timeout there is contention,
  not a hang -- the slowest test on its own is 80 s against a 600 s budget.
- Run the ocp tests on their own. Co-running them with
  `tests/unit/optimization` makes `bioptim_available()` return False and
  silently skips 16 of them, in either collection order and on `main` as well
  as here -- something in that directory poisons the import. Pre-existing, not
  this branch's, and worth its own issue: a skip that only appears in a
  combined run is exactly the kind CI hides.
- Two gates only reachable once the earlier ones passed, both fixed:
  `code-quality` fails at its **mypy** step, not ruff, on
  `crocoddyl_backend.py:316` -- `pin.Motion` is missing from the repo's own
  `stubs/pinocchio/__init__.pyi`, which `mypy_path = "stubs"` makes
  authoritative whether or not pinocchio is installed. Reproduce it in an
  environment WITHOUT pinocchio; a venv that has the real package hides
  nothing, but it is the stub mypy reads either way. `unit-test-gate` fails
  `test_divergence_inventory` until `python -m
scripts.shared_tools.divergence_inventory --write` re-records the 16 new
  `optimization/ocp/` files; that regeneration also rewrites unrelated
  authorship rows, which is expected -- the file is generated, not hand-edited.
- Architecture budget: the nine violations this branch authored were fixed by
  decomposition -- `CasadiSolveOptions` and `MaxSpeedOcpOptions` group the
  keyword arguments that pushed `solve_swing_casadi` and `build_max_speed_ocp`
  over the parameter budget, `_SwingModelSurface` moves the bioptim-independent
  half of the adapter to module level, and the parity benchmark is one function
  per backend. `crocoddyl_backend.solve_swing_ddp` is byte-identical to
  `origin/main` and carries a dated exception instead.
- CI state on PR #9768: the `hatchling` direct-reference fix (`52a8710`)
  cleared the eleven jobs that could not build the package at all. The
  `dependency-consistency` gate is red on `main` too, because #9716 added
  `openpyxl` and `imageio` to the `dev` extra without regenerating the
  locks; this branch carries the regenerated `requirements-dev.lock` and
  `environment.yml` so the gate passes here and no-ops once `main` catches
  up. Regenerate them only with Python 3.12 (`make sync-deps`) — that is
  the interpreter the gate runs, and 3.11 produces a different lock.
- `code-quality` runs `ruff` unpinned, so it floats ahead of the
  `ruff>=0.15.10` floor in `pyproject.toml`. 0.15.17 reformats a
  parenthesised lambda body in `benchmarks/bioptim_parity.py` that 0.15.8
  left alone; the committed form is stable under both.

## Vendor Pin & Alias Predicate: #9631

- `vendor/ud-tools` = Tools `eab74a901a`, carrying the Tools#5049 flattened-install alias fix
  (`f8b94bfe` is an ancestor); do NOT rewind — that drops Tools #5051-#5057 incl. the Sentinel
  ShellTool injection fix. Cargo rev, companion `pinned_commit`, `check_tools_pins.py` agree.
- `tests/unit/repo_hygiene/test_pinned_import_alias_contract.py` pins the flattened-install
  contract (RED at pre-#9657 pin `3d93bb2c`, GREEN at the current pin). Child copy stays
  unconverged per #9657. Remaining: re-cut the 2.1.3 release.

## In-Flight Issue Work

- `claude/issue-9612-video-suffix` (PR pending): `src/api/routes/video.py`
  derives the upload temp-file suffix from the filename against
  `SUPPORTED_VIDEO_SUFFIXES` and fails closed with 400 on unknown/missing
  extensions; state tracked in `docs/development/DEVELOPMENT_LOG.md` (DL-#9612).

## Protected Authority

- UpstreamDrift protected `main` is
  `3503674a90e3ca6d75e81f084f011299f5e95794`, the verified squash merge of
  turnover PR #9309. Its reviewed head and squash merge share exact tree
  `827608120154ecdfb6dc8b9f0c53b988a0454343`.
- #9306 adds signed-gap retention, bracketed opening/reattachment location on a
  declared linear state interpolant, duplicate-time event alignment, and the
  distributed replay adapter into the protected attribution kernel.
- #9308 adds the prospective six-case current-main smoke registration without
  executing or promoting an outcome. CI Standard run `33322181043`, Optional
  Stack run `33322180991`, and Bot CI Trigger run `33322180972` passed. Exact
  turnover evidence is recorded on #9153 in comment `issuecomment-5469958455`.
- PR #9302 remains the manufactured-evidence qualification parent at
  `7dc5f86af68907f19bc953c509d96d05f505cdab`; PR #9299 remains its CRBA
  requalification parent at `c8a283f4ffb408d5932bdc2da3f2f0c64665ef83`.
- The requalified paper has 253 pages, 2,011,818 bytes, and SHA-256
  `554fca211786ac5a06959f41b9f7d75720c89155168faeaac9d648524e8c9e36`.
  Tagged-PDF and embedded-font gates remain open archival limitations.
- AffineDrift #3993 pins the exact #9152 authority as protected squash
  `9b9cbcc2199f1fbf8cd281beb08c57d543b552b1`; handoff correction #3995 merged
  as `6cc909273d63147392b17078a35c6c4da034e1da`.
- Tools force-source frame #4873 merged as
  `cc883cbaf63157b58c71cba385a683df2762b0cb`; Tools #4142 remains the broader
  reusable-variation completion authority.

## Impact Explorer Web Route Producer: #9484

- PR: #9724 (open against `main`, `Fixes #9484`).
- The `rate_of_closure` tile declares `web.mode: route` for
  `/tools/impact-explorer`; `src/api/local_server.py` mounts
  `vendor/ud-tools/src/rate_of_closure/web/dist` when it exists. CI Standard's
  `impact-explorer-web-build` job now builds that bundle from the pinned Tools
  tree (`npm ci`, then `npm run build -- --base=/impact-explorer-app/`) and
  `scripts/check_declared_route_producers.py` fails any launcher route that no
  pipeline produces (unit tests in `tests/scripts/test_declared_route_producers.py`).
- Open decision (maintainer, #9417): whether the built bundle ships inside the
  wheel/image or is fetched as a Tools release artifact; this work deliberately
  does not re-architect distribution. The vendored build authority is Tools'
  `.github/workflows/rate-of-closure-web-distribution.yml` (node 22, `npm ci`
  in `src/rate_of_closure/web`); the base path comes from the route fallback
  in `ui/src/pages/ImpactExplorer.tsx`.

## Active Hybrid Authority Repair: #9236

- The candidate adds a study-scoped, hash-locked CPython 3.11.15 manylinux
  environment and an isolated runner-temp native job. The canonical serializer
  is deterministic, rejects NaN, writes atomically, and records the exact
  profile. Semantic comparison requires the complete six-gate tolerance set,
  the exact 18-field rolling compatibility policy, every governed result path
  in both records, finite numeric leaves, and an internally derived maximum for
  each record. Missing paths, policy entries, stale profiles, inconsistent
  maxima, negative residuals, and non-monotone convergence fail closed.
- The committed scientific record, checksum, release manifest, and
  claim-evidence manifest are byte-identical to `origin/main`. The next step is
  an ordinary protected PR that runs the exact locked Linux CPython 3.11.15
  native job. Accept a regenerated record only when two same-environment builds
  are byte-identical, the declared dependency/profile identity matches the job,
  all numeric evidence satisfies the declared comparison policy, and the
  checksum/manifest cascade is regenerated by canonical commands.

## Protected Prospective Smoke Registration: #9153

- PR #9308 protected-squash-merged as
  `651de90a4cf8e1195ec7f3ab3ae16883ec8f6172`. Reviewed head
  `142662fb3de86fbb8086b83772df09425897f9c2` and merge share tree
  `a0dd17ed057c0aea45f112172d406ace971610df`; remote `main` equals the squash.
- PR #9309 changed only this root handoff and the detailed turnover document;
  it introduced no runtime, smoke outcome, scientific promotion, human
  validation, or coaching authority. Any post-merge turnover correction PR is
  recorded in the latest #9153 comment rather than self-referentially embedded
  here; verify it is protected before implementation resumes.
- New registration source:
  `scripts/research/proximal_distal_energy/articulated_distributed_smoke_registration.py`.
- New frozen protocol:
  `docs/research/proximal_distal_energy_transfer/data/articulated_distributed_smoke_registration.json`.
- The prospective matrix contains six cases: MuJoCo and Pinocchio at 1.0,
  0.5, and 0.25 ms, using source case 0/sample 6, one station per hand, 1.5 mm
  slack, zero generalized initial velocity, 1 mm club displacement, and
  -0.8 m/s initial club velocity over 50 ms.
- Execution is explicitly `not_started`; retained outcomes are empty and
  promotion authority is none. The registration binds the protected evaluator
  revision/tree, seven evaluator-source hashes, and the 35,568-byte input NPZ
  with SHA-256 `9fa4364571ba5535995c63226289c0711ee1ebf37c58b7a3b4e4d14a98561779`.
- Only opening and reattachment are eligible. Friction-limit, static stick/slip,
  inferred discrete impact, causal counterfactual, biological, human, and
  coaching interpretations are prohibited.
- TDD evidence: the missing module produced the expected RED import failure;
  after implementation, four registration tests pass serially.

## Spec Check Reminder Fail-Safe Extraction (#9499)

- The `Verify SPEC.md freshness` job no longer carries its comment-posting
  logic as an inline `actions/github-script` heredoc (an unescaped backtick
  from the RM #1520 wording once aborted it with `SyntaxError: Invalid or
unexpected token`, swallowing the finding). Posting now runs
  `scripts/post_spec_reminder.py`, which prints the full diagnostic into the
  job log and exits 0 on any posting failure; the `always()`-guarded
  "Fail if spec is stale" step owns the non-zero exit.
- Contracts pinned by `tests/ci/test_spec_check_workflow.py`.

## Immediate Order

1. Verify the latest turnover-correction PR recorded on #9153 is protected on
   remote `main`; preserve its existing runs and auto-merge.
2. From that exact protected base, implement the current-main single-worker
   atomic smoke runner with RED contracts for registration identity, complete
   case enumeration, atomic resume, typed failures, and no outcome promotion.
3. Execute the six registered cases only after runner code and tests merge.
   Never import or relabel legacy checkpoints as outcomes.
4. #9483: the stale 15-tile nav-gap audit `reports/feature_navigation_gaps/` is deleted (findings dispositioned in the deleting PR); live tile truth is the `src/config/models.yaml` registry plus the generated launcher manifest (#9412/#9437/#9478).
5. Regenerate `requirements*.lock`/`environment.yml` via dispatch-only `lock-refresh.yml`
   for #9533 (DL-#9533, PR #9716) once it merges; the locks were left untouched there.

## Spec Merge-Driver Vendoring (#9476)

- `scripts/install_spec_merge_driver.py` and `shared_scripts/spec_changelog.py`
  are re-vendored from Repository_Management#1521's corrected copies and pinned
  byte-identical by `tests/unit/scripts/test_spec_merge_driver_vendor_drift.py`;
  `scripts/setup_hooks.py` now calls the installer (issue #9476), so the
  documented setup registers the `spec-rows` driver, and the installer
  docstring names this repository's entry point.

## Scientific Boundaries

- Event locations qualify the retained discrete trajectory only; they are not
  the continuous integrator's exact event solution.
- Compliant opening/reattachment records zero discrete event impulse and work by
  model definition; it is not evidence of a physical impact.
- Same-trajectory attribution is descriptive. Divergent forward
  counterfactuals require a separately registered design.
- Energy transfer, momentum redistribution, joint work, contact power, event
  timing, and clubhead speed are distinct estimands.
- Native-engine agreement verifies declared operators and the common contact
  law; it does not calibrate anatomy, grip, shaft, ground, equipment, or human
  strategy.
- #8556/#9004 remain governed human-data boundaries. Synthetic evidence cannot
  substitute for bilateral six-axis participant grip wrenches.

## Frozen External Boundary

- #8800 remains frozen at source `1bd4d57da7bd257b76b42b3cc19524b283b5f748`; only 93/830 checkpoints exist.
- ControlTower ground stopped at 45/48 and shaft at 48/48. Its WSL VHDX is unreadable (`0x80070570`). Do not retry WSL, repair/mount/copy/mutate the VHDX, restart services, or launch a replacement without explicit approval and a recoverability plan.
- DeskComputer remains runner-drained; keep tests serial and web tests at no more than two workers.
- The accumulated campaign worktree is `UpstreamDrift-worktrees/9153-forward-impulse-work`, last remote-equal at `1e5e823ca2fa9391134e8a0ccf140a36036a88a7`, with 233 commits ahead and 48 behind at the last audit. Preserve its evidence and quarantine.

## Active AffineDrift and External Program Boundaries

- Markerless Mocap Program (#9063): Tools #4706 owns capture/contract schemas;
  UpstreamDrift #9069 (folded into #9422) owns app orchestration; makes no physical-lab qualification claim.
  Rig bring-up evidence: `docs/motion_capture/usb_camera_rig_bringup.md` (#9586); consumer slices #9589–#9592.
- Foundation #9180 merged as `1af18489e8755933a0d189aa8edafe787fa94d0f`; publication #9214 merged as `a8073c42edc811522c5d5709744f55c5cbd0fa8e`.
- Governed companion workflows (#9190) define the 15-record registry, public executor, and CI execution evidence across 10 success and 4 failure fixtures.
- #9222 has exact tree `c468c0db`, but its protected-main run was cancelled with no jobs or artifacts. #9192 remains open pending post-#9236 exact bytes; #9174 remains open.
- ADR-0043 and schema v1 are one-way UpstreamDrift software-fact authority for AffineDrift #4010. #9064 remains design-manual authority and #9070 remains typed calculation-manifest authority.

## Validation

Use `C:\Users\diete\AppData\Local\Programs\Python\Python312\python.exe` with `-n 0` for pytest.

```powershell
python -m pytest -n 0 -q tests/research/test_articulated_distributed_smoke_registration.py
python -m pytest -n 0 -q tests/research/test_articulated_contact_events.py
python -m pytest -n 0 -q tests/research/test_articulated_forward_attribution.py
python -m scripts.research.proximal_distal_energy.articulated_distributed_smoke_registration validate
python scripts/check_document_title_case.py --changed-from origin/main
python scripts/ci/check_file_size_budget.py
python scripts/ci/check_architecture_budget.py
```

Also run claim/evidence integrity, release qualification, PDF inspection, and
affected full gates after publication changes. Never force-push, bypass branch
protection, relax tolerances after inspecting results, or create capacity-only
reruns. Do not restart the Actions runner or start WSL.

## UI Dependency Pin: #9249

- Dependabot now ignores `@vitejs/plugin-react` major updates (`.github/dependabot.yml`): 6.x needs Vite 8 (`peerDependencies.vite: "^8.0.0"`; Vite 7 exports no `./internal`), so a lone bump cannot merge. Stay on plugin-react ^5 with vite ^7.3.2 until a paired Vite-8 upgrade; pairing note lives in `ui/README.md`. Branch `claude/issue-9249-ui-pin`.

## OCP Compat Robust to Poisoned `sys.modules` (#9771)

- `tests/unit/conftest.py` installs spec-less `casadi`/`pinocchio` MagicMocks
  process-wide in `pytest_configure` and never removes them, so a lane
  collecting `tests/unit/optimization` and `tests/integration/optimization/ocp`
  together made `_compat.bioptim_available()` report the genuinely installed
  stack as absent and silently skip 16 bioptim ocp tests.
- `src/shared/python/optimization/ocp/_compat.py` now probes top-level names
  through `importlib.machinery.PathFinder` (ignores `sys.modules`), so a mock
  hides a real distribution only when no real one exists behind it;
  `require_bioptim()` evicts mocked `casadi`/`bioptim`/`biorbd_casadi`
  entries before importing so bioptim binds the genuine modules (the unit
  tree's autouse fixture reinstalls its own mocks per test, so unit-tree
  degradation semantics are unchanged).
- Contracts: `tests/integration/optimization/ocp/test_compat_poisoned_sys_modules.py`.
  Note the local reproducer cannot exercise real casadi/bioptim (Python 3.14
  has no casadi wheel); CI lanes with the `[bioptim]` extra must show the
  ocp bioptim legs running (not skipping) in combined lanes.
