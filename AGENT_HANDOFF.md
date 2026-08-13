# Agent Handoff — UpstreamDrift

Last updated: 2026-08-11
Update this file with every PR and every push to main.

## Where the Repo Is Heading

- **#8511 interactive proximal--distal dynamics workbench** — branch
  `feat/proximal-distal-workbench-integration` makes the pendulum launcher
  consume the canonical sibling Tools provider and dockable adapter. Tools
  owns both PyQt6 and React/Tauri clients plus the shared experiment/glossary
  catalog; UpstreamDrift owns the article integration, provider resolution,
  immutable vendor pin, and evidence boundary. Focused launcher tests pass.
  Remaining gates: merge Tools, pin its exact main commit, publish the
  AffineDrift reader page, run scoped repository gates, then protected merge.

- **#8490 launcher UI-setup size slice — ready PR
  [#8492](https://github.com/D-sorganization/UpstreamDrift/pull/8492)** — branch
  `codex/launcher-ui-setup-decomposition` moves sidebar/navigation/menu construction into `_launcher_navigation_ui.py` and top-bar status/search/runtime/view/zoom construction plus the historical widget types into `_launcher_top_bar_ui.py`.

- **Launcher-settings size slice — ready PR
  [#8489](https://github.com/D-sorganization/UpstreamDrift/pull/8489)** — branch
  `codex/settings-dialog-size-decomposition` extracts runtime dependency probes into `settings_runtime.py`.

- **#8485 Simscape 3D-viewer size slice — ready PR
  [#8486](https://github.com/D-sorganization/UpstreamDrift/pull/8486)** — branch
  `fix/viewer-3d-module-size-decomposition` moves user-defined body-segment shape construction into `_viewer_3d_segments.UserSegmentRenderer`.

- **#8483 main-launcher size slice — ready PR
  [#8484](https://github.com/D-sorganization/UpstreamDrift/pull/8484)** — branch
  `fix/upstream-launcher-module-size-decomposition` moves Sidekick API-readiness into `SidekickSidebarManager`.

- **#8476 Sentinel fix** — branch
  `fix/auth-timing-attack-16287999036686770098` fixes user enumeration via timing attack in login endpoint.
- **#8490 launcher UI-setup size slice — ready PR
  [#8492](https://github.com/D-sorganization/UpstreamDrift/pull/8492)** — branch
  `codex/launcher-ui-setup-decomposition` starts at exact PR #8489 head
  `2f664d2beaddf7444b12f90080ae9897aea24fcc`; reviewed implementation commit
  `ff7d937ccce767c432c53ef21e2193807ee77fdb` is published. It moves
  sidebar/navigation/menu construction into `_launcher_navigation_ui.py` and
  top-bar status/search/runtime/view/zoom construction plus the historical
  widget types into `_launcher_top_bar_ui.py`. `UISetupManager` keeps all 62
  historical methods exactly once across the facade and two inherited mixins;
  the public widget exports, dynamic manager-to-launcher method rebinding,
  monkeypatch-sensitive zoom description/window-control seams, and zero-argument
  `super()` behavior remain compatible. Independent-review regressions now
  prove that runtime zoom/menu builders dispatch through narrow facade hooks;
  the private mixins retain standalone defaults without importing the facade.
  The responsive source contract inspects the extracted search/zoom helper
  owners while retaining its original clipping assertions. The facade is now
  995 lines (down from 2,263), and its file-size/module-size exceptions plus
  four moved long-function exceptions are removed without renewal. The module-size,
  file-size, architecture, error-handling, suppression, TODO, and LoD gates are
  green, as are the focused decomposition and launcher source-contract tests.
  The official repository MyPy wrapper excludes `launcher_ui_setup.py`,
  `_launcher_navigation_ui.py`, and `_launcher_top_bar_ui.py`; the clean wrapper
  skip is not evidence that these modules are type-safe.
  The broader launcher contract selection has the same 15 failures as the exact
  parent, all caused by pre-existing shared theme/style export drift. The
  suite-marker and DRY duplication gates also fail identically to that exact
  parent. The suite-marker output is 275 lines with identical SHA-256
  `a47813dfc45d70ebf231c1a4fd5a9dd89d9b5931f2fe563c275ed614b4dfa391`;
  the duplication output is 1,571 lines with identical SHA-256
  `73efdb450b2dadcd6261ab27b91d01c7bdf44c2f053e703ceda49bbc24a44fb3`.
  Two independent review passes found and then verified closure of the
  responsive-source, facade-seam, and MyPy-disclosure defects. The ready PR is
  not released: protected CI, required human approval, and parent #8489 remain
  open. Issue [#8490](https://github.com/D-sorganization/UpstreamDrift/issues/8490)
  closes only after an ordinary protected merge; do not reuse closed #5922 or
  #7399.

- **Launcher-settings size slice — ready PR
  [#8489](https://github.com/D-sorganization/UpstreamDrift/pull/8489)** — branch
  `codex/settings-dialog-size-decomposition` starts at exact PR #8486 head
  `624043537a5ab10aa7ef56dc61685a004b872c0c`; published head
  `832969ebbd6c58c9892dc16f82638e67a05b20dc` is tracked by
  [#8487](https://github.com/D-sorganization/UpstreamDrift/issues/8487). It
  extracts runtime dependency probes and the WSL setup dialog into
  `settings_runtime.py`, and extracts
  diagnostics, log synchronization, and process-management behavior into the
  private `_settings_auxiliary_tabs.py` mixin. `SettingsWidget` preserves its
  constructor, signal, tab constants, control attributes, historical methods,
  and runtime compatibility imports. `settings_dialog.py` is now 1,124 lines
  (down from 2,190); its file-size and expired module-size exceptions plus the
  obsolete `WslScriptDialog._setup_ui` long-function exception are removed
  without renewal. All 44 focused settings/launcher contract tests pass, as do
  changed-file Ruff/format/compile, architecture, file-size, and error-ratchet
  gates. A wider 166-test launcher selection is 148 passed and 18 inherited
  failures caused by shared theme/UI export drift; the exact parent produces
  the same 18 failures (145 passed without the three new decomposition tests).
  The unrestricted launcher suite also reaches the same Windows access
  violation in `test_run_launcher` on the exact parent. The official Python
  3.12 MyPy wrapper excludes all three launcher modules by repository policy.
  The only remaining global module-size failure is the parent-identical expired
  `launcher_ui_setup.py` exception. The cited #5922 and #7341/#7342 issues are
  closed or unrelated; #8487 is now the truthful tracker. Independent review
  found no actionable regression and reran all 44 focused tests successfully.
  PR #8489 is ready for review, but required protected CI, approval, parent
  dependency, issue completion, and release state remain unresolved; do not
  claim any of them before normal protected repository behavior confirms them.

- **#8485 Simscape 3D-viewer size slice — ready PR
  [#8486](https://github.com/D-sorganization/UpstreamDrift/pull/8486)** — branch
  `fix/viewer-3d-module-size-decomposition` is stacked on ready PR #8484 at
  exact base head `89f87590981f789755c2b45e1b03ed2ee57247a3`; its exact current
  published head is `624043537a5ab10aa7ef56dc61685a004b872c0c`.
  It moves user-defined body-segment shape construction, fitting,
  library/theme resolution, artist lifecycle, and per-frame updates into the private
  `_viewer_3d_segments.UserSegmentRenderer`; `Viewer3DTab` retains its existing
  public segment methods as thin delegates. The viewer facade is now 1,127
  lines (down from 1,413), and both its file-size and expired module-size
  exceptions are removed without renewal. All 71 focused viewer/UI and 31
  budget-contract tests pass after initializing the repository's pinned
  `vendor/ud-tools` gitlink; the broader C3D/UI selection is 159 passed, 6
  skipped, and one parent-identical stale loader-message assertion. Protected
  CI remains incomplete with queued and cancelled contexts, so this is not yet
  merge-ready. The module-size gate still
  reports only the inherited `launcher_ui_setup.py` and `settings_dialog.py`
  violations.

- **#8483 main-launcher size slice (local, no publish)** — branch
  `fix/upstream-launcher-module-size-decomposition` starts at exact draft PR
  #8482 head `73dd11df09a2f37ea150835930134ae4354ee5a7`. It moves the
  Sidekick API-readiness state machine, terminal degradation report, and
  workspace-registry seeding into the existing `SidekickSidebarManager`, while
  preserving the historical `UpstreamDriftLauncher` methods as thin delegates
  through the single manager owned by the launcher, and keeping the clock,
  readiness probe, and Qt scheduler injectable. The
  launcher facade is now 1,198 lines (down from 1,315), and its file-size
  exception is removed without renewal. The module-size gate now reports three
  inherited oversized modules rather than four. Issue #8483 now provides
  accurate tracking; the retired exception had cited #7399, which is a closed,
  unrelated body-part-visualization PR. Remaining module-size failures are
  `launcher_ui_setup.py`,
  `settings_dialog.py`, and `viewer_3d_tab.py` with expired exceptions. The 25
  focused Sidekick-startup tests pass. The broader launcher selection remains
  blocked by the parent-identical Tools ownership mismatch for the already
  present `chat/_qt/runtime.py`; changed-file mypy also timed out under Python
  3.12 after the Python 3.13 run stopped in NumPy's version-gated stub.

- **Launcher-diagnostics size slice (local, no publish)** — branch
  `fix/launcher-diagnostics-size-decomposition` starts at exact draft PR #8480
  head `971649efd5ad2e5793240a5237a0314d45cc2faf`. It moves the local
  `vendor/ud-tools` pin/checkout/sibling/remote comparison into the focused
  `launcher_shared_tools_diagnostics` module while retaining
  `LauncherDiagnostics.check_shared_tools_freshness()` as the public recording
  boundary. `launcher_diagnostics.py` is now 1,196 lines, and its expired
  module-size exception plus its file-size and long-function exceptions are
  removed without renewal or policy widening. Do not describe this as closing
  an issue: cited maintainability issue #5922 is closed, cited #7341/#7342 are
  closed and concern Docker cancellation/layout reset, and open #8472 is scoped
  specifically to the chat dock. Confirm or create accurate tracking before
  publication. Three expired exceptions and four oversized production modules
  remain in the module-size release gate. The legacy
  `tests/unit/test_launcher_diagnostics.py` also has three parent-identical
  stale assertions (17 versus the current 48 models and the former product
  title); keep that separate from this structural slice.

- **#8472 chat-dock decomposition (local stacked candidate)** — branch
  `fix/8472-chat-dock-decomposition` is stacked directly on the #8479 parent
  commit `2c98644d3ef3e32820eb6c2df80e75250593392b`. It moves WebSocket event
  routing, terminal-mode mechanics, streaming-state initialization, and
  collapsed-view mutation into `chat._qt.runtime`, retaining the historical
  `ChatDockWidget` methods as thin delegates. The compatibility shell is now
  1,150 lines, and both its file-size and expired module-size exceptions are
  removed without renewal or policy widening. Focused behavior tests and the
  file-size gate pass. The parent already fails the Tools drift sentinel for
  `_chat_dock_widget_qt.py` and `models.py`; synchronize this decomposition to
  the canonical Tools source before updating that hash. After the local
  launcher-diagnostics slice, the repository module gate remains red with
  three expired exceptions/four oversized modules: launcher UI, settings, the
  main launcher, and the Simscape 3D viewer.

- **Exact-main release-gate unblocker candidate (local only)** — branch
  `fix/current-main-release-gate-unblockers` starts at exact `main`
  `69eb7e9db32ccd17e45824619315b1d04b400c27`. It repairs the two
  `CanonicalCoreShell` ESLint violations without suppressions, rejects stale
  status responses across mode changes, corrects the stale durable-task-manager
  SPEC paths after #8322 removed that implementation, and locks the three
  vulnerable transitive npm packages to patched same-range releases. It also
  restores the engine-store unload tests' mocked backend boundary after the
  earlier `apiFetch` migration left two tests attempting relative-URL network
  calls under Node. Focused React regression tests, the full React suite, lint,
  type-check, build, audit, and SPEC path checks must be green before
  publication. The expired/oversized Python module-size
  exceptions remain an inherited release blocker: this branch neither renews
  nor widens them. #8472 owns only the chat-dock split; the remaining modules
  need accurate open tracking rather than being attributed to that issue.

- **#8458 hand-path drift/control attribution** — consolidated branch
  `feat/hand-path-drift-control-attribution` defines the canonical same-state
  ZTCF/control/ZVCF contract; implements exact double-pendulum, one-arm, and
  mechanically closed two-arm adapters; and generates a deterministic
  joint/time-resolved evidence package with force vectors, impulse, power,
  work, cancellation-safe shares, common/differential two-hand modes, source
  hashes, and numerical closure. The two forward cases attribute 101.2% and
  103.5% of signed primary force work to drift because control is opposing. The
  prescribed two-arm local sweep instead has a 0.962 drift/control cancellation
  index. A separate bounded residual-couple preview test reduces the declared
  30 ms actuator RMSE by 57.6%; it is a signal-delay hypothesis, not evidence
  of muscle preactivation or human performance. The seven generated evidence
  figures and all pages of the final 106-page PDF have been visually inspected.
  Lossless object-stream compaction yields 792,985 bytes while preserving 110
  URI links and 122 outline entries. Remaining handoff: run the final repository
  gates, merge the protected PR, then pin the compact consumer snapshot and
  SVGs in AffineDrift to the actual merged commit. The unrelated chat-dock size
  exception had expired on `main`; #8472 now owns a final renewal through
  2026-08-31 so the all-files CI gate remains truthful and operational.

- **#8461 WSCG 2024 legacy-evidence audit** — both archived source decks were
  hash-matched to the user-supplied originals and all 12 slides were inspected.
  `docs/research/proximal_distal_energy_transfer/WSCG_2024_LEGACY_EVIDENCE_AUDIT.md`
  records the exact claims, pointwise ZTCF construction, two-hand couple
  mechanism, OOXML chart-cache provenance, and publication boundaries. The
  evidence supports passive late-downswing negative-torque plausibility in the
  planar model, not momentum isolation or human validation. No legacy image was
  reused: slides 7–10 are static JPEGs, the chart's source workbook is absent,
  and the slide-3 composite has no license note.

- **#8448 higher-order mechanism ladder** — first slice merged in full PR
  **#8456** at remote-main commit `85eae6a8ef1b132f93eda87bef2a2d6d51280c49`.
  The executed slice adds a
  common frame/reference-explicit wrench-power schema, exact frame and
  reference-transport audits, prescribed mobile-hub inverse dynamics, planar
  two-hand constraint rank/nullspace diagnostics, seven figures, and a model
  discrepancy table. The evidence record deliberately marks full-body
  cross-engine dynamics `not_executed`; capability is not reported as a result.
  Follow-up full PR **#8457** losslessly compacts the 90-page PDF with page,
  link, outline, and visual-equivalence checks. The later publication policy
  removed the obsolete 1 MiB ceiling while retaining artifact-integrity and
  GitHub hard-boundary controls.

- **#8447 gravity, momentum, damping, and shaft-flex separation** — full PR
  **#8455** on `research/momentum-gravity-shaft-flex`. A tested three-coordinate point-mass
  surrogate now has an exact matched rigid reduction, termwise acceleration and
  power attribution, interface force/moment accounting, closed work--energy
  balance, gravity and damping ablations, a 120-case
  stiffness/damping/torque-cut grid, impact-window and timestep sensitivity,
  eight figures, and a new publication chapter. The bounded reference result is
  +0.108 m/s flexible-minus-rigid delivery speed with 0.720 J peak shaft strain
  energy; gravity and joint-damping ablations are substantially larger. This is
  not a calibrated shaft or human-subject model.

- **#8446 two-hand passive-couple reproduction** — full PR **#8454** on branch
  `research/two-hand-passive-couple`. The archived 2,801-sample
  BASE/ZTCF/DELTA tables now have hash-traceable portable exports, a tested
  frame-explicit wrench and power audit, reversal/downsampling sensitivity,
  grip-separation and relative-orientation counterfactuals, eight figures, and
  a publication chapter. The key bounded finding is a -19.63 N m pointwise ZTCF
  midpoint couple generated entirely by separated contact forces; it is not yet
  a forward two-hand killswitch or human validation result.

- **#8445 counterfactual persistence** — full PR **#8453** on branch
  `research/counterfactual-killswitch-ensemble` adds a deterministic
  matched-state API; 96 cut/horizon/timestep cases; gravity, damping, and
  torque-switch audits; WSCG DELTA convention checks; four figures; and a
  visually verified 58-page article. Pointwise ZTCF is explicitly limited to
  instantaneous attribution; killswitch claims always state a forward horizon.

- **#8443 / #8444 interaction-force mechanisms** — full PR **#8452** on branch
  `research/interaction-force-transfer` adds an exact and tested
  double-pendulum wrist-force/power decomposition, matched-state torque
  killswitch, seven vector/mechanism figures, hash-registered WSCG 2024 source
  decks, and a 52-page rendered article. The source presentation is treated as
  project-originated hypothesis evidence, not independent validation. Follow-on
  issues #8445–#8451 cover counterfactual ensembles, two-hand equivalent-couple
  reproduction, gravity/flex separation, higher-order models, optimization,
  human validation, and the 90–110-page open monograph release.

- **Document title capitalization** — this branch normalizes the
  Proximal-to-Distal article headings and regenerated PDF/LaTeX, adds a
  changed-document gate to pre-commit and Docs Governance CI, and records the
  fleet convention in `AGENTS.md`. The full tracked-document audit command is
  `python scripts/check_document_title_case.py`.

- **Tools #4276** — PR **#8440** remains a partial, headless, fail-closed
  consumer for the canonical Tools ground v1 façade while preserving Tools
  records/provenance. Its exact published head `e2f436beebc3c2739dcc5f06b5efe5e130513c65`
  has been reconciled locally by a normal merge of current `main`
  `69eb7e9db32ccd17e45824619315b1d04b400c27`; the resulting local candidate is
  not published or release-qualified. Final Tools merges and exact vendor/Cargo
  pins, FastAPI/PyQt/React parity, clean-install smokes, protected CI, and
  independent current-head review remain open.

- **#8432** (`feat/launch-monitor-flexible-analysis`, replacing draft #8369)
  adds a versioned,
  vendor-neutral flexible-analysis contract with matched FastAPI and PyQt
  surfaces. It keeps aggregate observations out of regression, labels
  association as non-causal, validates option enums at the API boundary, and
  records deterministic dataset lineage. React/Vite parity and the final
  canonical Tools dependency pin remain tracked work rather than implied
  capabilities.

- **#8426** ("Proximal-to-distal swing mechanics — validation and open resource
  roadmap") — open. Full PR **#8428** neutralizes the reader-facing report, adds
  a 13-case model-parameter sensitivity analysis, and updates the rendered PDF.
  Remaining counterfactual-parity, model-fidelity, and human-data work stays open.
- **#8345** ("EPIC: 3-D Putt Simulation") — open. P2/P3/P4 are implemented in
  full PR **#8352** (`feat/putting-dynamics`): an advanced
  surface/friction/mode-machine package,
  finite-mass collision with loft and adjustable-hosel wrench/twist outputs, and the
  public-data review in `docs/physics/PUTTING_KINEMATICS_KINETICS_REVIEW.md`.
  Local evidence: 70 focused pytest tests, Ruff clean, and Python 3.12 mypy clean.
  P1 is implemented in full PR **#8354** (`feat/putting-3d-scene`): canonical FastAPI
  playback DTOs, generated TypeScript types, Zustand/TanStack state, theme-token R3F
  scene, visible ball spin and putter slowdown, orbit/playback controls, adjustable
  hosel/CG view, and desktop/mobile rendered QA. P5 public sharing remains.
- **#8339** ("Rate of Closure Impact Explorer") — merged. `vendor/ud-tools` submodule
  pin was provisional pending Tools#4092; confirm the submodule now points at the
  squash-merge commit on Tools `main`, not the old branch-head SHA, before relying on it.
- **#8353** (`fix/classic-launcher-missing-tools`) — full PR. Fixes classic-launcher startup
  from nested worktrees by locating the workspace-level Tools checkout. If no valid
  implicit Tools runtime exists, only the optional Sidekick sidebar is disabled;
  explicit `TOOLS_REPO_PATH` selections remain fail-closed. Commit `6699380d9` has
  28 focused launcher/overlay tests plus clean Ruff checks.
- `chore/consolidate-open-pr-backlog` (full PR **#8431**) consolidates the still-applicable changes from
  micro-optimization PRs #8335, #8371, #8408-#8411, and #8424 together with the
  GitHub Actions updates from #8329-#8332. Superseded or duplicate PRs are closed
  only after their exact replacement is linked.
- #8344 — physics fix (impact friction spin axis / gear-effect offset), independent.
- PR #8422 is independently synchronized to current `main` and retains only its
  focused ground-reaction-force reduction change plus the required SPEC entry.

## Must-Read Architecture Pointers

1. `CLAUDE.md` — authoritative contributor/agent policy: gate commands, CI requirements,
   error-handling ratchet, feature-parity registry, physics-engine gotchas.
2. `AGENTS.md` — shared infrastructure catalog (FK, reference poses, mocap loaders, theme,
   rendering helpers); **discovery-first** workflow: grep `src/shared/python/` then
   `src/tools/` + launchers before writing anything new.
3. `docs/adr/0013-launcher-composability.md` — embeddable-tool contract/registry design.
4. `docs/adr/0007-motion-pipeline-architecture.md` — mocap → tracked-motion CIR pipeline.
5. `docs/adr/0016-*` (error handling) — see `scripts/ci/check_error_handling_ratchet.py`
   and `scripts/config/error_handling_baseline.json`.

## In-Flight Branches (What Stacks on What)

The active branches are independent topic branches off `main` unless noted:

- `feat/hand-path-drift-control-attribution` — consolidated implementation for
  epic #8458 and children #8459–#8471; target `main` after local render/gates.
- `research/shoulder-velocity-drift-transfer` — active implementation for epic
  #8551. The TDD implementation adds a 90-case, five-phase, two-counterfactual
  fixed-hub velocity sweep, exact drift/control and reaction-force closure,
  plus a 60-program trajectory search varying proximal-drive cut, residual
  proximal torque, and wrist release. All invalid impact attempts remain in
  the evidence; speed, negative grip work, and peak force define the Pareto
  objectives. Six generated figures and the paper chapter
  `#sec-shoulder-velocity-transfer`. The current model coordinate is proximal
  link angular velocity, not anatomical shoulder or thorax velocity. The
  trajectory grid contradicts proximal speed as a standalone release rule.
  Tools #4406 supplies the model-neutral metrics and PyQt Drift Transfer tab;
  AffineDrift #3817 supplies the publication surface. A rotating-base two-hand
  tier remains the next model required for a causal torso-velocity test.
- `research/two-hand-passive-couple` — full PR **#8454** for #8446 under epic
  #8443; direct WSCG table reconstruction and passive equivalent-couple audit.
- `research/counterfactual-killswitch-ensemble` — full PR **#8453** for #8445
  under epic #8443; multi-phase persistence and numerical/physics sensitivity.
- `research/interaction-force-transfer` — full PR **#8452** for #8444 under
  epic #8443; exact interaction-force mechanics and first detailed article
  treatment.
- `fix/8429-private-launch-data` — issue #8429, full PR **#8430**. Removes the
  public 832-shot launch-monitor CSV and resolves it through the authenticated
  private data authority. Focused validation:
  `tests/unit/validation_pkg/test_kaggle_validation.py`.
- `feat/putting-3d-scene` — #8345 P1, full PR **#8354** to `main`
  after #8352 merged. Scoped evidence: 80 Python tests and 17 UI/theme tests pass; strict
  API mypy, Ruff, generated-type freshness, ESLint, TypeScript, color guard, and
  production build pass. The full UI baseline remains 781 passed / 2 unrelated
  `useEngineStore` unload failures, reproduced on the parent checkout.
- `fix/impact-friction-spin-axis-and-gear-offset` (#8344) — off `main`.
- `chore/consolidate-open-pr-backlog` (full PR **#8431**) — clean branch from current `main`; replaces
  the applicable micro-optimization and Dependabot branches listed above without
  carrying their stale historical merge differences.
- `feat/4276-ground-consumer-adapter` — PR **#8440**, a partial Tools #4276
  consumer slice locally reconciled onto `main` `69eb7e9d`; no vendor pin,
  current-head review, or release claim until the protected Tools ground stack
  lands and the remaining consumer surfaces are qualified.

## Gate Commands (Run These Before Opening/Updating a PR)

```bash
python3 -m ruff check .                            # lint
python3 -m ruff format --check .                   # format check (separate CI step)
python3 -m pytest -n auto --timeout=60              # full test suite
python3 -m pytest -m unit -n auto --timeout=60      # unit tests only
python3 -m pytest -m "not slow and not live_simulation" -n auto --timeout=60
python3 scripts/ci/check_file_size_budget.py        # 1200-line file budget
python3 scripts/ci/check_error_handling_ratchet.py  # error-handling anti-pattern ratchet
python3 scripts/check_docs_governance.py            # docs-only PR gate (docs-ci.yml)
python3 -m scripts.generate_feature_parity_matrix   # after editing feature_parity.json
maturin develop                                     # build Rust extensions locally
# Ground consumer focus:
python3 -m pytest tests/unit/ground_model/test_consumer_gateway.py -q
```

CI entry points: `.github/workflows/ci-standard.yml` (full matrix: `code-quality`,
`security-scans`, `repo-structure-gates`, `unit-test-gate`, `quality-gate`, `tests`,
`rust-quality-gate`, etc.) and `.github/workflows/docs-ci.yml` (docs-only PRs, requires
`quality-gate` + SPEC.md freshness).

## Do-Not List

- **Do not edit `vendor/ud-tools/`** — it is vendored from Tools; changes are erased on
  the next `git submodule update`/vendor bump. Fix upstream in Tools instead.
- **Do not import `upstream_drift_tools`** in new `src/`/`tests/` code — it is a
  deprecated compat alias; use `sidekick` (enforced by
  `tests/unit/repo_hygiene/test_no_deprecated_imports.py`).
- **Do not let Tools import UpstreamDrift** — dependency arrow is one-way (UD vendors
  Tools; Tools cannot import UD). If Tools needs UD physics, promote via the documented
  vendor pattern; do not create a cycle.
- **Do not grow files past 1200 lines** (`scripts/config/file_size_budget.json` holds
  exceptions) or modules past the `module_size_budget_baseline.json` ratchet.
- **Do not add `try/except Exception: pass`, bare `subprocess.Popen`, unchecked
  `asyncio.gather`, or generic `RuntimeError`** — use the `core.process_safety` /
  `core.contracts.exceptions` helpers (see CLAUDE.md "Error handling").
- **Do not skip the agent-lease protocol** — this repo is touched by multiple agents
  (`claude`, `codex`, `jules`, `local`, `gaai`, `maxwell-daemon`); check/post a lease via
  `Repository_Management` scripts before starting work on a numbered issue (see
  CLAUDE.md "Multi-agent coordination").
- **Do not open PRs as drafts** — every PR must open ready-for-review (fleet policy,
  Repository_Management#1390).
- **Do not batch a day's work into one commit** — commit small, frequent, conventional
  commits.

## Short-Term Roadmap (Ordered)

1. Land this handoff-policy PR (Repository_Management#1390 rollout for UpstreamDrift).
2. Confirm `vendor/ud-tools` submodule pin from #8339 points at the Tools `main`
   squash-merge commit (not the old branch-head SHA).
3. Clear the small independent queue: `bolt-*` perf PRs, #8344 physics fix, dependabot
   bumps.
4. Review and merge PR **#8352** (`feat/putting-dynamics`) for #8345 P2/P3/P4.
   Important audit
   corrections include the 1.64 m/s full-chord capture bound, signed overspin settling,
   down-grain friction semantics, immutable field ownership, and consistent tangential
   impulse/backspin vector recomposition.
5. Review full PR **#8354** for #8345 P1; it consumes the
   curated `putting_dynamics` façade without duplicating physics in React. Then
   complete P5 public sharing and parity registration.
6. After Tools ground merges, repin its exact commit and finish #4276 UI/release
   gates. Closed draft #8369 is not a parent; replacement #8432 already merged.
