# Agent Handoff — UpstreamDrift

Last updated: 2026-08-11
Update this file with every PR and every push to main.

## Where the Repo Is Heading

- **#8497 arm--wrist torque allocation and preload** — branch
  `feat/8426-phase-9-forward-distributed-shaft` adds an exact same-state 8 N m
  actuator-allocation sweep, a separately declared dead-zone transmission
  model, preload and role-reversal sensitivity, a coupled forward modal-shaft
  tier, deterministic evidence, publication figures, and explicit bilateral
  wrench/stiffness/EMG/holdout falsifiers. The persistent-direction advantage
  is conditional; neither the proximal subspace nor slack is identified as a
  measured scapular or tissue property.

- **#8493 ground-reaction drift attribution** — isolated branch
  `feat/grf-drift-decomposition` adds a strict constrained-contact solve for
  configuration, velocity, control, and external-load reaction components;
  pointwise reaction ZTCF/ZVCF definitions; componentwise RMSE, normalized
  RMSE, R-squared, bias, and impulse-error metrics; and a deterministic
  fixed-support double-pendulum benchmark. The benchmark closes below
  `2e-13 N`; ZTCF yields componentwise R-squared values of 0.871 and 0.814 but
  large amplitude errors, and its vertical impulse exceeds the total because
  control is opposing. The new chapter maps the quantities to GRF, COP, and
  free-moment measurements and specifies a held-out human falsification
  protocol. It explicitly does not claim bilateral allocation or human
  validation. NotebookLM corpus review is still blocked by expired local
  authentication; independently verified primary sources are linked directly.

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
