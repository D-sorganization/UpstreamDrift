# Agent Handoff — UpstreamDrift

Last updated: 2026-08-10
Update this file with every PR and every push to main.

## Where the repo is heading

- **#8448 higher-order mechanism ladder** — full PR **#8456** on
  `research/higher-order-mechanism-ladder`. The first executed slice adds a
  common frame/reference-explicit wrench-power schema, exact frame and
  reference-transport audits, prescribed mobile-hub inverse dynamics, planar
  two-hand constraint rank/nullspace diagnostics, seven figures, and a model
  discrepancy table. The evidence record deliberately marks full-body
  cross-engine dynamics `not_executed`; capability is not reported as a result.

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

## Must-read architecture pointers

1. `CLAUDE.md` — authoritative contributor/agent policy: gate commands, CI requirements,
   error-handling ratchet, feature-parity registry, physics-engine gotchas.
2. `AGENTS.md` — shared infrastructure catalog (FK, reference poses, mocap loaders, theme,
   rendering helpers); **discovery-first** workflow: grep `src/shared/python/` then
   `src/tools/` + launchers before writing anything new.
3. `docs/adr/0013-launcher-composability.md` — embeddable-tool contract/registry design.
4. `docs/adr/0007-motion-pipeline-architecture.md` — mocap → tracked-motion CIR pipeline.
5. `docs/adr/0016-*` (error handling) — see `scripts/ci/check_error_handling_ratchet.py`
   and `scripts/config/error_handling_baseline.json`.

## In-flight branches (what stacks on what)

The active branches are independent topic branches off `main` unless noted:

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

## Gate commands (run these before opening/updating a PR)

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
```

CI entry points: `.github/workflows/ci-standard.yml` (full matrix: `code-quality`,
`security-scans`, `repo-structure-gates`, `unit-test-gate`, `quality-gate`, `tests`,
`rust-quality-gate`, etc.) and `.github/workflows/docs-ci.yml` (docs-only PRs, requires
`quality-gate` + SPEC.md freshness).

## Do-not list

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

## Short-term roadmap (ordered)

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
