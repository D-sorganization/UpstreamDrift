# Agent Handoff — UpstreamDrift

Last updated: 2026-08-04
Update this file with every PR and every push to main.

## Where the repo is heading

- **Repository_Management#1390** ("EPIC: Fleet-wide Agent Handoff & PR Policy") — this PR
  adds this file plus the `AGENTS.md` policy section. Status: in progress (this PR).
- **#8345** ("EPIC: 3-D Putt Simulation") — open, labeled `claim:claude`. Five phases (P1–P5):
  R3F 3-D putt scene/collision viz, advanced surface+friction model
  (`src/shared/python/putting_dynamics/`), two-body putter/ball collision model,
  a public research-data review doc, and a public-sharing build. Not started yet.
- **#8339** ("Rate of Closure Impact Explorer") — merged. `vendor/ud-tools` submodule
  pin was provisional pending Tools#4092; confirm the submodule now points at the
  squash-merge commit on Tools `main`, not the old branch-head SHA, before relying on it.
- Several open `bolt-*` PRs (#8334, #8335, #8336, #8341) — small automated numpy
  micro-optimizations (vdot/ndarray.sum/boolean reduction) in trendline/curve-fit code.
  Independent, not stacked.
- #8344 — physics fix (impact friction spin axis / gear-effect offset), independent.
- Dependabot PRs #8329–#8332 — routine GitHub Actions version bumps.

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

None of the currently open branches stack on each other — all are independent topic
branches off `main`:

- `docs/agent-handoff-1390` (this branch) — off `origin/main`.
- `fix/impact-friction-spin-axis-and-gear-offset` (#8344) — off `main`.
- `bolt-vdot-optimization-*`, `bolt/trendline-boolean-sum-optimization-*`,
  `bolt/ndarray-sum-optimization-*`, `bolt/optimize-rsquared-vdot-*` — each off `main`,
  independent Bolt micro-optimization PRs.
- `dependabot/github_actions/*` — off `main`, routine version bumps.

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
4. Begin #8345 EPIC phase P1 (3-D putt scene & collision visualization) once claimed
   work is scheduled — see the phase breakdown in the issue for P2–P5 ordering
   (surface/friction model → collision model → data-review doc → public sharing).
