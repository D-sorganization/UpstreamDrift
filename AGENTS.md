# AGENTS.md — Discovery Workflow & Shared-Infrastructure Directory

> **Read this first.** This file exists because we kept reinventing
> infrastructure that already lived in the repo (FK solvers, skeleton
> renderers, reference golfer poses, mocap loaders, theme constants).
> Tracked as issue #4377; updated as we discover new shared modules.

For binding repo policy (ruff, file-size budgets, branch naming, PR
targets, etc.) see [`CLAUDE.md`](CLAUDE.md). For background and
historical agent-workflow notes see
[`docs/development/agents.md`](docs/development/agents.md).

This file is narrower: a **discovery workflow** for agents and a
**directory of shared infrastructure** so we stop duplicating work.

## Document Title Capitalization

Use title case for every document title, subtitle, section heading, navigation
label, figure caption, and chart title: capitalize the first letter of every
significant word. Keep articles, coordinating conjunctions, and short
prepositions lowercase unless they begin or end a title or follow a colon.
Preserve acronyms, mathematical notation, units, filenames, and product names.
This convention applies to Markdown, Quarto, LaTeX, Word, and PDF outputs; edit
the canonical source and regenerate rendered artifacts. Run
`python scripts/check_document_title_case.py` for a full tracked-document audit.

### Engineering Design Manual Authority

The sole editable source for the calculation-level engineering design manual is
`manuals/upstreamdrift` QMD. Generated LaTeX, PDF, DOCX, and HTML are
non-editable artifacts; correct QMD and regenerate them through the qualified
toolchain. Before changing calculations, public scientific pathways, or manual
governance, read `scripts/config/design_manual_governance.json` and run
`python3 -m scripts.check_design_manual_governance`. Missing inventory,
freshness, provenance, page review, or approval keeps release blocked.

---

## A. Before you write new code — discovery workflow

When you're about to add functionality, run these five steps **in
order** before writing a line:

1. **Grep `src/shared/python/` for the concept.** Most cross-engine
   primitives live there (FK, reference poses, mocap loaders, theme
   constants, club-target structures, optimisation drivers).
2. **Grep `src/tools/` and `src/launchers/` for similar tools.** If a
   tile already does ≥60 % of what you want, extend it instead of
   forking.
3. **Read the relevant `__init__.py`** (public API surface) before
   importing internals. The `motion_matching/diagnostics/__init__.py`
   in particular is a curated façade that hides private helpers.
4. **Read the docs.** `docs/` has design notes for the bigger systems
   and `MATLAB_GOLF_MODEL_GUIDE.md` documents pitfalls (e.g. the
   cm-vs-inches Wiffle xlsx gotcha).
5. **Only then** propose new code. If you find yourself
   reimplementing something that lives in `src/shared/`, stop and use
   the shared version — even if it forces minor API massaging on your
   side.

> **The "I'll just write it inline, it's only 20 lines" trap is real.**
> Twenty lines today becomes a divergent re-implementation tomorrow
> when the shared module changes its sign convention.

---

## B. Shared Infrastructure Directory

Before adding functionality, read the [shared infrastructure directory](docs/agents/shared-infrastructure.md). It records reusable engine, motion, rendering, camera, theme and analysis modules with their public interfaces and design references. Add newly discovered modules there; keep the discovery workflow above as the required starting point.

---

## C. Where new code goes — decision tree

```
Used by exactly one engine?
    → src/engines/<engine>/python/<module>.py
Used by 2+ engines?  (e.g. mocap loader, FK, cost terms)
    → src/shared/python/<topic>/
Standalone PyQt6 desktop tool?
    → src/tools/<tool_name>/  (PACKAGE, not a single file)
        - __init__.py, __main__.py, gui.py, core.py, README.md
        - register a tile in src/config/models.yaml
Engine-specific MATLAB code (Simscape model dynamics, helpers)?
    → src/engines/<engine>/matlab/
Inner-loop math kernel reused by Python AND WASM?
    → rust_core/<crate>/
        - PyO3 bindings via maturin
        - parity-tested against pure-Python fallback
Backend service / API?
    → src/api/{routes,services}/
```

When relocating something, leave a thin shim at the old path with a
`DeprecationWarning` and a one-line re-export so any external pinned
references keep working through one release cycle. Example: see the
shims at
`src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Motion Capture Plotter/starting_pose_*.py`
which redirect to the new `src/tools/starting_pose_matcher/` package.

---

## D. Tile-launcher contract

The main UpstreamDriftLauncher reads `src/config/models.yaml` and renders one
tile per entry. A tile entry needs:

```yaml
- id: "starting_pose_matcher"
  name: "Starting-Pose Matcher" # display name
  description: "..." # one-line tooltip
  type: "special_app" # other types: model, custom_humanoid, drake, ...
  path: "src/tools/starting_pose_matcher/__main__.py" # relative to REPO_ROOT
  launcher:
    category: "tool" # tool | physics_engine | document
    logo: "assets/<icon>.png" # optional; falls back to default tile art
    status: "ready" # ready | beta | broken
```

Tiles are launched as `subprocess.Popen` with the system Python. Any
tool listed here should be runnable as `python -m <package>` from the
repo root — the launcher resolves the path, prepends the repo root to
`sys.path`, and invokes the module directly.

If your tool has runtime dependencies beyond core (PyQt6, matplotlib,
pandas, openpyxl), declare them in a named `[project.optional-dependencies]`
extra in `pyproject.toml` and reference the install command in the
tile's `description`. Example: the `gui-tools` extra installs PyQt6 +
matplotlib + pandas + openpyxl for the desktop tools.

For a tool that should open inside the launcher tab/dock host, add all three
registration surfaces together:

1. Implement a lazy `_embed_adapter.py` (or equivalent module) that imports GUI
   dependencies inside `create_main_widget()` and calls
   `register_embeddable_tool()` at import time.
2. Add a `[project.entry-points."upstream_drift.embeddable_tools"]` entry whose
   value is the adapter module path. `src/launchers/embedded_tool_bootstrap.py`
   imports entry-point adapters first and keeps its source list only as an
   editable-install fallback.
3. Add the matching `src/config/models.yaml` manifest tile with
   `launcher.category: "tool"` so manifest coverage warnings stay clean.

---

## E. Tests

Tests go in `tests/<area>/<subarea>/test_*.py` — mirror the source
layout under `src/`. See `tests/README.md` and `tests/conftest.py` for
the full set of markers and fixtures.

### Markers (from `pyproject.toml`)

- `unit` — fast, deterministic, no engines.
- `integration` — exercises 2+ modules together.
- `slow` — > 5 s runtime; skipped from default CI lane.
- `live_simulation` — actually runs MuJoCo / Drake / Simscape; **always
  skipped** in default `pytest` runs. Opt in with `-m live_simulation`.
- `requires_gl`, `requires_mocap_fixtures` — skipped on CI fleet that
  lacks the resource.
- Engine-specific: `requires_mujoco`, `requires_drake`, `requires_pinocchio`,
  `requires_opensim`, `requires_matlab`.

### Optional GUI deps

PyQt6 / matplotlib are optional. Tests that need them MUST skip
cleanly when the import fails — `pytest`'s user-site Python on Windows
sometimes has a broken PyQt6 DLL search path even when the regular
interpreter loads it fine. Pattern:

```python
def _load_module():
    try:
        import PyQt6.QtCore  # noqa: F401
        import matplotlib    # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6/matplotlib not loadable: {exc}")
    ...
```

### Pure-data layer

For PyQt6 desktop tools, separate the pure-data math (`core.py`) from
the GUI (`gui.py`) so the data layer can be tested in any environment.
The matcher (`src/tools/starting_pose_matcher/`) is the canonical
example of this split.

---

## F. When to use Rust

The repo has exactly two Rust crates (as of this writing):

- `rust_core/upstream-physics/` — RK4 integrator, aerodynamics, contact,
  swing-plane. Inner-loop numerics; used identically by the Python
  backend (`src/shared/python/physics/rust_kernel.py`) and the WASM
  browser frontend.
- `ui/src-tauri/golf-modeling-suite` — Tauri desktop shell. Spawns the
  Python API server; **no physics or GUI logic in Rust**.

The criteria that produce a Rust crate in this project:

1. **Profiling shows a sub-millisecond inner loop** that gets called
   thousands of times per simulation, AND
2. The same kernel needs to run identically in Python AND another
   runtime (WASM, embedded C++, etc.).

Notably **NOT** Rust-shaped:

- GUI tools. PyQt6 is faster to iterate, has a far richer ecosystem
  for desktop scientific UIs, and binds the matplotlib stack we already
  use everywhere else.
- Data pipelines. pandas / numpy / scipy already cover everything we
  need at speeds that don't bottleneck.
- Config, orchestration, tests. These should stay in Python.

If you're tempted to rewrite something in Rust, profile first and
verify that the workload genuinely lives in inner loops. Most of the
time you'll find the bottleneck is matplotlib's 3D renderer or pandas'
xlsx parser, neither of which Rust can help with.

---

## G. Lessons learned (this section grows)

- **Wiffle xlsx is in CM, not inches.** The "Definitions" tab of the
  workbook is wrong; trust `load_club_target_excel.m` (line 53:
  `CM_TO_METRES = 0.01`).
- **The Simscape body chain has a `torso` joint** between spine and hub
  that hosts the revolute Z (twist) joint. Don't collapse to
  `hip → spine → hub` directly — you'll lose the visible torso coil.
  See
  `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/model/mdl_reference/GolfSwing3D_Kinetic.mdl`
  block "Torso Kinetically Driven" (SID 8331).
- **Don't subclass `BaseLauncher` for single-purpose tools.**
  `BaseLauncher` is for grid-of-tiles launcher windows. Standalone
  tools should be plain `QMainWindow` subclasses registered as tiles.
- **Test-environment pip vs. interpreter mismatch.** If pytest under
  `C:\Python314\python.exe -m pytest` fails to load PyQt6 with
  `DLL load failed`, it's because pytest is being resolved from
  `AppData\Roaming\Python\Python314` (user-site). Tests should skip
  gracefully — never assume PyQt6 imports.
- **The shared FK has known asymmetry** in hand height with the
  reference Address pose. When deriving fallback skeletons, use FK
  for "the body shape" (torso, shoulders) but be cautious about
  hand heights — they may need hand-tuning.
- **Touching ANY file under `src/shared/python/motion_matching/` may
  surface pre-existing mypy/bandit debt.** PR #4924 cleared the worst
  of it (engine_init_profiler md5/sha1 → usedforsecurity=False;
  precondition lambdas wrapped in bool(); excel loader pandas-cell
  type ignores). If a hook fails on a file you didn't author, follow
  the same sidecar pattern.
- **Two-tool live coupling.** When a tool publishes high-frequency
  state (e.g. Pose Studio's canonical pose at 30 Hz), use the
  WebSocket transport — file pub-sub adds 100+ ms of disk latency.
  The `realtime` facade picks automatically via the channel
  registry; don't hard-code. See
  [`docs/development/realtime_ipc.md`](docs/development/realtime_ipc.md).
- **Embed adapters keep PyQt6 imports lazy.** The
  `EmbeddableTool` Protocol module deliberately doesn't import
  PyQt6 (widget types are spelled `typing.Any`) so headless CI and
  the docs builder can introspect adapters without the GUI extras.
  Your `_embed_adapter.py` should follow the same pattern: import
  PyQt6 inside `create_main_widget`, not at module top.
- **`cleanup()` must be idempotent.** The host calls it on tab
  close, on parent shutdown, and on `closeEvent` — sometimes more
  than once during teardown. Drop the widget reference first, then
  tear down resources, and guard with `if widget is None: return`
  at the top.

---

## H. How to update this file

Found a shared module you wished you'd known about? Add it to section
B’s linked directory with a one-liner. Found a new trap? Add it to section G. Keep
the file focused — if it's growing past 400 lines, split sub-pages
into `docs/agents/`.

After adding to the shared infrastructure directory, also link the change in `CLAUDE.md`'s
"Shared Code Layout" section if one exists, and tag the file in your
PR description so reviewers can verify discoverability.

---

<!-- BEGIN FLEET-MANAGED: reasoning-engagement -->

## 🧠 Reasoning & Engagement

> This section is managed centrally by Repository_Management and synced fleet-wide.
> Do NOT edit it directly in individual repositories — edit the source in Repository_Management/AGENTS.md.

These rules govern _how_ you engage with a task before and during implementation. They exist because LLM agents tend to pick an interpretation silently, overcomplicate the solution, and edit code they were not asked to touch. Each rule directly counteracts one of those failure modes.

- **Surface ambiguity. Do not guess silently.** If the request has more than one plausible interpretation, list the options and ask before implementing. Picking one and running with it is the single most common cause of rework in this fleet.
- **Push back on overcomplication.** If a simpler approach would satisfy the request, say so before you build the complicated one. Do not implement bloated 1000-line constructions when 100 would do. The senior-engineer test: would they call this overcomplicated? If yes, simplify.
- **Stay surgical.** Every changed line must trace directly to the user's request. Do not "improve" adjacent code, comments, formatting, or imports. Do not refactor things that are not broken. Match existing style even if you would do it differently.
- **Spotted ≠ fix.** If you notice unrelated dead code, latent bugs, or stylistic problems while working, _mention them in the PR body or as a follow-up issue_ — do not fix them in the same PR. (The `mcp__ccd_session__spawn_task` tool is the right channel when working interactively.)
- **Clean up only your own orphans.** If your changes leave imports, variables, or functions newly unused, remove them. Do not delete pre-existing dead code unless the task asked for it.
- **State a verifiable success criterion before coding.** For a bug fix, that's a failing test that reproduces it (RED → GREEN, see TDD section below). For a feature, the explicit check that says "done." "Make it work" is not a success criterion.

**The diff test:** every line in your final diff should answer "this is here because the user asked for X." If you cannot answer that for a given line, remove it.

<!-- END FLEET-MANAGED: reasoning-engagement -->

---

<!-- BEGIN FLEET-MANAGED: network-api-hygiene -->

## 🛑 NETWORK & API HYGIENE (CRITICAL)

> This section is managed centrally by Repository_Management and synced fleet-wide.
> Do NOT edit it directly in individual repositories — edit the source in Repository_Management/AGENTS.md.

### GitHub API Quotas

| API Type                  | Quota        | Consumed By                                                        |
| ------------------------- | ------------ | ------------------------------------------------------------------ |
| REST (`gh api repos/...`) | 5,000 req/hr | Safe for polling                                                   |
| GraphQL                   | 5,000 req/hr | `gh pr list --json`, `gh pr checks`, `gh pr create`, `gh pr merge` |

GraphQL and REST have **separate** quotas. Exhausting GraphQL blocks PR creation and merging fleet-wide for an entire hour.

### Mandatory Rules

- **NO MASS POLLING**: Agents MUST NEVER use `gh pr list`, `gh issue list`, or arbitrary REST/GraphQL loops in a bulk manner to "scan" or "sweep" the repository fleet. Single, scoped repository lookups are allowed when needed (e.g., checking if a specific PR exists).
- **LOCAL FIRST**: Rely on local `.md` files, previously generated `issues.json` artifacts, or user assistance to find task context — do not query GitHub to discover what to work on.
- **NO PARALLELIZED GITHUB CLI**: Never write or execute scripts that loop over multiple repositories performing `gh` operations (automated PR merge scripts, fleet-wide status sweeps, etc.).
- **NO TIGHT POLLING LOOPS**: Never implement `while true; do gh pr checks $PR; sleep 30; done` patterns. Each iteration of such a loop costs 1–3 GraphQL calls; at 30-second intervals that drains the 5,000/hr quota in under 3 hours.
  - ❌ `while true; do gh pr checks; sleep 30; done`
  - ✅ `gh run watch <run-id>` — streams CI events without polling
  - ✅ Check status once at natural work breakpoints (after completing other tasks)
- **BATCHING**: If remote information is absolutely necessary, use a single focused query — not a loop of queries.
- **REST OVER GRAPHQL FOR CI STATUS**: Use REST endpoints for CI polling; they don't consume the GraphQL quota.
  - ❌ `gh pr checks <N>` (GraphQL)
  - ✅ `gh api repos/OWNER/REPO/actions/runs` (REST)
  - ✅ `gh api repos/OWNER/REPO/actions/jobs/<id>/logs` (REST)
- **STOP MONITORS IMMEDIATELY**: When using background monitor tasks, call `TaskStop <id>` the moment the monitored condition is satisfied. Do not leave monitors running "just in case."
- **LONG POLLING INTERVALS**: Background monitors must use ≥270-second intervals (keeps the prompt cache warm). Default to 1200–1800 s for idle monitoring. Never chain short sleeps to work around the 60-second minimum.
- **SILENT FAILURES**: If an API rate limit is hit, HALT NETWORK ACTIVITY IMMEDIATELY. Do not write retry-loops that further exhaust the quota. Alert the user and pivot to local work.

### Checking Rate Limit Status

```bash
gh api rate_limit | python3 -c "
import json, sys, datetime
d = json.load(sys.stdin)['resources']
for k in ['core', 'graphql']:
    r = d[k]
    reset = datetime.datetime.fromtimestamp(r['reset']).strftime('%H:%M:%S')
    print(f'{k}: {r["remaining"]}/{r["limit"]} remaining — resets {reset}')
"
```

<!-- END FLEET-MANAGED: network-api-hygiene -->

## Closing issues — non-negotiable rule

NEVER close a feature or bug issue without one of:

1. A merged PR that demonstrably implements the acceptance criteria (use `Closes #N` in the PR description), OR
2. An explicit `wontfix`, `roadmap`, `duplicate`, or `invalid` label.

The Verify-Issue-Closure workflow will automatically reopen any issue closed without evidence. Do not work around it.

When implementing an issue:

- Write or update tests FIRST (TDD: red → green → refactor)
- Add Design-by-Contract preconditions/postconditions where it clarifies invariants
- Respect Law of Demeter — don't reach through three layers of objects
- Don't duplicate code (DRY)
- Run the tests locally before pushing; don't rely on CI to find basic breakage
- If you can't fully implement, leave the issue open and post a status comment instead of closing

---

## Agent Handoff & PR Policy

Fleet-wide policy from `Repository_Management#1390`:

1. **Full PRs, never drafts.** Every PR must open ready-for-review — do not use
   `gh pr create --draft` or the GitHub UI draft toggle.
2. **Commit frequently.** Use small, conventional commits that save progress as you go;
   never batch a day's work into a single commit.
3. **Agent handoff document.** [`AGENT_HANDOFF.md`](AGENT_HANDOFF.md) at the repo root
   tracks current-state-only info (active epics/PRs, must-read architecture pointers,
   in-flight branch stacking, gate commands, a do-not list, and the short-term roadmap).
   Update it as part of **every PR you open** and **every push that lands on `main`**.
   It is not a changelog — history lives in git; keep it current-state only and under
   150 lines.

---

<!-- BEGIN FLEET-MANAGED: spec-changelog-rows -->

> This section is managed centrally by Repository_Management and synced fleet-wide.
> Do NOT edit it directly in individual repositories — edit the source in Repository_Management/AGENTS.md.

### Change-log rows are keyed by pull request

Binding fleet-wide from
[Repository_Management#1520](https://github.com/D-sorganization/Repository_Management/issues/1520)
(program [#1505](https://github.com/D-sorganization/Repository_Management/issues/1505)):

- A substantive pull request adds **exactly one** row to the SPEC.md change
  log: `| YYYY-MM-DD | #<your PR or issue> | one-line summary |`.
- **Never put a serial spec version in a row**, and **never bump the
  `Spec Version` field**. That field is release-derived — set by
  `scripts/bump_spec_version.py` when a release is cut.
- **Never renumber, reorder, or reword another contributor's row**, including
  while resolving a rebase. If a rebase conflicts inside the table, keep both
  rows; that is always the correct resolution.
- Register the merge driver once per clone so git resolves it for you:
  `python scripts/install_spec_merge_driver.py`.
- Verify locally with `python shared_scripts/fleet_hooks.py spec-changelog`.

Rationale: a serial version plus a header field that must match it are global
counters. Two concurrent pull requests necessarily choose the same next value
and necessarily edit the same two lines, so every second merge conflicted and
the only resolution was a mechanical renumber — twelve of them in one day
across four repositories. A pull request number cannot collide.

<!-- END FLEET-MANAGED: spec-changelog-rows -->

---

<!-- BEGIN FLEET-MANAGED: repo-context-codemap -->

## 🧭 Repo Context & Codemap Freshness

> This section is managed centrally by Repository_Management and synced fleet-wide.
> Do NOT edit it directly in individual repositories — edit the source in Repository_Management/AGENTS.md.

Use repo-local context before broad exploration:

- Read `AGENTS.md` first, then check `docs/codemap.md` or `docs/operations/codemap_freshness_runbook.md` when present.
- If `.codemap/` exists, treat it as a generated local cache for navigation; verify important claims against source files before editing.
- If `.codemap/` is missing or stale, use source search (`rg`), focused file reads, and tests as the fallback. Report the missing/stale index as a rollout gap instead of blocking unrelated work.
- Do not commit `.codemap/` or `.codemap/index.db`. Codemap indexes are cache/artifact data and must stay ignored.
- To audit local fleet posture, run `python -m scripts.codemap_context_inventory --root .. --format markdown` from `Repository_Management`. This is a local, network-free inventory; it is not a substitute for repo-specific validation.

<!-- END FLEET-MANAGED: repo-context-codemap -->

## Where to edit

- **Tools (chat/sidekick/shared)**: Edit in the upstream Tools repository. Do NOT edit in \endor/ud-tools\ or shadow it in \src/shared/python/\.
