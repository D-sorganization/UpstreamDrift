# CLAUDE.md — UpstreamDrift

> **GAAI Fleet Member.** GAAI framework installed in `.gaai/`. Read `.gaai/core/GAAI.md` for full governance spec.
> Rules: `@.gaai/core/contexts/rules/base.rules.md` and `@.gaai/project/contexts/rules/project.rules.md`
> All work on `staging` branch. PRs target `staging`. Never push directly to `main`.

## What This Is

Golf ball flight and physics modeling suite. Simulates aerodynamics, ball-club impact,
and trajectory using multiple physics engines (MuJoCo, Drake, Pinocchio, OpenSim).
Optional Rust extensions built via Maturin for performance-critical paths.

## Key Directories

- `src/` — core library: physics wrappers, URDF loaders, simulation runners
- `tests/` — pytest suite (unit, integration, live simulation)
- `scripts/` — CI helpers including `check_file_size_budget.py`
- `scripts/config/file_size_budget.json` — per-file size exceptions
- `module_size_budget_baseline.json` — modules exceeding default size limits
- `rust/` — optional Rust features built with Maturin

## Python and Tooling

- **Python 3.10+**. Always `python3`, never `python`.
- **Formatter:** Ruff format (NOT Black). 88-char line limit.
- **Linter:** Ruff check. These are **separate CI steps** — both must pass independently.

## Development Commands

```bash
python3 -m ruff check .                          # lint
python3 -m ruff format --check .                  # format check
python3 -m ruff format .                          # auto-format
python3 -m pytest -n auto --timeout=60            # full test suite
python3 -m pytest -m unit -n auto --timeout=60    # unit tests only
python3 -m pytest -m "not slow and not live_simulation" -n auto --timeout=60
python3 scripts/check_file_size_budget.py         # file size check
maturin develop                                   # build Rust extensions locally
```

## CI Requirements (All Must Pass)

1. `ruff check` — zero violations
2. `ruff format --check` — zero diffs (separate step from lint)
3. File size budget: **1200 lines max** per file. Exceptions in `scripts/config/file_size_budget.json`
4. Module size budget: checked against `module_size_budget_baseline.json`
5. No TRACKED_TASK/TRACKED_DEFECT unless tied to a tracked GitHub issue
6. pytest with `-n auto`, 60s timeout, **10% coverage minimum**
7. No `print()` in `src/` — use logging

## Test Markers

`unit`, `integration`, `slow`, `live_simulation`, `requires_gl`, `headless_safe`,
`benchmark`, `scientific`

## Physics Engine Gotchas

- **Pinocchio:** NO `computeTotalEnergy`. Use `computeKineticEnergy` + `computePotentialEnergy` separately.
- **Drake:** Must use explicit imports: `from pydrake.X import Y`. Attribute access on `pydrake` namespace does not work. Use `body.body_frame()` directly, NOT `FixedOffsetFrame`.
- **Test pollution:** Never `sys.modules["pydrake"] = MagicMock()` at module level. Use `patch.dict("sys.modules", ...)` which auto-cleans after the test.

## Known Constraints

- **Branch naming:** `fix/issue-XXXX-description`
- **Remote:** origin URL references `Golf_Modeling_Suite.git`
- Rust builds: `maturin develop` for local dev; CI handles wheel builds

## Coding Standards (Enforced by CI and QA)

- **DRY:** No duplicated logic blocks >5 lines. CI tracks DRY adoption metrics.
- **DbC:** Public functions validate preconditions, raise `ValueError`/`TypeError` with descriptive messages. Document postconditions in docstrings.
- **LOD:** No method chains >2 levels (`a.b.c.d()` violates). Add delegating methods instead.
- **TDD:** Tests in same PR as implementation. Coverage must not decrease.
- **File size:** If approaching 1200 lines, refactor before adding more.

## Cross-Repo Dependencies

- **Imports from Tools** (D-sorganization/Tools): URDF generation, signal processing, shared utilities.
- Breaking changes to Tools public API require a coordinated PR here.
- Gasification_Model also depends on Tools — avoid transitive breakage.

## Slash Commands

- `/gaai-deliver` — Run Delivery Loop for next ready backlog item
- `/gaai-status` — Show current backlog and memory state
