# CLAUDE.md — UpstreamDrift

`CLAUDE.md` is the authoritative contributor and agent policy file.

> **GAAI Fleet Member.** GAAI framework installed in `.gaai/`. Read `.gaai/core/GAAI.md` for full governance spec.
> Rules: `@.gaai/core/contexts/rules/base.rules.md` and `@.gaai/project/contexts/rules/project.rules.md`
> PRs target `main`. Use focused topic branches such as `fix/...`, `feat/...`, `chore/...`, or `claude/...`.
>
> **Before writing new code, read [`AGENTS.md`](AGENTS.md)** — it lists the
> shared infrastructure (FK, reference poses, mocap loaders, theme,
> rendering helpers) you should reuse instead of reinventing.

## ⚠️ Multi-agent coordination — read before opening any PR

This repo is part of the D-sorganization fleet. Multiple agents
(`claude`, `codex`, `jules`, `local`, `gaai`, `maxwell-daemon`) and the
repo owner all touch this codebase. **Coordination is mandatory** to
avoid the kind of duplicate-work collisions that have happened in the
past.

The full protocol lives in
[`Repository_Management/docs/agent-lease-protocol.md`](https://github.com/D-sorganization/Repository_Management/blob/main/docs/agent-lease-protocol.md)
and
[`Repository_Management/docs/agent-coordination-strategy.md`](https://github.com/D-sorganization/Repository_Management/blob/main/docs/agent-coordination-strategy.md).
Short version:

1. **Before starting work on an issue**, run from a clone of `Repository_Management`:

   ```bash
   python -m scripts.check_agent_claim --repo UpstreamDrift --issue <N>
   ```

   If the result is `{"held": true, ...}` and the agent is not you, **pick a different issue.**

2. **If the issue is free**, post a lease before creating a branch:

   ```bash
   python -m scripts.post_agent_lease \
       --agent claude \
       --session <session-id> \
       --repo UpstreamDrift \
       --issue <N>
   ```

   This adds a `claim:claude` label and posts a `<!-- agent-lease v1 -->`
   comment. Default TTL is 2 h; an open `Fixes #N` PR implicitly extends
   it to 24 h.

3. **Do not modify or delete lease comments left by other agents.**

4. **Priority order** (see `Repository_Management/shared_scripts/agent_identity.py`):
   `user > maxwell-daemon > claude > codex > jules > local > gaai`. The
   `Jules-Redundant-PR-Closer` workflow in this repo enforces this — when
   two agents file PRs against the same issue, the lower-priority PR is
   auto-closed with a deferral comment.

5. **The `Agent-Lease-Reaper` runs every 30 minutes from `Repository_Management`** and sweeps stale claims fleet-wide. Manual release via a `<!-- agent-lease v1 release -->` comment is welcome but not required.

6. **Fail-open** — if the claim/lease scripts error, proceed but accept the risk of duplication. Coordination is best-effort, not a hard gate.

## What This Is

A unified platform for golf swing analysis across multiple physics engines and
biomechanical modeling approaches.
Optional Rust extensions built via Maturin for performance-critical paths.

## Key Directories

- `src/` — core library: physics wrappers, URDF loaders, simulation runner
- `src/shared/python/pose_interchange/` — engine-agnostic canonical pose + per-engine adapters / services
- `src/shared/python/launcher_embed/` — embeddable-tool contract + registry (see [ADR-0013](docs/adr/0013-launcher-composability.md))
- `src/shared/python/realtime/` — file + WebSocket pub-sub IPC
- `src/launchers/embedded_host.py` — in-launcher tool host (tabs + docks)
- `src/tools/pose_studio/` — interactive cross-engine pose editor (launcher tile: `pose_studio`)
- `tests/` — pytest suite (unit, integration, live simulation)
- `scripts/` — CI helpers including `check_file_size_budget.py`
- `scripts/config/file_size_budget.json` — per-file size exceptions
- `scripts/config/module_size_budget_baseline.json` — modules exceeding default size limits
- `rust_core/` — optional Rust features built with Maturin

## Motion Pipeline

- **User Guide**: [`docs/motion_pipeline/README.md`](docs/motion_pipeline/README.md) — From video to tracked motion in 5 commands
- **Format Matrix**: [`docs/motion_pipeline/formats.md`](docs/motion_pipeline/formats.md) — Supported mocap formats and quirks
- **Troubleshooting**: [`docs/motion_pipeline/troubleshooting.md`](docs/motion_pipeline/troubleshooting.md) — Common failure modes and fixes
- **Architecture**: [`docs/adr/0007-motion-pipeline-architecture.md`](docs/adr/0007-motion-pipeline-architecture.md) — CIR design and module boundaries

## Python and Tooling

- **Python 3.10+** is the supported minimum from `pyproject.toml`.
- **Python 3.11** is the default CI interpreter in `.github/workflows/ci-standard.yml`.
- Always `python3`, never `python`.

- **Formatter:** Ruff format. 88-char line limit.
- **Linter:** Ruff check. These are **separate CI steps** — both must pass independently.

## Development Commands

```bash
python3 -m ruff check .                          # lint
python3 -m ruff format --check .                  # format check
python3 -m ruff format .                          # auto-format
python3 -m pytest -n auto --timeout=60            # full test suite
python3 -m pytest -m unit -n auto --timeout=60    # unit tests only
python3 -m pytest -m "not slow and not live_simulation" -n auto --timeout=60
python3 scripts/ci/check_file_size_budget.py      # file size check
maturin develop                                   # build Rust extensions locally
```

## CI Requirements (All Must Pass)

1. `ruff check` — zero violations
2. `ruff format --check` — zero diffs (separate step from lint)
3. File size budget: **1200 lines max** per file. Exceptions in `scripts/config/file_size_budget.json`
4. Module size budget: checked against `module_size_budget_baseline.json`

5. No TODO/FIXME unless tied to a tracked GitHub issue
6. pytest with `-n auto`, 60s timeout, and the coverage threshold defined by `fail_under` in `pyproject.toml [tool.coverage.report]`
7. No `print()` in `src/` — use logging

## Test Markers

`unit`, `integration`, `slow`, `live_simulation`, `requires_gl`, `headless_safe`,
`benchmark`, `scientific`

## Physics Engine Gotchas

- **Pinocchio:** NO `computeTotalEnergy`. Use `computeKineticEnergy` + `computePotentialEnergy` separately.
- **Drake:** Must use explicit imports: `from pydrake.X import Y`. Attribute access on `pydrake` namespace does not work. Use `body.body_frame()` directly, NOT `FixedOffsetFrame`.
- **Test pollution:** Never `sys.modules["pydrake"] = MagicMock()` at module level. Use `patch.dict("sys.modules", ...)` which auto-cleans after the test.

## Known Constraints

- **Branch naming:** use focused topic branches such as `fix/...`, `feat/...`, `chore/...`, or `claude/...`
- **Remote:** `D-sorganization/UpstreamDrift`
- Rust builds: `maturin develop` for local dev; CI handles wheel builds

## Coding Standards (Enforced by CI and QA)

- **DRY:** No duplicated logic blocks >5 lines.
- **DbC:** Public functions validate preconditions, raise `ValueError`/`TypeError` with descriptive messages. Document postconditions in docstrings.
- **LOD:** No method chains >2 levels (`a.b.c.d()` violates). Add delegating methods instead.
- **TDD:** Tests in same PR as implementation. Coverage must not decrease.
- **File size:** If approaching 1200 lines, refactor before adding more.

## Error handling (issue #5911 / ADR-0016)

Three anti-patterns are blocked by `scripts/ci/check_error_handling_ratchet.py` from growing beyond the baseline in `scripts/config/error_handling_baseline.json`. Pre-existing instances are grandfathered with `# noqa: <code>`; **new code must use the helpers**.

| Don't                                              | Do                                                                                                    | Helper                              |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ----------------------------------- |
| `try: ... except Exception: pass`                  | `with narrow_catch(ValueError, OSError, log_message="op"): ...`                                       | `core.process_safety.narrow_catch`  |
| `subprocess.Popen(cmd, ...)` for short-lived spawn | `with managed_popen(cmd, timeout=T) as proc: ...`                                                     | `core.process_safety.managed_popen` |
| `await asyncio.gather(*tasks)`                     | `await safe_gather(*tasks)` (or `raise_on_all_failed=True`)                                           | `core.process_safety.safe_gather`   |
| `raise RuntimeError("X is closed")`                | `raise StateError(...)` from `core.contracts.exceptions` or a domain subclass from `core.error_utils` | existing hierarchy                  |
| `logger.error("...: %s", e)` in `except`           | `logger.exception("...")`                                                                             | stdlib (preserves traceback)        |
| `for line in open(path):`                          | `with open(path) as f: for line in f:`                                                                | stdlib (no helper needed)           |

Lint rules enforced (no longer in `extend-ignore`):

- `BLE001` — blind `except Exception`
- `F841` — unused local variable
- `F401` — unused import (use `__all__` or redundant-alias `import X as X`)

If you genuinely need to break one of these rules, add `# noqa: <CODE> - <reason>` and explain in the PR description. The ratchet allows the count to stay equal, so swap one in for one out.

## Cross-Repo Dependencies

- **Tools integration surface:** shared Python utilities are vendored in `vendor/ud-tools/`, and optional editable sibling wiring lives behind `scripts/setup_tools_workspace.sh` plus the pytest `--tools-mode` fixtures in `tests/conftest.py`.
- Breaking changes to Tools public API require a coordinated PR here.
- Gasification_Model also depends on Tools — avoid transitive breakage.

## Where to edit shared code

Tools is the source of truth for the shared utilities vendored here as
`vendor/ud-tools/`. Within this repo those modules live at
`src/shared/python/chat/`, `src/shared/python/ai/`, and the sidekick package
(canonical name as of Stage 2, #5619) is provided by `vendor/ud-tools/`.
**Never edit these inside `vendor/ud-tools/`**; vendor changes are erased on the
next `git submodule update` or vendor bump.

### Package naming — sidekick is canonical

`sidekick` is the canonical package name for the shared tools utility library.
`upstream_drift_tools` is a **deprecated alias** (compat shim provided by Tools
PR #2885 / Stage 1). New code in `src/` and `tests/` must import from `sidekick`:

```python
# Correct (Stage 2+)
from sidekick.theme import CatppuccinTheme
import sidekick

# Deprecated — do not use in new code
from upstream_drift_tools.theme import CatppuccinTheme  # noqa: removed in Stage 2
```

The compat shim in `vendor/ud-tools` keeps `upstream_drift_tools` importable
during the transition period, but the hygiene test
`tests/unit/repo_hygiene/test_no_deprecated_imports.py` enforces that no
`src/` or `tests/` file uses the old name.

Repository-hygiene tests at `tests/unit/repo_hygiene/` enforce this:

- `test_no_shadow_of_tools_shared.py` — fails if a UD module shadows a Tools shared module without an allow-list entry
- `test_vendor_submodule_clean.py` — fails if the vendor submodule has uncommitted edits in its working tree
- `test_no_deprecated_imports.py` — fails if any `src/` or `tests/` file imports `upstream_drift_tools` (Stage 2+)

See issue #5623, #5619.

## Slash Commands

- `/gaai-deliver` — Run Delivery Loop for next ready backlog item
- `/gaai-status` — Show current backlog and memory state

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
