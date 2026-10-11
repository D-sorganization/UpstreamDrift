# Cartographer Report — UpstreamDrift — 2026-10-11

**Role**: The Silent Cartographer (scheduled pass)
**Scope**: UpstreamDrift only (single-repo worktree; no sibling repos
available for the fleet-wide inventory script in this run)

## What Was Checked

1. **`docs/agent_context/` catalog freshness.** Recomputed SHA-256 for every
   file listed in `catalog.json` (`sources`, `documentation`, `tests`) and
   compared against the committed hashes in `reviews.json`.
   - 56 catalog source paths: all exist.
   - 73 documentation/test paths: all exist.
   - 66 review input hashes across all 6 registered relations
     (`capture-atlas`, `launcher-atlas`, `model-backends`, `pipeline-api`,
     `pose-conventions`, `project-task-native-replay`): all match the
     current working tree byte-for-byte. No stale review found.
2. **`docs/architecture/C4.md` contract.** `python scripts/architecture_map_contract.py`
   → PASS (1 context, 1 container, 8 features, 2 changelog rows).
3. **`docs/codemap/` accuracy.** Cross-checked against the retirement of
   the in-tree codemap copy (#9406, documented in `CLAUDE.md`). Found both
   `docs/codemap/README.md` and `docs/codemap/agents.md` still described
   `src/shared/python/codemap/` as a live byte-identical copy and gave a
   fallback invocation (`PYTHONPATH=src`) against a module path that no
   longer exists in `src/`. Fixed in this pass (see diff).
4. **`docs/architecture/PROJECT_MAP.md` / `CAPABILITY_ATLAS.md`.** Confirmed
   `PROJECT_MAP.md` already self-identifies as a stale historical
   background doc pointing readers at the generated `CAPABILITY_ATLAS.md`;
   logged the residual tile-count mismatch as a low-priority backlog item
   rather than editing a document that already disclaims itself.
5. **`vendor/ud-tools` submodule.** Not initialized in this worktree
   (consistent with the known pre-existing gap tracked elsewhere); the
   `agent_context` and `codemap` CLI/MCP commands documented in
   `docs/agent_context/USAGE.md` could not be exercised live here. Static
   checks (steps 1–4) do not require the submodule and were unaffected.

## Findings and Actions

| Finding | Action |
| --- | --- |
| `docs/codemap/README.md`, `docs/codemap/agents.md` referenced the retired in-tree codemap copy and a dead `PYTHONPATH=src` fallback | Fixed in this PR — both now point at `vendor/ud-tools/src` and link #9406 |
| `codemap_tools.py`'s `_try_local_api()` fallback imports the same module path as `_try_tools_api()` and can never diverge from it; its docstring still describes a live two-path resolution | Logged, not fixed (behavior change, out of cartographer lane) — see suggestion log |
| `PROJECT_MAP.md` §1 tile count (11) vs. the generated manifest (58 records) | Logged only — the doc already redirects to the current atlas |

Full detail: [`cartographer-suggestions.md`](../cartographer-suggestions.md).

## Not Done This Pass

- Fleet-wide inventory (`scripts/codemap_context_inventory.py`) — requires
  sibling repo checkouts not present in this worktree.
- Any live exercise of `agent_context`/`codemap` CLI/MCP commands — blocked
  on `vendor/ud-tools` submodule init, which timed out when attempted here.
- No behavior changes to `src/` — the one source-adjacent finding
  (`codemap_tools.py`) is logged for the owning area to pick up rather than
  fixed inline, per the cartographer mandate to maintain maps, not implement
  product fixes.
