# Cartographer Suggestion Log

Improvement suggestions from Fleet Cartographer passes on UpstreamDrift's
codebase maps. Logged here per
[`fleet-cartographer.md`](https://github.com/D-sorganization/Repository_Management/blob/main/docs/fleet-cartographer.md)
rather than filed as individual issues, unless a suggestion is promoted
(high priority, single small PR cannot cover it, and it needs cross-repo
coordination).

## Suggestion Log

### 2026-10-11 — UpstreamDrift — `docs/codemap/` Referenced a Retired In-Tree Copy

**Current state**: `docs/codemap/README.md` and `docs/codemap/agents.md`
described `src/shared/python/codemap/` as a byte-identical in-tree copy of
the Tools canonical implementation, and gave a Codex/MCP fallback example
(`PYTHONPATH=src`, `python -m shared.python.codemap export --jsonl`) that
assumed that path existed. The directory was retired by
[#9406](https://github.com/D-sorganization/UpstreamDrift/issues/9406); the
CLI/watcher/MCP server now resolve only from the pinned Tools tree
(`vendor/ud-tools/`), as `CLAUDE.md` and `docs/agent_context/USAGE.md`
already correctly state.
**Suggested improvement**: Corrected both files in this pass to point at
`vendor/ud-tools/src` instead of the retired in-tree path.
**Rationale**: An agent following the stale fallback instructions would
`PYTHONPATH=src python -m shared.python.codemap...` against a module that
no longer exists, land on a misleading `ModuleNotFoundError`, and lose time
diagnosing a problem the map itself caused.
**Priority**: Medium
**Status**: Implemented (this pass)

### 2026-10-11 — UpstreamDrift — `codemap_tools.py` API-Resolution Fallback Looks Dead

**Current state**: `src/shared/python/ai/tools/codemap_tools.py` resolves
the codemap API via `_try_tools_api()` first and `_try_local_api()` as a
documented "in-tree shadow copy" fallback. Both functions execute the
identical `from shared.python.codemap import api as ...` import (there is
no second import path left after #9406 retired the in-tree copy), so
`_try_local_api()` can never return a different result than
`_try_tools_api()` already would have. The module docstring still describes
a two-path "canonical vs. in-tree shadow" resolution order that no longer
exists in the import graph.
**Suggested improvement**: Either delete `_try_local_api()` and the
`flavor="local"` branch, or point it at a genuinely distinct fallback
(e.g. a direct `vendor/ud-tools/src` path insertion) if a second resolution
path is still wanted. Update the module docstring to match whichever shape
is kept.
**Rationale**: Dead fallback branches that still read as live design in the
docstring mislead the next agent who touches this file into preserving
behavior that cannot occur in practice.
**Priority**: Low — not a behavior bug (the primary path still resolves
correctly), but worth a small follow-up PR by whoever next touches this
file rather than a cartographer-authored behavior change.
**Status**: Logged

### 2026-10-11 — UpstreamDrift — `PROJECT_MAP.md` Tile Count Is Stale Background, Not Wrong per Se

**Current state**: `docs/architecture/PROJECT_MAP.md` §1 lists "11 tiles
currently visible in the launcher" (dated 2026-06-10), while
`docs/agent_context/README.md` reports the generated launcher manifest at
58 records. The document already carries a banner disclaiming itself as
"historical... retained as background" in favor of
`docs/architecture/CAPABILITY_ATLAS.md`.
**Suggested improvement**: None actioned this pass — the existing banner
already redirects readers to the generated, current atlas. Left as a
backlog item in case a future pass wants to either regenerate the historical
inventory or trim §1 down to a pointer-only stub.
**Priority**: Low
**Status**: Logged
