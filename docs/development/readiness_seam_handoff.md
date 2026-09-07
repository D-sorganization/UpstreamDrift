# Readiness Program: Seam Retirement and Failure Triage

Updated: 2026-09-06. Program is
[Repository_Management#1505](https://github.com/D-sorganization/Repository_Management/issues/1505).
This document covers UpstreamDrift #9406 (retire the `src/shared/python` shadow
tree) and #9474 (triage the `main` failure list). Live work is PR #9569.

Current state only. History is in git and on the issues.

## Seam State

Source of truth is `docs/shared_tools/seam_rulings.v1.json`. **16 of 36**
actionable rulings are `cleaned`, 20 `pending-cleanup`. Inventory totals:
identical 476, diverged 262, ud-only 1183, tools-only 518. `check_seam_drift.py`
passes with 17 notes, down from 32.

Retirement works because `src/__init__.py` registers the pinned Tools tree as an
import **fallback** — a meta-path finder appended to `sys.meta_path`, so a child
copy that still exists is still what resolves, and only a deleted one falls
through. Extending `src.shared.python.__path__` is _not_ sufficient: that
package's own `__init__` imports submodules while it executes, before any code
could extend the finished module's path.

Retired: `deprecation.py`, `README.md`, `chat_contracts`, `file_watcher`,
`logging_pkg`, `safe_eval.py`, `safe_pandas_eval.py`, `scripting`, `tests`,
`cors.py`, `rotation_transforms`, `upstream_drift_tools`, `compatibility.py`,
`codemap`, `programmatic_pid`, `plot_engine`.

## Three Traps This Work Has Already Hit

**1. The `diverged` count is roughly twice the real figure.** Every inventory
entry carries a `spelling_only` flag _alongside_ its classification, and the two
disagree: 153 of the 292 files classified `diverged` differ only by the
`src.shared.python` import prefix. Only **139 differ in content**. Filter on
`spelling_only` before sizing any of this work.

**2. A large negative byte delta is not evidence of staleness.** In `codemap` it
meant UpstreamDrift was behind. In `chat` the same signal meant UpstreamDrift had
_decomposed_ the widget into `_qt/runtime.py` (#8553) while canonical is still
the pre-refactor monolith — `initialize_streaming_state` appears zero times
upstream. Retiring `chat` would delete that refactor while the drift gate, the
divergence count and the ruling status all reported progress. `chat` needs
re-ruling to `ud-canonical`. So do `notes` and `plot_theme`, which have zero real
divergence but one UpstreamDrift-only file each — the schema's definition of
`split`, not `tools-canonical`.

**3. `import_aliases.py` cannot be retired yet.** Adopting canonical widens
`_external_src_package_is_available()` from an exact path equality to
`is_relative_to(repo_root)` and adds `contracts` to
`_DOWNSTREAM_SRC_ALIAS_ROOTS`. Together those make `src.shared.python.ai.*`
resolve upstream even though `ai` is still UpstreamDrift-owned, which broke
`test_gemini_adapter_capabilities` — passing on `main` and not in the quarantine
ledger, so a real regression. Checking `plot_theme` / `plot_engine` resolution is
**not** sufficient evidence: the predicate affects every `src.`-spelled shared
import, not a named list. Retire `ai` and `contracts` first.

## Failure Triage (#9474)

`main` reaches `short test summary info` since #9490 (the `get_cv2` abort) and
#9527 (plugin autoload in isolated child runners). The measured verdict:

```
760 failed, 38632 passed, 1359 skipped, 96 errors in 7591.46s (2:06:31)
```

The **305** figure quoted in earlier comments came from runs that aborted before
printing a summary. It is not a count and should not be used for sizing. The lane
runs ~126 minutes, so `timeout-minutes: 150` is correct — an earlier
recommendation to cut it to 105–110 was based on truncated runs and is withdrawn.

Fixed on #9569: `model_generation` preconditions (53), a `success` →
`solver_status` rename mis-applied to `BuildResult` by #5845 (14),
`_get_writable_model` and `_HUMANOID_PRESETS` (16).

Open and needing an owner decision: **#9584** — 38 mesh-generator failures test
`MakeHumanMeshGenerator` / `SMPLXMeshGenerator` implementations that are imported
by **zero** source files. The live pair lacks the tested API; the dead pair lacks
a path-traversal guard (`_validate_output_path_within_base`). Every available
move loses something.

## Guard Note

`tests/unit/repo_hygiene/test_tools_child_copy_contract.py` consults
`seam_rulings.v1.json` and exempts `ud-canonical` clusters. Without that it
reported fixes to UpstreamDrift's _own_ `model_generation` and
`humanoid_character_builder` as forbidden child-copy edits — making the two
largest failure clusters unfixable from either side, while their failing tests
sat in UpstreamDrift's own suite. It also permits an edit that makes a child copy
byte-identical to canonical (convergence), and still forbids anything else.

## Commands

```bash
python -m scripts.shared_tools.divergence_inventory --write
python scripts/shared_tools/check_seam_drift.py
python -m scripts.shared_tools.check_tools_pins
python -m pytest tests/unit/repo_hygiene/test_vendored_tools_fallback.py -q
python -m pytest tests/unit/repo_hygiene/test_tools_child_copy_contract.py -q
```

Retiring a cluster: confirm every tracked file is byte-identical to the pinned
tree (`git ls-files` — a bare `cmp` sweep counts `__pycache__` and lies), delete,
set the ruling to `cleaned`, regenerate the inventory, and confirm the drift gate
still passes. Never mark `cleaned` without deleting: that switches the gate from
reporting the path to enforcing it.
