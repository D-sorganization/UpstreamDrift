# Consolidation Hardening Audit (2026-09)

Program: D-sorganization/Repository_Management#1505 · Phase 1 · Pillar P1
Issue: #8776 (re-parented under the seam epic #9406)

## Scope

#8776 reports that three large merges silently removed protective code and
protective configuration, and that the gates which should have caught it were
themselves disabled:

| Merge               | Subject                                                                            | Files / lines                  |
| ------------------- | ---------------------------------------------------------------------------------- | ------------------------------ |
| `3e09be404`         | feat(sidekick): overhaul assistant settings                                        | 158 files, +21,529 / -2,733    |
| `0575fb4b8` (#8322) | chore: consolidate overlapping UpstreamDrift PR backlog                            | 1,248 files, +40,796 / -24,827 |
| #8746               | Bolt: memoize recursive ModelTree node (bundled a `--write-baseline` regeneration) | DRY baseline                   |

This audit re-checks every item in the issue against `origin/main` at
`bb067bb5f` (2026-09-03) **and** against the pinned Tools tree
(`vendor/ud-tools` at `c0a395d5`), because every file the deletions touched is
a Tools-owned child copy under `src/shared/python/ai/` — the seam rulings in
`docs/shared_tools/seam_rulings.v1.json` rule `ai/adapters` `tools-canonical`.

## Findings

### 1. Provider Formatters: `if current_message.strip():` Guard (From `3e09be404`)

| Adapter   | UD copy `src/shared/python/ai/adapters/` | Tools copy `vendor/ud-tools/.../ai/adapters/`                                                              |
| --------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| anthropic | **absent**                               | present (1 site)                                                                                           |
| openai    | **absent**                               | present (1 site)                                                                                           |
| ollama    | **absent**                               | present (1 site)                                                                                           |
| gemini    | **absent**                               | present as `if not effective_message.strip() and msg_list:` plus DbC preconditions `bool(message.strip())` |

Status: **restored upstream, not in UD's copy.** Tools carries the guard on
every provider (Tools last-touch 2026-08-18/25); UD's copies are the
2026-08-01 #8322 snapshot and still send the empty trailing user turn.
UpstreamDrift runs the _Tools_ copy under the launcher and under pytest only
when `shared.python.ai` resolves to vendor — which, as the seam inventory
showed, it does not (`tests/conftest.py` puts `src/` first). The live bug
therefore persists in UD desktop sessions until `ai/adapters` is deleted from
`src/shared/python` (seam PR-3 follow-up, ruling `tools-canonical`).

### 2. Ollama Typed Error Ladder (From `3e09be404`)

`isinstance(e, httpx.ConnectError)` / `httpx.TimeoutException` ladder and the
"Is Ollama running? Start with: ollama serve" hint: **present** in both the UD
copy (`ollama_adapter.py:468`, `:650`) and the Tools copy. Restored by #8775
on 2026-08-20. Closed.

### 3. BitNet `_MAX_PROMPT_BYTES` / `_build_validated_prompt()` (From #8322)

| Copy                                                 | State                                                                |
| ---------------------------------------------------- | -------------------------------------------------------------------- |
| UD `src/shared/python/ai/adapters/bitnet_adapter.py` | **absent** (no prompt size ceiling, no UTF-8 validation before argv) |
| Tools `vendor/ud-tools/.../bitnet_adapter.py`        | present (Tools 2026-08-25, 2,079 bytes larger than UD's)             |

Status: same as finding 1 — fixed in Tools, still missing in the UD shadow.
Resolution is deletion of the shadow, not a third copy of the fix.

### 4. `testpaths` Lost `src/shared/python/tests` (From #8322)

`pyproject.toml` `[tool.pytest.ini_options] testpaths` contains neither
`src/shared/python/tests` nor the commented-out
`vendor/ud-tools/src/shared/python/tests`. On `main` UD's
`src/shared/python/tests/` now holds only `__init__.py` — the eight test
files the issue mentions were removed on 2026-08-18 (`55f6ccda1`). The Tools
copy of that package (`test_contracts_shared.py`, `test_cors.py`,
`test_god_class_guard.py`, `ui/`) is exercised by Tools CI, and by the
`shared-tools-consumer-contracts` lane here. Status: **moot for UD**; the
ruling for `tests` is `tools-canonical`. No action.

### 5. Test Artifacts Committed to the Repo Root (From #8322)

`base.csv`, `pytest_report*.txt`, `patch_trace2.py` and friends: **gone** from
`main` (`ls` returns nothing; Phase 0 PR #9427 also swept `.scratch/` and
`output/`). `scripts/check_root_clutter.py` runs in `repo-structure-gates`.
Closed.

### 6. DRY Baseline Clobber (From #8746)

Restored on `main` before this audit (issue text); verified here:
`python scripts/ci/check_dry_duplication_gate.py` exits 0 on the PR-3 tree
with every recorded fingerprint at or below baseline. Closed.

### 7. Security Headers, Input Validation, CORS, Auth (Requested By #9406)

`git show 0575fb4b8 -- src/api src/shared/python/cors.py src/shared/python/security`
removes 863 lines, 838 of which are `src/api/task_manager_durable.py` (a
durable task manager that no route imported; its `max_retries`
plumbing is the only "limit" vocabulary in the diff). In
`src/shared/python/cors.py` the fail-closed check survives — the message
changed from `CORS_ORIGINS must not contain '*' when credentials are enabled
(fail-closed)` to a longer explanation and a per-origin well-formedness check
was **added**. `src/api/server.py` still validates `*`+credentials at startup
(`server.py:128`) and calls `_assert_production_secrets()`. No security header,
auth dependency or input validator was removed by #8322 in `src/api`.

`3e09be404` removed only adapter-level `raise AI*Error` sites that Tools has
since restored (findings 1–2); it did not touch `src/api`.

## Summary

| Item                               | Dropped by | Present in Tools pin `c0a395d5` | Present in UD shadow     | Action                                        |
| ---------------------------------- | ---------- | ------------------------------- | ------------------------ | --------------------------------------------- |
| strip guards (4 providers)         | 3e09be404  | yes                             | **no**                   | delete UD `ai/adapters` (seam PR-3 follow-up) |
| Ollama error ladder                | 3e09be404  | yes                             | yes (#8775)              | none                                          |
| BitNet prompt ceiling              | #8322      | yes                             | **no**                   | delete UD `ai/adapters`                       |
| `src/shared/python/tests` testpath | #8322      | n/a (Tools CI)                  | files removed 2026-08-18 | none                                          |
| root test artifacts                | #8322      | n/a                             | removed                  | none                                          |
| DRY baseline                       | #8746      | n/a                             | restored                 | none                                          |
| API security headers / CORS / auth | —          | —                               | unchanged                | none                                          |

The two remaining gaps are in a package this repository must not edit
(`src/shared/python/ai` is a Tools child copy; `error_handling_baseline.json`
says as much). The mechanical guard #8776 asks for — a ratchet baseline that
shrinks inside an unrelated change must fail review — is provided by
`scripts/ci/check_dry_duplication_gate.py` refusing fingerprints that
disappear without `--write-baseline`, plus the new `seam-drift-gate`
(`scripts/shared_tools/check_seam_drift.py`), which fails when a package
ruled `tools-canonical` grows a UD copy again.

## Regression Guard Added by This PR

`tests/unit/shared_python/ai/test_vendor_adapter_hardening.py` asserts that the
pinned Tools adapters carry the strip guard on every provider and the BitNet
prompt ceiling, so a vendor bump that loses them fails CI here even before
the UD shadow is gone.
