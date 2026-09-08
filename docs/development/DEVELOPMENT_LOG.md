# Development Log — UpstreamDrift

State table for every feature in flight in this repository. Update entries
**in place**; never append dated sections. One entry per feature, from proposal
to ship. See the `development-logs` section of `AGENTS.md` for the binding rules
and `shared_scripts/development_log.py` for the validator.

- **Portfolio:** golf
- **WIP limit:** 8
- **Last audited:** 2026-09-08 by claude

## States

`proposed` → `in_progress` → `in_review` → `shipped`, with `parked` reachable
from any live state and `abandoned` from `parked`. `shipped` never returns to
`in_progress`; open a new entry instead.

## Active

### DL-#9733 · Fail Fast on the Uninitialized Vendored Tools Fallback

- **State:** in_review
- **Owner:** claude
- **Issue:** #9733
- **Branch:** `claude/issue-9733-fail-fast`
- **PR:** #9743
- **Paths:** `src/__init__.py`, `tests/unit/repo_hygiene/test_src_fallback_fail_fast_9733.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`dbc6727aa`)
- **Summary:** When `vendor/ud-tools` is uninitialized, `src/__init__.py`'s
  fallback registration mistook UpstreamDrift's own aliased `shared.python` copy
  for an installed Tools distribution, installed `_VendoredToolsFallbackFinder`,
  and livelocked pytest collection in meta-path `find_spec` recursion. The
  registration probe now raises an actionable ImportError naming the remediation
  command before any finder is installed; the initialized path is unchanged.
- **Next step:** Open the PR against `main` with `Fixes #9733` and record CI.

## Shipped (Last 90 Days)

Entries stay here for 90 days after merge, then move to the archive.

## Archive

Older entries live in `DEVELOPMENT_LOG_ARCHIVE_<year>.md`.
