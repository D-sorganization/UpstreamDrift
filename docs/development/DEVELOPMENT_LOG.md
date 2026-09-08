# Development Log — UpstreamDrift

State table for every feature in flight in this repository. Update entries
**in place**; never append dated sections. One entry per feature, from proposal
to ship.

- **Portfolio:** golf
- **WIP limit:** 3
- **Last audited:** 2026-09-07 by claude

## States

`proposed` → `in_progress` → `in_review` → `shipped`, with `parked` reachable
from any live state and `abandoned` from `parked`. `shipped` never returns to
`in_progress`; open a new entry instead.

## Active

### DL-#9699 · Unify model-generation MJCF conversion handlers; 422 for malformed XML

- **State:** in_review
- **Owner:** claude
- **Issue:** #9699
- **PR:** not created
- **Branch:** `claude/issue-9699-mjcf-422`
- **Paths:** `src/shared/python/model_generation/api/**`,
  `tests/unit/tools/model_generation/**`
- **Started:** 2026-09-07
- **Last verified:** 2026-09-07 (`SELF`)
- **Summary:** `convert_mjcf_to_urdf` existed as three independent copies
  (`rest_api_routes`, `rest_api_generation` mixin, `generation_handlers`), each
  with a defective `except (ValueError, KeyError, OSError)` clause that let
  `ET.ParseError` (a `SyntaxError` subclass) escape into a 500. The canonical
  conversion core now lives once in `rest_api_support.mjcf_to_urdf_response`
  with `ET.ParseError` mapped to 422; the three handler copies delegate to it.
  The stale 501 assertions in `test_remove_not_implemented` (now
  `test_remove_model_succeeds`) and `test_path_param_value_is_captured` were
  rewritten to assert the implemented behavior.
- **Next step:** Merge the PR protected once CI `quality-gate` passes
  (`Fixes #9699` closes it).

## Shipped (Last 90 Days)

(None yet — this log was first created with DL-#9699.)

## Archive

Older entries live in `DEVELOPMENT_LOG_ARCHIVE_<year>.md`.