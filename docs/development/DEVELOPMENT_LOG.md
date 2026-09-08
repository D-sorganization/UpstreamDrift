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


### DL-#9542 · Bunker Exit State Consistency, Provenance, and Result Envelope

- **State:** in_review
- **Owner:** claude
- **Issue:** #9542
- **Branch:** `claude/issue-9542-exit-envelope`
- **PR:** #9728 (open)
- **Paths:** `src/bunkershot3d/ball/**`
- **Started:** 2026-09-07
- **Last verified:** 2026-09-07 (`SELF`)
- **Summary:** `SandDelivery` now refuses contradictory exit speed/vector pairs and owns copies of list-supplied exit vectors so post-construction mutation cannot invalidate the frozen record; the `to_post_impact_state` boundary carries explicit `ExitVectorProvenance` labels, and `PostImpactEnvelope` wraps the flight handoff with the validity verdict, F0 tier, per-group frames, the proper `HEAD_FRAME_TO_FLIGHT_TRANSFORM`, a schema version, and a SHA-256 source digest with JSON round trip. Reflection rejection itself was already delivered by PR #9574 and is not redone.
- **Next step:** Open the protected PR to `main` with `Fixes #9542`, label `agent:claude`, and RED/GREEN evidence in the body.

## Shipped (Last 90 Days)

Entries stay here for 90 days after merge, then move to the archive.

## Archive

Older entries live in `DEVELOPMENT_LOG_ARCHIVE_<year>.md`.

## Field Reference

| Field           | Required                   | Notes                                                          |
| --------------- | -------------------------- | -------------------------------------------------------------- |
| `State`         | Always                     | One of the six states above                                    |
| `Owner`         | Always                     | Agent id from the fleet roster, or `unassigned`                |
| `Issue`         | While live                 | Governing GitHub issue; enforces the entry/issue join          |
| `Branch`        | `in_progress`, `in_review` | Enforces the entry/branch join                                 |
| `PR`            | Always                     | Number and state, or `not created`                             |
| `Paths`         | Always                     | Globs; drives silent-entry detection                           |
| `Started`       | Always                     | Drives cycle time                                              |
| `Last verified` | Always                     | Date plus SHA — the liveness signal                            |
| `Summary`       | Always                     | One or two sentences                                           |
| `Next step`     | While live                 | Exactly one action; if it needs two sentences, split the entry |
| `Parked`        | When `parked`              | Date plus reason                                               |

Never place credentials, tokens, or customer data in a development log.

### DL-#9387 · Unit-Gate Worker Corruption: `src`-Identity Sentinel and Leak Fixes

- **State:** in_review
- **Owner:** claude
- **PR:** #9741 (open; `Fixes #9387`)
- **Paths:** `tests/unit/repo_hygiene/test_src_identity_sentinel.py`,
  `tests/imports/test_gui_import_boundaries.py`,
  `tests/integration/test_golf_launcher_integration.py`,
  `tests/unit/engines/pinocchio/test_tasks.py`,
  `tests/unit/test_ux_enhancements.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`c70d5ddae`)
- **Summary:** Static audit of `tests/` (conftests excluded) found 83
  `sys.modules['src*']` mutation sites in 23 files with 0 unambiguous
  leakers; the judgment-call leaks (ux-enhancements fixture, pinocchio
  tasks fixture, GUI import-boundary drops, golf-launcher pops) are now
  explicitly snapshot/restored, and a runtime sentinel runs each of the
  four documented victim files in a serial subprocess asserting
  `sys.modules['src']` identity and the `src.*` namespace are unchanged.
- **Next step:** Watch this PR's `quality-gate` run once after opening;
  on green, protected squash merge closes #9387, then rerun the flaky
  `unit-test-gate` histories of #9384/#9374/#9404 to confirm no fresh
  worker-corruption victims appear.

### DL-#9494 · Resolve the CLAUDE.md `--no-verify` Contradiction by Fixing the Windows Hook Environment

- **State:** shipped
- **Owner:** claude
- **Issue:** #9494
- **Branch:** `claude/issue-9494-precommit-env`
- **PR:** #9744
- **Paths:** `CLAUDE.md`, `AGENT_HANDOFF.md`, `docs/development/DEVELOPMENT_LOG.md`, `SPEC.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`dbc6727aa`)
- **Summary:** CLAUDE.md forbade `git commit --no-verify` while agents on
  Windows reported every pre-commit invocation failing (hook virtualenvs
  targeting Python 3.11, absent from the workstation). Investigation found no
  interpreter pin left in `.pre-commit-config.yaml` (`default_language_version:
python: python`, 3.11 pin removed by #1792/#2720), and on Python 3.13.3 every
  commit-stage hook plus pre-push `mypy`/`bandit` passes after a from-scratch
  environment build. Option (a) of the issue is therefore satisfied; CLAUDE.md
  now documents the resolved environment and states that the `--no-verify`
  prohibition stands on Windows with no blanket exception.
- **Next step:** Record CI on PR #9744; on merge, confirm the protected-main
  sync lands the resolved hook environment note.

## Shipped (Last 90 Days)

Entries stay here for 90 days after merge, then move to the archive.

## Archive

Older entries live in `DEVELOPMENT_LOG_ARCHIVE_<year>.md`.
