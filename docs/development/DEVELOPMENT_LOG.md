# Development Log — UpstreamDrift

State table for every feature in flight in this repository. Update entries
**in place**; never append dated sections. One entry per feature, from proposal
to ship. See the `development-logs` section of `AGENTS.md` for the binding rules
and `shared_scripts/development_log.py` for the validator.

- **Portfolio:** golf
- **WIP limit:** 5
- **Last audited:** 2026-09-08 by W2_9249 (claude)

## States

`proposed` → `in_progress` → `in_review` → `shipped`, with `parked` reachable
from any live state and `abandoned` from `parked`. `shipped` never returns to
`in_progress`; open a new entry instead.

## Active

### DL-#9249 · UI: pin @vitejs/plugin-react to ^5 until Vite 8

- **State:** in_review
- **Owner:** claude
- **Issue:** #9249
- **Branch:** claude/issue-9249-ui-pin
- **PR:** not created
- **Paths:** `.github/dependabot.yml`, `ui/README.md`, `AGENT_HANDOFF.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (SELF)
- **Summary:** Dependabot ignores `@vitejs/plugin-react` major updates because 6.x needs Vite 8 (Vite 7 exports no `./internal`); the pairing constraint is documented in `ui/README.md`.
- **Next step:** Merge the guard PR; revisit the paired vite@8 + plugin-react@6 upgrade once `vitest`/`@react-three/*` are Vite-8 ready.

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