# Sanitation Report — 2026-10-11

Filed by the Sanitation Engineer role per
[`docs/fleet-sanitation.md`](https://github.com/D-sorganization/Repository_Management/blob/main/docs/fleet-sanitation.md).
First sanitation report ever committed in this repo — see Findings below.

## Scope

A scheduled sweep of worktree/branch/PR hygiene and root/docs clutter on
`UpstreamDrift`, run from an isolated worktree on `DeskComputer`. Did not
touch other agents' worktrees, did not merge or admin-override anything,
did not file bulk remediation issues.

## Preserved (fail-closed)

- 16 sibling `staff-worktrees/UpstreamDrift-run-*` entries — each is on a
  distinct `staff/<role>-task-*` branch, several with uncommitted or
  in-flight work (visible via `git worktree list`); none were inspected
  for dirtiness or verified against a merged PR head, because this role's
  own guard is "don't clean active agent worktrees" and six other agent
  roles are plausibly still using theirs. No deletion candidate was
  confirmed, so nothing was removed.
- One `/tmp/pytest-of-dieterolson/...` worktree entry (marked `prunable`
  by `git worktree list`) — this sandbox's Bash tool denies access to
  paths outside the repo worktree, so the target directory's actual state
  could not be verified before running `git worktree prune`. Left alone
  rather than guessed at.
- 17 open, non-draft PRs (`#12274` down to `#11359`) — all updated within
  the last ~24h, none stale by this role's criteria. No branch backing an
  open PR was touched.
- `candidates/*.json` — looked like scratch output at first glance, but
  each file is referenced by the motion-matching pipeline
  (`src/shared/python/motion_matching/...`, `tests/integration/motion_matching/`)
  and tied to tour-matching epic #9921 commits (`#9921` / #10040, #10045,
  #10047). Kept as intentional evidence, not temp clutter.

## Cleaned

- `git fetch --prune origin` — removed ~20 dead local remote-tracking
  refs (`origin/tests/*-coverage`, `origin/wip/desk-20260609`, etc.) for
  branches already deleted upstream. Local ref-list hygiene only; no
  repository state changed, nothing to commit.

## Findings (not actioned this sweep)

- **This playbook's own scaffolding was never created in UpstreamDrift.**
  `docs/fleet-sanitation.md` (source of truth, lives in
  `Repository_Management`) points to `docs/templates/sanitation-report.md`
  and to example reports at `docs/operations/worktree_cleanup_2026-04-*.md`,
  `docs/operations/stale_branch_pr_hygiene_*.md`, and
  `docs/operations/docs_ia_hygiene_*.md` — none existed in this repo before
  this PR. Every prior scheduled sanitation run had nowhere to file a
  report. This PR adds the template and this first report; the three
  example-report paths are still unfilled and should be created the next
  time this role actually executes a worktree/branch/PR cleanup (not a
  scaffolding pass).
- **`AGENT_HANDOFF.md` is 1,232 lines / ~150 KB**, well past the
  `AGENTS.md` "Agent Handoff & PR Policy" guideline of "keep it current-state
  only and under 150 lines." Rewriting a live, actively-referenced handoff
  doc is a content-judgment task (what's still load-bearing vs. stale),
  not a structural sanitation action — flagged for the document owner /
  Night Watch rather than edited here.
- 17 concurrently open non-draft PRs is below this repo's stated
  consolidation trigger (6+ open PRs *and* ≥70% runner use, or explicit
  ask) only in the sense that consolidation is a `pr-remediator`/owner
  call, not a Sanitation Engineer action — noting the count for visibility,
  not proposing a merge.

## Numbers

| Category                          | Count/Size |
| ---------------------------------- | ---------- |
| Worktrees reviewed                 | 17 (16 sibling + 1 own) |
| Worktrees preserved (fail-closed)   | 16 (own excluded) |
| Worktree-prune candidates deferred (unverifiable) | 1 |
| Stale remote-tracking refs pruned  | ~20 |
| Open non-draft PRs observed        | 17 |
| Temp artifacts archived            | 0 (none found — root and `candidates/` both legitimate) |

## Handoff

- Open question for Night Watch / owner: is the `/tmp/...` `prunable`
  worktree entry safe to clean with `git worktree prune` on a host where
  that path is reachable? This sandbox's tool policy blocked inspection.
- PR opened this sweep: see PR description for link.
