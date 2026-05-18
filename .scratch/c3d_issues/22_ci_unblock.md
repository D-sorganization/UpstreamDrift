# chore(ci): unblock motion-matching wave PRs failing on SPEC.md / quality-gate / leaderboard

## Why

Five open motion-matching PRs are stuck behind orthogonal CI failures unrelated to their actual changes:

| PR    | Title                     | Failing checks                                                    |
| ----- | ------------------------- | ----------------------------------------------------------------- |
| #4488 | ClubBallTarget            | `quality-gate`                                                    |
| #4493 | unify C3D readers         | `regenerate cross-engine leaderboard`, `Verify SPEC.md freshness` |
| #4500 | Motion-Match Preview tile | `Verify SPEC.md freshness`                                        |
| #4505 | source-toggle UI          | `Verify SPEC.md freshness`                                        |
| #4496 | body-skeleton segments    | (last sweep clean — recheck)                                      |

All five are blocking the headline "wire C3D plot into matcher" issue (#4512). Get them green so they merge.

## What to do

Open each of the five PRs in turn (in their respective branches via worktree); for each:

1. Pull the failing-check log via `gh run view --log-failed` and identify the root cause.
2. Apply the smallest possible fix:
   - **`Verify SPEC.md freshness`** — run `python3 scripts/regenerate_spec.py` (or whatever the project uses; check `.github/workflows/spec-check.yml` for the command). Commit the regenerated SPEC.md. If the PR is documentation-only or non-spec-affecting, add the `spec-exempt` label to the PR.
   - **`quality-gate`** — usually mypy or ruff strict. Open the failing log; if it's a pre-existing baseline error in a file the PR didn't touch, leave it alone (CI baselines for that). If it's introduced by this PR, fix the typing/lint issue.
   - **`regenerate cross-engine leaderboard`** — run the leaderboard regeneration command. Commit the result.
   - **`alembic-postgres-round-trip`** — almost certainly flaky, retrigger; if persistent, leave as-is (not blocking).
3. Push the fix to the PR's branch.
4. Wait for CI to re-run and confirm green.

## Constraint

Don't change the implementation logic of any PR — this is a CI-unstick pass only. If a fix would require non-trivial code changes, leave a comment explaining what's needed and move on.

## Acceptance criteria

- [ ] Every failing CI check on each of the five PRs is either:
  - green after a fix, OR
  - exempted via the appropriate label, OR
  - flagged in a PR comment as needing maintainer attention.
- [ ] No PR has any logic change introduced by this work — only CI fix-ups (regenerated artefacts, labels, lint nits).

## Files touched

Per-PR. Likely:

- Regenerated `SPEC.md` files
- Regenerated `cross_engine_leaderboard.json` or similar
- `spec-exempt` label additions

## Out of scope

- Implementing any pending feature.
- Investigating flaky tests that aren't gating these PRs.
