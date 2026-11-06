# PR Size Advisory (#14741)

This repository measures **changed lines** (added + deleted from `git diff --numstat`
on the merge-base triple-dot diff) for each pull request. That number is a **coding
standard**, not a hard merge gate: the `pr-size` job in `.github/workflows/pr-size-gate.yml`
always succeeds and emits `::notice` / `::warning` annotations. When the count is over
the reference cap (default 500), CI also applies the `size:large` label (best-effort).

Prefer small, cohesive PRs; do **not** split a cohesive fix only to stay under a line
count.

## Tooling

- `scripts/ci/check_pr_diff_size.py` — computes the count and writes step outputs
  (`within_limit`, `advisory`, `total_lines`). Default exit code is **0**.
- `scripts/config/pr_size_allowlist.json` — optional list of PR numbers for bookkeeping in messages.
- `--strict` on the script opts into **exit 1** when over the cap and not exempt (tests
  or local policy only; CI does not pass `--strict`).

Exemptions (still reported, advisory only): deletion-only diffs, label `golden-regen`,
or allowlisted PR numbers.
