# Jules Push/PR Noise — Batch 1 (UD-CI-003)

Frozen copies of workflows removed from `.github/workflows/` by issue #13575.
GitHub Actions does not load YAML from this directory; copies exist for rollback only.

To restore a workflow, move its file back to `.github/workflows/` and re-enable it
with `gh workflow enable .github/workflows/<file>`.
