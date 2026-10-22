# Merge Queue Batch Size Experiment (UD-OPS-033)

Parent epic: [Repository_Management#2215](https://github.com/D-sorganization/Repository_Management/issues/2215).
Tracking issue: UpstreamDrift#13574.

## Change

Lower merge-queue grouping from five concurrent entries to three (within the
2–3 acceptance window) while full-tier CI pass rates are unstable. Settings are
stored in
[`settings-snapshots/UpstreamDrift-merge-queue-target.json`](settings-snapshots/UpstreamDrift-merge-queue-target.json)
and applied with `scripts/ci/apply_fleet_merge_queue_settings.py`.

## Before / After

| Field                  | Baseline (2026-10-04) | Target |
| ---------------------- | --------------------: | -----: |
| `max_entries_to_build` |                     5 |      3 |
| `max_entries_to_merge` |                     5 |      3 |

Baseline file: `settings-snapshots/UpstreamDrift-merge-queue-baseline-2026-10-04.json`.

## Observation Window

Fourteen calendar days starting when the target snapshot is applied to GitHub
and `python3 scripts/check_fleet_merge_settings.py --live` reports a match.

## Metrics (RM-CI-000)

Pull merge-throughput and merge-queue eviction counts from the Runner Dashboard
RM-CI-000 baseline export for `D-sorganization/UpstreamDrift`. Compare the
14-day experiment window to the prior 14-day baseline window.

## Revert Trigger

Re-apply the baseline snapshot when **either**:

1. Merge-queue eviction rate rises more than 20% relative to the baseline window, or
2. Median time-to-merge for armed PRs rises more than 30% relative to baseline.

Record the revert in a follow-up issue comment with the RM-CI-000 window dates
and the `check_fleet_merge_settings.py --live` confirmation after rollback.
