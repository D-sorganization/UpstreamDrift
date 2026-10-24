# Fleet Settings Snapshots

This directory records GitHub repository settings that are not expressed in
workflow YAML but still need a versioned rollback baseline. The merge queue for
`main` is governed by repository ruleset
`Merge Queue (PR Fast Lane, RM#1900)` (see D-sorganization/Repository_Management#1900).

## Merge Queue Batch Size (UD-OPS-033)

| Snapshot                                             | Purpose                                                                 |
| ---------------------------------------------------- | ----------------------------------------------------------------------- |
| `UpstreamDrift-merge-queue-baseline-2026-10-04.json` | Pre-experiment settings (`max_entries_*` = 5) from the RM#1900 rollout  |
| `UpstreamDrift-merge-queue-target.json`              | Target experiment settings (`max_entries_*` = 3, within the 2–3 window) |

Apply the target snapshot with:

```bash
python3 scripts/ci/apply_fleet_merge_queue_settings.py \
  --snapshot docs/operations/settings-snapshots/UpstreamDrift-merge-queue-target.json
```

Validate (offline structure check, or live against GitHub):

```bash
python3 scripts/check_fleet_merge_settings.py
python3 scripts/check_fleet_merge_settings.py --live
```

Rollback: re-apply the baseline snapshot file above, then confirm with
`--live`.

Experiment protocol and revert triggers:
[`merge-queue-batch-experiment.md`](../merge-queue-batch-experiment.md).
