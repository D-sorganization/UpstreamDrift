# Cross-engine differential-testing report v1

- Schema: `upstreamdrift.cross_engine_differential.v1`
- Generated: `2026-05-31T11:13:43+00:00`
- Status: `blocked_by_draft_dependencies`
- Source shape: `cc11`
- Engines: mujoco-canonical-v2, pinocchio-canonical-v2

## Summary

| Metric                                | Value |
| ------------------------------------- | ----: |
| comparisons                           |     0 |
| passed                                |     0 |
| failed                                |     0 |
| registered divergences                |     0 |
| contact-free torque max RMS % of peak |   n/a |
| worst tolerance ratio                 |     0 |

## Dependency Status

- CC-7 conformance harness: draft PR #6826 (https://github.com/D-sorganization/UpstreamDrift/pull/6826)
- CC-9 Pinocchio canonical-v2 adapter: draft PR #6828 (https://github.com/D-sorganization/UpstreamDrift/pull/6828)
- CC-10 MuJoCo canonical-v2 adapter: draft PR #6829 (https://github.com/D-sorganization/UpstreamDrift/pull/6829)

## Comparison Rows

No live adapter comparison rows are claimed yet. Current `origin/main` does not include the CC-7 harness or both canonical-v2 adapters.

## Regeneration

```powershell
python scripts\validation\cross_engine_differential_report.py
```
