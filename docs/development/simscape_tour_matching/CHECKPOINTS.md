# Incremental Checkpoints and Replay (Current)

Canonical handoff: [AGENT_HANDOFF.md](../../../AGENT_HANDOFF.md).
Historical checkpoint narrative (prefix-100ms through early 0.75 s runs) lives in
[CHECKPOINTS_HISTORY.md](CHECKPOINTS_HISTORY.md) and must not be treated as
current resume instructions.

## Current Resume Entry (MS-60 / Run-102)

Best committed Simscape matched-swing evidence for the 0–0.85 s horizon is
`native_evidence/two_window_fit_9967_102/`:

| Artifact                      | Path                                                                      |
| ----------------------------- | ------------------------------------------------------------------------- |
| MatchedSwingCandidate package | `native_evidence/two_window_fit_9967_102/candidate.npz`                   |
| R2025b run manifest           | `native_evidence/two_window_fit_9967_102/run_manifest.json`               |
| Marker overlay playback       | `native_evidence/two_window_fit_9967_102/playback.gif`                    |
| Qualified cold-replay receipt | `native_evidence/two_window_fit_9967_102/qualified_candidate_replay.json` |
| Native replay entry           | `native_evidence/two_window_fit_9967_102/replay_returned102_r2025b.m`     |

Headline metrics (from `qualified_candidate_replay.json`): whole **20.267 mm**,
early **9.995 mm**, terminal **40.301 mm**, club **8.389 mm**, pelvis yaw
**0.610%**, Simscape-vs-Pinocchio max Euclidean **60.5 µm**. Terminal gate
(≤ 35 mm) remains open; do not claim G1/G3 acceptance from this alone.

## Reproducible Runner (R2025b Only)

```powershell
powershell scripts/matlab/run_simscape_candidate.ps1 -Run two_window_fit_9967_102 -Replay
```

The script launches `C:\Program Files\MATLAB\R2025b\bin\matlab.exe` explicitly
and refuses other releases. Unlicensed default CI must not treat a missing
MATLAB as a native pass. Refresh the Python package after a licensed replay:

```bash
PYTHONPATH=. python scripts/matlab/materialize_run102_candidate_package.py
```

## Contract Reminders

- MATLAB release: **R2025b only** (no R2026a substitution).
- DbC: `write_run_manifest.m` / `export_candidate.m` fail closed without host,
  SHAs, or release identity; Python schema is
  `src/shared/python/motion_matching/simscape_run_manifest.py`.
- Connect model/workspace assets through MS-102/103 when those leases are free.
