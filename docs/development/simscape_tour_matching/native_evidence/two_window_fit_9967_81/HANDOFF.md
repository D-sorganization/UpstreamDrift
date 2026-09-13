# Two-Window Direct-Node SLSQP Continuation 81 — Widened Box, Still Rejected

Run81 is TERMINAL (exit0,1738.4 s launch-to-terminal, runtime78,31 residual
evaluations,32 sensitivity solves,60 primal window replays,15 iterations,
"Iteration limit reached"). It restarts from the exact returned run80 candidate
with one documented change: `--box-factor 2` widens the physical control box
over parent19 from ±2 to ±4 N/Nm. Node recentering, node box ±0.05, weights,
penalty, horizon and budgets equal run80. Parity gate against the in-run
uninterrupted replay of the restart:2.6e-12 relative; initial scaled defect
7.2e-11; projected continuity rank42. Returned candidate
`dfafdff1cdec1a7fa15c41a34d41f054ef45d8a7df0a1faf8fab7898ab855776`, REJECTED.

## Uninterrupted Original-State Replay (0–0.85 s, Zero Resets)

| Metric             |  run73 |  run79 |  run80 |  run81 |
| ------------------ | -----: | -----: | -----: | -----: |
| Whole RMS (mm)     | 28.105 | 27.563 | 26.797 | 26.366 |
| Early RMS ≤0.6 s   | 10.860 | 11.359 | 11.208 | 11.427 |
| Terminal RMS (mm)  | 65.398 | 55.470 | 55.208 | 46.305 |
| Club cluster (mm)  | 30.041 | 23.270 | 23.536 | 15.955 |
| Pelvis yaw error % |  6.164 | 11.288 | 11.320 | 13.923 |
| Score (+effort)    | 18.300 | 15.070 | 14.678 | 12.238 |

Linear model at the returned point12.421 versus actual12.238. Scaled
continuity defect1.61e-3 (tolerance1e-4); segmented-versus-uninterrupted
terminal gap1.16 mm. Active bounds50 of123: theta lower6, upper23 (29 of81
controls on the widened box),21 node coordinates at ±0.05; theta max3.06 N/Nm.

Reading (marker-comparison.png): all three continuations only reshape the
last0.1 s; the error growth between0.4 and0.75 s is identical to run73.
Terminal and club improve steadily, early RMS and pelvis yaw regress under the
100x terminal weight, and the whole-swing RMS improvement (1.7 mm over three
runs) is far from the25 mm gate while the fixed-attachment rigidity floor is
17.4 mm. Neither the0.85 s prefix nor the full1.81 s capture is matched.

## Next Decisions (Each Its Own Receipt)

1. Weighting: the terminal weight dominates; a run with the terminal rows
   reduced (or an early-motion weight) must be compared on the same
   uninterrupted metrics, not on score.
2. Node box: 21 chart coordinates bind at ±0.05; relaxing it is a separate
   factor and must keep the retraction radius0.5 and closure tolerance1e-8.
3. Horizon: extending beyond0.85 s requires new integrated nodes on actual
   capture samples and a same-input replay check before any fit.
4. Any accepted candidate still requires R2025b replay and visual review.

## Evidence

`fit.py` (SHA256 da0edf86…, identical to the executed remote copy), `config.json`,
`receipt.json`, `returned.json`, `returned-candidate.json`, `returned-nodes.json`,
`returned-replay.npz`, `evaluations.jsonl`, `jacobians.jsonl`, `launch.json`,
`runtime-receipt.json`, `marker-comparison.png` and `raw-run.zip` (SHA256
`a5e08f27cfffbad199894abca826c778e5e23286f02e98374d22cbba2c35c2df`). Remote
output `/mnt/c/Users/diete/native-two-window-fit-9967-81`; all processes are
terminal.
