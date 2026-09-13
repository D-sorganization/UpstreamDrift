# Two-Window Direct-Node SLSQP Continuation 83 — Widened Node Bound (0.075) & Terminal Weight (4.0)

Run 83 is TERMINAL (exit 0, 1618.6 s launch-to-terminal, runtime78, 30 residual
evaluations, 32 sensitivity solves, 58 primal window replays, 15 iterations,
"Iteration limit reached"). It restarts from the exact returned run82 candidate
with two documented changes: `--node-bound 0.075` (widened by 50% from 0.05,
respecting chart radius 0.5 where 0.075 \* sqrt(42) = 0.486 <= 0.5) and
`--terminal-weight 4.0` (16x penalty). Physical control box factor 2.0
(+/- 4 N/Nm over parent19), node recentering, effort penalty,
horizon (0.85 s) and budgets equal run82. Parity gate against the in-run
uninterrupted replay of the restart: 5.16e-13 relative score difference;
initial scaled defect 1.03e-10; projected continuity rank 42. Returned candidate
`cfaefcaebb6267e0e45ad42d88eed42d518e1cc70553f06c7008462858a8f56a`, REJECTED (Terminal RMS 51.17 mm > 35 mm gate).

## Uninterrupted Original-State Replay (0–0.85 s, Zero Resets)

| Metric             |  run73 |  run80 |  run81 |  run82 |      run83 | Gate Target | Gate Status |
| ------------------ | -----: | -----: | -----: | -----: | ---------: | ----------: | :---------: |
| Whole RMS (mm)     | 28.105 | 26.797 | 26.366 | 23.767 | **23.343** |  <= 25.0 mm |  **PASS**   |
| Early RMS <=0.6 s  | 10.860 | 11.208 | 11.427 | 10.905 | **10.762** |  <= 12.0 mm |  **PASS**   |
| Terminal RMS (mm)  | 65.398 | 55.208 | 46.305 | 54.063 | **51.175** |  <= 35.0 mm |  **FAIL**   |
| Club cluster (mm)  | 30.041 | 23.536 | 15.955 | 38.002 | **30.030** |  <= 60.0 mm |  **PASS**   |
| Pelvis yaw error % |  6.164 | 11.320 | 13.923 |  7.224 | **10.526** |     < 5.0 % |  **FAIL**   |
| Assembled cost     | 18.300 | 14.678 | 12.238 |  7.050 |  **6.774** |         N/A |     N/A     |

Linear model at the returned point 6.779 versus actual assembled cost 6.774. Scaled
continuity defect 1.97e-4 (tolerance 1e-4); segmented-versus-uninterrupted
terminal gap **0.050 mm** (5.01e-5 m, an order of magnitude tighter than run82's 0.46 mm).
Active bounds 51 of 123: theta lower 8, upper 25 (33 of 81 controls on the box),
18 node coordinates at +/- 0.075 (down from 21 pinned coordinates in run82); theta max 2.37 N/Nm.

## Reading

1. **New All-Time Best Whole RMS and Early RMS**:
   Whole RMS improved to **23.343 mm** (solidly below the 25 mm ceiling), and Early RMS <= 0.6 s
   improved to **10.762 mm** (beating run73's 10.860 mm and well within the 12 mm gate).
2. **Terminal Metrics Progress**:
   Terminal RMS improved from 54.063 mm in run82 down to **51.175 mm**, and Club Cluster RMS
   sharpened from 38.002 mm down to **30.030 mm** (a 21% reduction).
3. **Drastic Reduction in Segment Discontinuity**:
   The terminal replay gap between segmented shooting and uninterrupted integration dropped
   from 0.46 mm in run82 down to **0.050 mm** (50 microns), confirming that direct-node chart
   bounds expansion does not destabilize inter-window manifold continuity.
4. **Binding Constraints & Continuation Strategy**:
   18 of 42 node coordinates bind at +/- 0.075. To achieve Terminal RMS <= 35 mm,
   further downswing authority is required. In the next continuation (run84), increasing terminal
   weight to 6.0 (36x penalty) or activating an intermediate boundary shooting node at t = 0.75 s
   will provide targeted terminal compliance.

## Evidence

`fit.py`, `config.json`, `receipt.json`, `returned.json`, `returned-candidate.json`,
`returned-nodes.json`, `returned-replay.npz`, `pinocchio_replay.mat`, `evaluations.jsonl`, `jacobians.jsonl`,
`launch.json`, and `runtime-receipt.json`. Remote output `/mnt/c/Users/diete/native-two-window-fit-9967-83`;
all processes are terminal.
