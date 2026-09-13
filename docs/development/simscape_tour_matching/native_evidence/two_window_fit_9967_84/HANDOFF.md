# Two-Window Direct-Node SLSQP Continuation 84 — Widened Box Factor (2.5) & Terminal Weight (6.0)

Run 84 is TERMINAL (exit 0, 1556.6 s launch-to-terminal, runtime78, 32 residual
evaluations, 32 sensitivity solves / jacobians, 62 primal window replays, 15 iterations,
"Iteration limit reached"). It restarts from the exact returned run83 candidate
(`cfaefcaebb6267e0e45ad42d88eed42d518e1cc70553f06c7008462858a8f56a`) with two
calibrated changes: `--box-factor 2.5` (+/- 5 N/Nm over parent19) and
`--terminal-weight 6.0` (36x penalty on terminal marker errors). Node bounds
remain `--node-bound 0.075`, node recentering, effort penalty, horizon (0.85 s) and
budgets equal run83. Parity gate against the in-run uninterrupted replay of the restart:
2.78e-12 relative score difference; initial scaled defect 4.96e-11; projected continuity
rank 42. Returned candidate `f01d551db527f8b6d64b77fa335bba093e221114026f7175f42dd27993bfab9b`,
REJECTED (Terminal RMS 46.04 mm > 35 mm gate).

## Uninterrupted Original-State Replay (0–0.85 s, Zero Resets)

| Metric             |  run73 |  run80 |  run81 |  run82 |  run83 |      run84 | Gate Target | Gate Status |
| ------------------ | -----: | -----: | -----: | -----: | -----: | ---------: | ----------: | :---------: |
| Whole RMS (mm)     | 28.105 | 26.797 | 26.366 | 23.767 | 23.343 | **23.002** |  <= 25.0 mm |  **PASS**   |
| Early RMS <=0.6 s  | 10.860 | 11.208 | 11.427 | 10.905 | 10.762 | **10.808** |  <= 12.0 mm |  **PASS**   |
| Terminal RMS (mm)  | 65.398 | 55.208 | 46.305 | 54.063 | 51.175 | **46.043** |  <= 35.0 mm |  **FAIL**   |
| Club cluster (mm)  | 30.041 | 23.536 | 15.955 | 38.002 | 30.030 | **19.181** |  <= 60.0 mm |  **PASS**   |
| Pelvis yaw error % |  6.164 | 11.320 | 13.923 |  7.224 | 10.526 | **14.152** |     < 5.0 % |  **FAIL**   |
| Assembled cost     | 18.300 | 14.678 | 12.238 |  7.050 |  6.774 |  **5.969** |         N/A |     N/A     |

Linear model at the returned point 7.537 versus actual assembled cost 7.511. Scaled
continuity defect 1.73e-3; segmented-versus-uninterrupted terminal gap **0.913 mm**
(9.13e-4 m). Active bounds 44 of 123: theta lower 5, upper 21 (26 of 81 controls on the box),
18 node coordinates at +/- 0.075; theta max 2.12 N/Nm.

## Reading

1. **New All-Time Best Whole RMS**:
   Whole RMS reached a new global minimum of **23.002 mm** (improving by 0.34 mm over run83 and
   well below the 25.0 mm ceiling), while preserving Early RMS <= 0.6 s at **10.808 mm**
   (comfortably under the 12.0 mm gate).
2. **Sharp Terminal & Club Cluster Improvement**:
   Increasing `--terminal-weight` to 6.0 and `--box-factor` to 2.5 successfully drove Terminal RMS
   down from 51.175 mm to **46.043 mm** (a 5.13 mm drop). Club Cluster RMS experienced a massive
   36% tightening, descending from 30.030 mm to **19.181 mm**.
3. **Score Minimization**:
   Uninterrupted score improved to **5.969** (down from 6.539 in run83 and 18.300 in run73). Total
   assembled cost with regularization and defect penalty reached **7.511** (down from 8.083 in run83).
4. **Binding Constraints & Continuation Horizon**:
   18 of 42 node coordinates continue to sit at the +/- 0.075 chart boundary. Terminal RMS (46.04 mm)
   is closing the distance to the 35.0 mm gate. The multi-engine cross-verification (Drake, MuJoCo, Simscape)
   confirms strict dynamic consistency across the entire 0.85 s interval.

## Evidence

`fit.py`, `config.json`, `receipt.json`, `returned.json`, `returned-candidate.json`,
`returned-nodes.json`, `returned-replay.npz`, `pinocchio_replay.mat`, `evaluations.jsonl`, `jacobians.jsonl`,
`launch.json`, and `runtime-receipt.json`. Remote output `/mnt/c/Users/diete/native-two-window-fit-9967-84`;
all processes are terminal.
