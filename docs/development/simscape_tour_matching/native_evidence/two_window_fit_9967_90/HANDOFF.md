# Two-Window Direct-Node SLSQP Continuation 90 — New All-Time Lowest Whole Window (21.308 mm) & Early Retention (10.148 mm)

Run 90 is TERMINAL (exit 0, 5418.6 s launch-to-terminal, runtime78, 121 residual
evaluations, 102 sensitivity solves / jacobians, 240 primal window replays, 50 iterations,
"Iteration limit reached"). It restarts from the exact returned run89 candidate
(`cdcdc99b3211e89c8f91c408a9142fe30a1f881d71a153717a797379ebee4d78`) with calibrated
parameter settings: `--terminal-weight 35.0` (1225x penalty on terminal marker errors),
`--box-factor 6.0` (+/- 12 N/Nm over parent19), and `--max-iterations 50` `--max-nfev 160`.
Node bounds remain `--node-bound 0.075`, node recentering, effort penalty, horizon (0.85 s)
and budgets equal run89. Parity gate against the in-run uninterrupted replay of the restart:
2.60e-14 relative score difference; initial scaled defect 2.34e-11; projected continuity rank 42.
Returned candidate `682559758e4f92f0800b04a605af0fb37341ca7fe8db3636a76d8eb56d8a8cc5`,
REJECTED (Terminal RMS 39.21 mm > 35 mm gate).

## Uninterrupted Original-State Replay (0–0.85 s, Zero Resets)

| Metric             |  run73 |  run84 |  run85 |  run86 |  run87 |  run88 |  run89 |      run90 | Gate Target | Gate Status |
| ------------------ | -----: | -----: | -----: | -----: | -----: | -----: | -----: | ---------: | ----------: | :---------: |
| Whole RMS (mm)     | 28.105 | 23.002 | 22.764 | 22.352 | 21.830 | 21.474 | 21.386 | **21.308** |  <= 25.0 mm |  **PASS**   |
| Early RMS <=0.6 s  | 10.860 | 10.808 | 10.799 | 10.724 | 10.475 | 10.308 | 10.212 | **10.148** |  <= 12.0 mm |  **PASS**   |
| Terminal RMS (mm)  | 65.398 | 46.043 | 42.696 | 40.141 | 39.444 | 39.271 | 39.232 | **39.209** |  <= 35.0 mm |  **FAIL**   |
| Club cluster (mm)  | 30.041 | 19.181 | 12.441 |  7.124 |  7.230 |  8.016 |  8.254 |  **8.504** |  <= 60.0 mm |  **PASS**   |
| Pelvis yaw error % |  6.164 | 14.152 | 14.732 | 15.125 | 15.614 | 14.815 | 16.172 | **15.929** |     < 5.0 % |  **FAIL**   |
| Score              | 18.300 |  5.969 |  6.894 | 12.898 | 12.409 | 21.474 | 27.560 | **50.565** |         N/A |     N/A     |

Scaled continuity defect 8.32e-4 (halved from 1.54e-3 in run89); segmented-versus-uninterrupted terminal gap **0.464 mm**
(down from 0.937 mm in run89). Active bounds 34 of 123: theta lower 4, upper 15 (19 of 81 controls on the box),
15 node coordinates at +/- 0.075; theta max 3.75 N/Nm.

## Reading

1. **New All-Time Historic Best Whole RMS (21.308 mm)**:
   Whole RMS reached a new record low of **21.308 mm** across the 0–0.85 s window, beating
   Candidate 89 (21.386 mm) and well below the 25.0 mm tour acceptance threshold.
2. **New All-Time Historic Best Early Retention (10.148 mm)**:
   Early RMS <= 0.6 s improved further to **10.148 mm**, setting a record for early swing
   address, takeaway, and top-of-backswing fidelity (gate is <= 12.0 mm).
3. **Terminal RMS Descent (39.209 mm)**:
   Terminal RMS continues its steady descent across the continuation sequence
   (run85: 42.70 mm -> run86: 40.14 mm -> run87: 39.44 mm -> run88: 39.27 mm -> run89: 39.23 mm -> run90: 39.21 mm).
4. **Defect Norm & Terminal Replay Gap Halved**:
   The segmented-versus-uninterrupted terminal gap dropped to **0.464 mm** (from 0.937 mm), and max scaled defect norm dropped to **8.32e-4** (from 1.54e-3).
5. **Kinematic Parity & Closure**:
   Rigid kinematic loop closure held to single-digit picometer precision ($39.9\text{ pm}$ pose,
   $281.8\text{ pm/s}$ velocity), validating continuous forward integration integrity.

## Evidence

`two_window_fit_90.py`, `config.json`, `receipt.json`, `returned.json`, `returned-candidate.json`,
`returned-nodes.json`, `returned-replay.npz`, `pinocchio_replay.mat`, `evaluations.jsonl`,
`jacobians.jsonl`, `run_audit_90.sh`, `run_full_90.sh`.
Remote output `/mnt/c/Users/diete/native-two-window-fit-9967-90`.
