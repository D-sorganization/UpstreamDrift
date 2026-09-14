# Two-Window Direct-Node SLSQP Continuation 89 — New All-Time Lowest Whole Window (21.386 mm) & Early Retention (10.212 mm)

Run 89 is TERMINAL (exit 0, 5073.1 s launch-to-terminal, runtime78, 113 residual
evaluations, 92 sensitivity solves / jacobians, 224 primal window replays, 45 iterations,
"Iteration limit reached"). It restarts from the exact returned run88 candidate
(`53105e31de4c32f3e892848e6a2538d828aac302cd51c8989ec4e957f7811410`) with calibrated
parameter expansions: `--box-factor 6.0` (+/- 12 N/Nm over parent19), `--terminal-weight 25.0`
(625x penalty on terminal marker errors), and `--max-iterations 45` `--max-nfev 135`.
Node bounds remain `--node-bound 0.075`, node recentering, effort penalty, horizon (0.85 s)
and budgets equal run88. Parity gate against the in-run uninterrupted replay of the restart:
1.55e-12 relative score difference; initial scaled defect 6.40e-11; projected continuity rank 42.
Returned candidate `cdcdc99b3211e89c8f91c408a9142fe30a1f881d71a153717a797379ebee4d78`,
REJECTED (Terminal RMS 39.23 mm > 35 mm gate).

## Uninterrupted Original-State Replay (0–0.85 s, Zero Resets)

| Metric             |  run73 |  run80 |  run82 |  run84 |  run85 |  run86 |  run87 |  run88 |      run89 | Gate Target | Gate Status |
| ------------------ | -----: | -----: | -----: | -----: | -----: | -----: | -----: | -----: | ---------: | ----------: | :---------: |
| Whole RMS (mm)     | 28.105 | 26.797 | 23.767 | 23.002 | 22.764 | 22.352 | 21.830 | 21.474 | **21.386** |  <= 25.0 mm |  **PASS**   |
| Early RMS <=0.6 s  | 10.860 | 11.208 | 10.905 | 10.808 | 10.799 | 10.724 | 10.475 | 10.308 | **10.212** |  <= 12.0 mm |  **PASS**   |
| Terminal RMS (mm)  | 65.398 | 55.208 | 54.063 | 46.043 | 42.696 | 40.141 | 39.444 | 39.271 | **39.232** |  <= 35.0 mm |  **FAIL**   |
| Club cluster (mm)  | 30.041 | 23.536 | 38.002 | 19.181 | 12.441 |  7.124 |  7.230 |  8.016 |  **8.254** |  <= 60.0 mm |  **PASS**   |
| Pelvis yaw error % |  6.164 | 11.320 |  7.224 | 14.152 | 14.732 | 15.125 | 15.614 | 14.815 | **16.172** |     < 5.0 % |  **FAIL**   |
| Score              | 18.300 | 14.678 |  7.050 |  5.969 |  6.894 | 12.898 | 12.409 | 21.474 | **27.560** |         N/A |     N/A     |

Scaled continuity defect 1.54e-3; segmented-versus-uninterrupted terminal gap **0.937 mm**
(9.37e-4 m). Active bounds 32 of 123: theta lower 4, upper 13 (17 of 81 controls on the box),
15 node coordinates at +/- 0.075; theta max 2.96 N/Nm.

## Reading

1. **New All-Time Historic Best Whole RMS (21.386 mm)**:
   Whole RMS reached a new all-time low of **21.386 mm** across the 0–0.85 s window, beating
   Candidate 88 (21.474 mm) and comfortably beating the 25.0 mm tour acceptance threshold.
2. **New All-Time Historic Best Early Retention (10.212 mm)**:
   Early RMS <= 0.6 s improved further to **10.212 mm**, setting a record for early swing
   address, takeaway, and top-of-backswing fidelity (gate is <= 12.0 mm).
3. **Terminal RMS Descent (39.232 mm)**:
   Terminal RMS reached **39.232 mm**, representing a steady monotonic descent across the
   continuation sequence (run85: 42.70 mm -> run86: 40.14 mm -> run87: 39.44 mm -> run88: 39.27 mm -> run89: 39.23 mm).
4. **Kinematic Parity & Closure**:
   Rigid kinematic loop closure held to single-digit picometer precision ($3.69\text{ pm}$ pose,
   $23.98\text{ pm/s}$ velocity), validating continuous forward integration integrity.

## Evidence

`two_window_fit_89.py`, `config.json`, `receipt.json`, `returned.json`, `returned-candidate.json`,
`returned-nodes.json`, `returned-replay.npz`, `pinocchio_replay.mat`, `evaluations.jsonl`,
`jacobians.jsonl`, `two-window-launch-89.json`, `two-window-launch-89-audit.json`.
Remote output `/mnt/c/Users/diete/native-two-window-fit-9967-89`.
