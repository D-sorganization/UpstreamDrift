# Two-Window Direct-Node SLSQP Continuation 80 — Recentered Node, Still Rejected

Run80 is TERMINAL (exit0, runtime78,33 residual evaluations,32 sensitivity
solves,64 primal window replays,15 iterations, "Iteration limit reached").
It restarts from the exact returned run79 candidate with the0.6 s node chart
recentered on that candidate's own integrated state (zero coordinates), the
inherited run73 physical box (±2 N/Nm over parent19) recomputed relative to the
restart (max increment outside the box2.2e-14), and otherwise the run79
settings. Returned candidate
`96c786ec825535c7b88f99eaa4cbcacce175dea917f5ea27dd96316f7c247a0e`, REJECTED.

## Parity and Chart

Zero-displacement segmented objective versus an in-run uninterrupted replay of
the restart: relative difference7.1e-12; markers within3.2e-11 m; initial
scaled defect8.2e-11; reference closure1.6e-11; projected continuity rank42
(singular values1.00–348.4); first-Jacobian linear model equals the objective
at zero (15.069915).

## Uninterrupted Original-State Replay (0–0.85 s, Zero Resets)

| Metric             |  run73 |  run79 |  run80 |
| ------------------ | -----: | -----: | -----: |
| Whole RMS (mm)     | 28.105 | 27.563 | 26.797 |
| Early RMS ≤0.6 s   | 10.860 | 11.359 | 11.208 |
| Terminal RMS (mm)  | 65.398 | 55.470 | 55.208 |
| Club cluster (mm)  | 30.041 | 23.270 | 23.536 |
| Pelvis yaw error % |  6.164 | 11.288 | 11.320 |
| Score (+effort)    | 18.300 | 15.070 | 14.678 |

Objective15.070→14.679 (linear model14.680 at the returned point). Scaled
continuity defect1.13e-3 exceeds the1e-4 tolerance; segmented-versus-
uninterrupted terminal gap0.28 mm. Active bounds59 of123: theta lower12,
upper29 (41 of81 controls on the run73 box),18 of42 node coordinates at ±0.05.
Whole and terminal RMS improve again over the restart and over run73, early RMS
recovers slightly but stays above run73, pelvis yaw remains about11 %. The
25/35 mm acceptance gates are not met; the head-marker rigidity floor applies.

## Next Bounded Step

Because the inherited physical box binds41 controls, run81
(`../two_window_fit_9967_81/fit.py`, SHA256 da0edf86…) continues from exact
returned80 with exactly one documented change: `--box-factor 2` widens the
box to ±4 N/Nm over parent19. Node bound, weighting, horizon and budgets are
unchanged. A node-box or weighting change would be a further separate receipt.

## Evidence

`fit.py` (SHA256 7bc33a45…, identical to the executed remote copy),
`config.json`, `receipt.json`, `returned.json`, `returned-candidate.json`,
`returned-nodes.json`, `returned-replay.npz`, `evaluations.jsonl`,
`jacobians.jsonl`, `launch.json`, `runtime-receipt.json`,
`marker-comparison.png` (run73/79/80 valid-marker RMS versus time) and
`raw-run.zip` (SHA256
`dcd561407bec4116f3e0027e9f5fe7e4f94a59e2936c6b528db8b892f017c3e1`).
Remote output `/mnt/c/Users/diete/native-two-window-fit-9967-80`.
