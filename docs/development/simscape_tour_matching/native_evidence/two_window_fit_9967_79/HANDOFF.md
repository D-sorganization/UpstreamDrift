# Two-Window Direct-Node SLSQP Trial 79 — Improved Uninterrupted Replay, Not Accepted

Run79 is TERMINAL (exit0, driver on runtime78,30 residual evaluations,32
window sensitivity solves,58 primal window replays,15 SLSQP iterations,
"Iteration limit reached"). Identical formulation to run78 with primal-only
residual replays and budgets max_iterations15/max_nfev45. The zero-displacement
parity gate reproduced run73's score to1.9e-12 and the first-Jacobian linear
model equals the objective at zero. The returned candidate
`5313c2833210c009503e341d7be49a0157ee6091aeefcfaa484305821d6daa3e` is an
exploratory two-window result and remains REJECTED against the25/35 mm gates.

## Uninterrupted Original-State Replay (0–0.85 s, Zero Resets)

| Metric             |  run73 |  run79 |  Change |
| ------------------ | -----: | -----: | ------: |
| Whole RMS (mm)     | 28.105 | 27.563 |   −0.54 |
| Early RMS ≤0.6 s   | 10.860 | 11.359 |   +0.50 |
| Terminal RMS (mm)  | 65.398 | 55.470 |   −9.93 |
| Club cluster (mm)  | 30.041 | 23.270 |   −6.77 |
| Pelvis yaw error % |  6.164 | 11.288 |   +5.12 |
| Score (+effort)    | 18.300 | 15.070 | −17.7 % |

The whole/terminal improvement satisfies the "useful uninterrupted improvement"
criterion of the plan, but early error and pelvis yaw worsen: the terminal
weight (100x) reshapes only the last0.1 s (see marker-comparison.png; the
error growth from0.4 s is unchanged). Closure in the returned replay: pose
1.3e-11, rate6.8e-10. Terminal replay gap segmented-versus-uninterrupted
0.20 mm. Scaled continuity defect3.99e-4 exceeds defect_tolerance1e-4, so the
fit is not accepted on continuity either; the uninterrupted replay above is
the physical statement.

## Optimizer Path

Assembled/objective/linear-model values per evaluation are in
`evaluations.jsonl` and `returned.json`. Accepted steps: evaluations4,6,8,
10,12,14,16,18,20,22,24,26,28,30 (objective18.300→15.070); every full step
was backtracked. The first-Jacobian linear model predicts the accepted steps
within2 % (e.g.15.229 predicted versus15.070 actual at eval30), so the
Jacobians are consistent with the nonlinear objective along the path. Active
bounds at the returned point:63 of123 variables: theta lower12, theta upper
27 (39 of81 controls across23 coordinates, listed in receipt/config) and24
of42 node chart coordinates at ±0.05. The inherited run73 physical box
(±2 N/Nm over parent19) and the local node box are both binding; widening
either is a separate factor requiring its own receipt.

## Evidence and Next Step

`fit.py` (SHA256 d45f9fe6…, identical to the executed remote copy), `config.json`,
`receipt.json`, `returned.json`, `returned-candidate.json`, `returned-nodes.json`,
`returned-replay.npz` (uninterrupted q/qd/markers/targets/validity/clock),
`evaluations.jsonl`, `jacobians.jsonl`, `launch.json`, `runtime-receipt.json`,
`marker-comparison.png` (local matplotlib render, visually inspected) and
`raw-run.zip` (SHA256
`5202a2faaa6b700a68ecc7caa3688181c2ac24c4706121cb062e5aef4c8407e7`; outputs,
logs, launch vector, driver, runtime78 source). Remote output
`/mnt/c/Users/diete/native-two-window-fit-9967-79`.

Run80 (`../two_window_fit_9967_80/fit.py`, SHA256 7bc33a45…) continues from
the exact returned79 candidate with the node chart recentered on its own
integrated0.6 s state (zero coordinates), the run73 physical box kept over
parent19, and the same budgets; its parity gate compares against an in-run
uninterrupted replay of the restart.
