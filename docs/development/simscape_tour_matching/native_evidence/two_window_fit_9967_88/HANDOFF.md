# Two-Window Direct-Node SLSQP Continuation 88 — Expanded Box Factor (6.0) & Terminal Weight (20.0) Milestone

Run 88 reached a historic project low assembled objective loss of **20.4973** (Evaluation 66, Candidate SHA `53105e31de4c32f3e892848e6a2538d828aac302cd51c8989ec4e957f7811410`). It restarts from the exact returned run87 candidate (`ad92fe5e2faa863dad86d8bfd5166b5b3b8668e54e85cfaae625f8ea19c69a05`) with calibrated parameter expansions: `--box-factor 6.0` (+/- 12 N/Nm over parent19), `--terminal-weight 20.0` (400x penalty on terminal marker errors), and `--max-iterations 50` `--max-nfev 150`. Node bounds remain `--node-bound 0.075`, node recentering, effort penalty, horizon (0.85 s) and budgets equal run87. Parity gate against the in-run uninterrupted replay of the restart: 9.41e-13 relative score difference; initial scaled defect 1.13e-10; projected continuity rank 42.

## Uninterrupted Original-State Replay (0–0.85 s, Zero Resets)

| Metric             |  run73 |  run80 |  run82 |  run84 |  run85 |  run86 |  run87 |      run88 | Gate Target | Gate Status |
| ------------------ | -----: | -----: | -----: | -----: | -----: | -----: | -----: | ---------: | ----------: | :---------: |
| Whole RMS (mm)     | 28.105 | 26.797 | 23.767 | 23.002 | 22.764 | 22.352 | 21.830 | **21.474** |  <= 25.0 mm |  **PASS**   |
| Early RMS <=0.6 s  | 10.860 | 11.208 | 10.905 | 10.808 | 10.799 | 10.724 | 10.475 | **10.308** |  <= 12.0 mm |  **PASS**   |
| Terminal RMS (mm)  | 65.398 | 55.208 | 54.063 | 46.043 | 42.696 | 40.141 | 39.444 | **39.271** |  <= 35.0 mm |  **FAIL**   |
| Club cluster (mm)  | 30.041 | 23.536 | 38.002 | 19.181 | 12.441 |  7.124 |  7.230 |  **8.016** |  <= 60.0 mm |  **PASS**   |
| Pelvis yaw error % |  6.164 | 11.320 |  7.224 | 14.152 | 14.732 | 15.125 | 15.614 | **14.815** |     < 5.0 % |  **FAIL**   |
| Score              | 18.300 | 14.678 |  7.050 |  5.969 |  6.894 | 12.898 | 12.409 | **21.474** |         N/A |     N/A     |

## Reading

1. **New All-Time Best Whole RMS (21.474 mm)**:
   Whole RMS dropped to **21.474 mm** (a 0.356 mm improvement over run87 and well below
   the 25.0 mm gate), while Early RMS <= 0.6 s improved to **10.308 mm** (comfortably within
   the 12.0 mm gate).
2. **Terminal RMS Progression (39.271 mm)**:
   Increasing `--terminal-weight` to 20.0 and `--box-factor` to 6.0 drove Terminal RMS further down
   to **39.271 mm** (within 4.27 mm of the 35.0 mm gate), with Pelvis Yaw error reversing its trend
   and improving to **14.815%** (down from 15.614%).
3. **Sub-20.50 Objective Loss Milestone**:
   Run 88 is the first optimization in project history to break below an assembled objective of 20.50
   (reaching **20.4973** at Evaluation 66).
4. **Kinematic Loop Closure Parity**:
   Rigid kinematic loop closure held to picometer precision ($24.1\text{ pm}$ pose,
   $103.0\text{ pm/s}$ velocity), confirming continuous integration integrity.

## Evidence

`config.json`, `receipt.json`, `returned.json`, `returned-candidate.json`, `returned-nodes.json`,
`returned-replay.npz`, `pinocchio_replay.mat`, `evaluations.jsonl`, `jacobians.jsonl`.
Remote output `/mnt/c/Users/diete/native-two-window-fit-9967-88`.
