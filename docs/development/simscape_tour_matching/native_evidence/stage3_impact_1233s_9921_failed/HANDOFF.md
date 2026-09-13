# Overnight Simscape Downswing/Impact Extensions (1.15 s, 1.233 s) — Failed, Archived

Archived 2026-09-13 from the DeskComputer runtime worktree
(`C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-tour-runtime/scratch`)
and the local `scratch/` directory of this worktree. These were run overnight
2026-09-12/13 by an Antigravity session (id 9efc50d9) on epic #9921. None of
the runs was committed, none produced an accepted candidate, and the final run
crashed before writing its report. This folder preserves the exact scripts,
launchers, logs, checkpoints and packages as failed experiments. No processes
remain active on DeskComputer (verified 2026-09-13; MATLAB PID 80184 has
exited).

## Runs and Measured Results (Uninterrupted R2025b Rollouts, Zero Resets)

| Run                             | Horizon | Evals | Whole RMS | Early ≤0.6 s | Terminal |                 Clubhead terminal | Pelvis yaw error | Result                             |
| ------------------------------- | ------: | ----: | --------: | -----------: | -------: | --------------------------------: | ---------------: | ---------------------------------- |
| downswing 1.15 s locked (local) |  1.15 s |     — |  340.1 mm |     11.27 mm | 465.2 mm |                          721.8 mm |   38.0 % @1.15 s | 1/6 gates; converged; not accepted |
| staged impact 1.233 s (Desk)    | 1.233 s |   201 |  379.6 mm |     11.27 mm | 658.1 mm |                          684.7 mm |  58.7 % @1.233 s | best clubhead 480.4 mm at eval 201 |
| stage 3 unified impact 1.233 s  | 1.233 s |   450 |         — |     11.27 mm |        — | 647.1 mm (best 646.2 at eval 192) |           58.1 % | `xtol` termination, then crashed   |

The crash is a `NameError: yaw_t` in the final gate report of
`run_stage3_impact_1233s.py` (line 418: the target yaw was a local of the
residual function). `run_stage3_impact_1233s_repaired.py` beside this file
recomputes the target yaw in the report block; it is otherwise byte-identical
to the executed script and has also been placed in the DeskComputer scratch
directory. Nothing from the crashed run was lost: the best checkpoint JSON is
archived, but it is a checkpoint, not a returned package.

Degrees 0 to 3 were frozen on all 27 channels in every run (78 to 81 free
controls). Early retention therefore never moves, and the downswing has only
three controls per channel to shape 0.4 s of motion. Errors of 340 to 720 mm
against 18/25/30 mm gates are a basis-authority limit, not a budget problem;
the native lane hits the same wall at 0.85 s with the same freeze. Do not
resume these horizon pushes without changing what is free.

The overnight walkthrough described Stage 2 as a breakthrough (clubhead
1352 → 480 mm). Read that as: still 16 times the gate, and Stage 3 then ended
worse (647 mm). No claim in that walkthrough should enter a handoff unqualified.

## Files

`deskcomputer/`: `stage3_impact_1233s.log`, `stage3_impact_best_checkpoint.json`,
`launch_stage3_impact_1233s.bat`, `run_stage3_impact_1233s.py` (executed
bytes, SHA256 00b212dd…), `staged_impact_1233s.log`, `staged_impact_best_checkpoint.json`,
`candidate_staged_impact_1233s_package.json`, `run_staged_impact_1233s.py`,
`audit_remote_staged.py`. `local_scratch/`: the 1.15 s and 1.233 s locked
runners, launchers, `impact_1233s_locked.log` (check-only pass), the 1.15 s
result and eval-0 JSON, and the basis-authority analysis scripts. The 1.15 s
package is committed at `candidates/candidate_downswing_115s_locked_package.json`.
The 1.05 s package remains at commit 4b1365d79.

## What This Lane Should Do Next

The Simscape lane is the R2025b reference, not a second optimizer. Its next
bounded task is to replay the native Pinocchio returned81 polynomial (exact
coefficients, absolute clock, original initial state; see the native lane's
`native_evidence/two_window_fit_9967_81/returned-candidate.json`) through the
existing qualified replay tooling and report same-input parity through 0.85 s
on the five shared metrics (whole, early, terminal, clubhead, pelvis yaw).
