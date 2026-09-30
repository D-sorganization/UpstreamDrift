# From Tracked Match to Pure Forward Dynamics: Playbook and Roadmap

Epic [#10950](https://github.com/D-sorganization/UpstreamDrift/issues/10950).
Living document, started 2026-09-29. It records **everything learned about
matching quickly and accurately**, with evidence, and the path from today's
servo-tracked model to a pure forward-dynamics model that creates the motion.
Add to it whenever a run teaches something, including runs that failed.

## The Goal

A **pure forward-dynamics** GS3DX model is one where joint torques τ(t) are
played **open loop** from the start state:

- no PD feedback toward reference angles;
- no balance loop;
- no prescribed (motion-driven) joints, the neck included;
- the pelvis stays free on ground contact.

The result must reproduce the capture within stated bounds: joint angles,
pelvis path, club-head path and speed at impact, and ground reaction force.
The matching machinery that finds τ must also be **fast and repeatable** for
a new capture (`capture-O` and beyond). Convergence speed is a design goal,
not an afterthought.

## Where We Stand: The Ladder

| Rung | Model                                                                             | State                                                            |
| ---- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| L0   | Kinematic IK (`gs3dx_whole_body_ik`)                                              | Done; ROM penalty added (#11158, docs/ROM.md)                    |
| L1   | Forward dynamics + PD tracking + balance loop (`GS3DX_FitBalance`, `GS3DX_Human`) | Done; looks like a golf swing                                    |
| L2   | Learned feedforward carries the motion, PD gains annealed toward zero             | Upper body only; ILC diverges after iteration 2 (docs/FIT.md §6) |
| L3   | Open-loop torques, balance feedback removed                                       | Not started; balance-off replay tips 503 mm by impact            |
| L4   | Optimized open-loop torques (shooting / collocation), neck actuated               | Not started                                                      |

Each rung's residual control effort is the **distance to the goal**: the RMS
PD torque, the balance-loop command and the prescribed-joint torque. The
single number to drive to zero is the RMS of all feedback torque over the
swing, per joint and in total. Record it for every run.

## Lessons Learned (With Evidence)

### Reference and IK

| Lesson                                                                                                  | Evidence                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Joint centres alone admit mirror branches; bound the IK by human ROM                                    | Unpenalized IK: knees hyperextended 10–53° with the legs spun about their axes; spine side-bend 81° (docs/ROM.md)                                                                     |
| A universal joint has two branches, (a, b) and (a + 180, 180 − b); seed the trunk at zero               | Torso twisted 100° at address, 90–260° steps between frames without the posture pull (docs/FIT.md §3)                                                                                 |
| Track forward then backward and keep the lower cost per frame                                           | The backward pass repairs local minima of the forward pass (docs/FIT.md §3)                                                                                                           |
| Gap-filled targets must still anchor the pelvis                                                         | Pelvis wandered 15–36 cm where its markers dropped out (`gap_weight`)                                                                                                                 |
| Filter the reference (12 Hz) before differentiating                                                     | `gs3dx_upper_body_reference`                                                                                                                                                          |
| Neutral poses matter: a zero neck at address continues the trunk axis                                   | Head 126 mm off the capture's head markers (docs/HUMAN.md)                                                                                                                            |
| A drawn hand across the shaft leaves the wrist neutral undefined                                        | Hands 66° off the forearm axis (#11157)                                                                                                                                               |
| Never fit isolated frames: each needs a warm start from a near neighbour                                | Address, top and impact alone: 47 mm RMS unregularized, 79 mm regularized (160 mm at address); whole trial 6.3 mm                                                                     |
| A range penalty cannot rescue a bad seed; it trades marker error for range                              | Same three frames: ROM weight 1 gave 137 mm, weight 10 gave 334 mm, with the ranges nearly met (2026-09-29)                                                                           |
| Wrap a range check at ±180° only far from the range; the hinge jumps at the wrap                        | `mod` wrap about the neutral makes the residual discontinuous there; finite differences then stall                                                                                    |
| A penalty applied from the first frame derails the tracker; carry warm starts from an unpenalized chain | Every 10th frame to impact, warm-started: no penalty 23 mm mean; weight 0.3 86 mm, 1 39 mm (427 mm worst), 3 100 mm: not monotonic, lost frames seed their neighbours (2026-09-30)    |
| Apply a constraint penalty by continuation: fit free, then polish each frame from its free pose         | Weights 1, 3, 10 by continuation: 37.6, 37.7, 38.4 mm mean with 2.85°, 0.71°, 0.08° excess; worst frame 104-137 mm instead of 427-541 mm (docs/ROM.md)                                |
| Joint angle and marker offset are not jointly identifiable: the prior decides, so calibrate under it    | Lead elbow (#11156): free IK 51.8–65.9° flexion; raw capture markers 31–62°. Golf band 20° at weight 3 with free-calibrated offsets: held 18.5–41.7°, 43.2 mm vs 23.3 mm (2026-09-30) |

### Dynamics and Balance

| Lesson                                                                                | Evidence                                                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Joint tracking alone does not balance a free-standing body                            | Balance off: pelvis 206 mm RMS, 503 mm at impact (docs/FIT.md §6)                                                                                                                                                                     |
| Centre-of-mass feedback through the legs holds it                                     | Kp 3, Kd 0.4: pelvis 27 mm RMS (docs/FIT.md §7)                                                                                                                                                                                       |
| The reference must be dynamically consistent with the model's mass distribution       | A model-offset COM reference asked for a 2.63 BW impact spike; the capture's own COM gives 1.44 BW (docs/FIT.md §7)                                                                                                                   |
| Start from the reference state, not rest                                              | A 1.43 BW landing at 0.05 s from a zero-velocity start (docs/FIT.md §6)                                                                                                                                                               |
| Contact geometry decides stance stability                                             | Toe contacts 7 cm past the midfoot folded the foot; 15 mm short made the body drift 31–43 mm (docs/HUMAN.md)                                                                                                                          |
| Softer, human-like ankles break the current balance loop                              | Ankle 10 N·m/deg: pelvis 906 mm at impact (docs/FIT.md §7)                                                                                                                                                                            |
| The leg servo and the balance loop fight: the leg reference is not balance-consistent | `GS3DX_Human`, whole swing: leg servo 328 N·m RMS and balance share 314 N·m RMS, but their sum is only about 101 N·m RMS (roadmap step 1)                                                                                             |
| Joint ids differ between variants; identify joints by block path                      | `GS3DX_Human`'s neck renumbers every KinematicsSolver joint after it (`gs3dx_upper_body_joints` `.block`)                                                                                                                             |
| Summed leg-acceleration objectives fight; solve them together (seen in two engines)   | MuJoCo pipeline (#11166): adding root regulation (100, 20) took capture-A from 0.089 to 0.441 m, and the ZMP filter's CoM shift took it to 0.475 m. Simscape: leg servo and balance loop cancel (above)                               |
| A second golfer fails on reference feasibility, not on scaling                        | MuJoCo, capture-O: 0.282 m unscaled, 0.567 m scaled; capture-A scaled to the owner stays at 0.060 m. capture-O's reference ZMP is outside the support in 86 % of frames (capture-A 47 %), and the drift starts at downswing to impact |
| Peak accelerations do not predict trackability; the ZMP margin does                   | capture-A asks for 3x capture-O's root angular acceleration (4492 vs 1380 rad/s²) and tracks well                                                                                                                                     |

### Learning (ILC)

| Lesson                                             | Evidence                                                                 |
| -------------------------------------------------- | ------------------------------------------------------------------------ |
| One ILC update removes most of the angle error     | 0.96° to 0.25° RMS in one iteration (docs/FIT.md §6)                     |
| The PD torque then grows while the angles stay put | PD 10.3, 14.7, 20.7 N·m over iterations 2–4; torso holds 34 N·m at 0.19° |
| Filtering all of F (Q-filter) was not enough       | Same growth with the robust Q-filter form                                |

### Tooling

| Lesson                                                                                            | Evidence                                                                                                                                    |
| ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| One simulation of the swing is 17 min; iterating in Simscape alone is too slow                    | ILC timing (docs/FIT.md §6)                                                                                                                 |
| Signal logging passes the 1000-block Home licence limit; read the Simscape log instead            | `gs3dx_track_learn`; budget ≤ 975 compiled                                                                                                  |
| `-batch` MATLAB can hang after the script ends; always run through the lock runner's watchdog     | 2026-09-29 probes hung at exit; never wrap the runner in `timeout`                                                                          |
| End batch scripts with `quit(0, "force")`, not `exit(0)`                                          | Finished probes held the lock 6 and 15 min at `exit(0)`                                                                                     |
| Even `quit(0, "force")` can hang after the last line; watch the log and kill the job's own MATLAB | The contiguous-frame probe held the lock 12 min after its last save (2026-09-30)                                                            |
| Print a done marker last and let the runner kill the hung exit                                    | Scripts print `GS3DX_BATCH_DONE`; the lock runner kills its own MATLAB 90 s after it. Three probes in a row held the lock 10-15 min at exit |

## Why the Learning Drifts, and What to Do About It

The two arms and the club form a closed loop. Any torque pattern that only
squeezes or twists the loop moves no joint. The PD sees no angle error from
it, so ILC has no restoring signal, and noise in that null space integrates
over the iterations. The same applies to the spine and torso pair where they
act about nearly parallel axes. Two remedies:

- Project each learning update onto the torques that actually produce motion
  (the row space of the loop constraint). Equivalently, remove the loop's
  internal-force component.
- Let only one side of the loop learn (the lead arm), and leave the trail
  arm's loop joints to a weak PD.

Both are testable within the existing `gs3dx_track_learn` by recording the
per-joint PD torque over 6+ iterations. A forgetting factor
(`F ← λF + update`, λ ≈ 0.95) is the blunt fallback.

## Ideas for Fast Convergence

Ranked by expected payoff per effort. Each entry records the rationale and
the experiment that would test it.

1. **Inverse dynamics as the first guess, not ILC from zero.**
   - Drive every joint by motion (computed torque) along the reference and
     log the torques.
   - That is F in one simulation instead of eight 17-minute iterations.
   - The floating base needs the ground reaction split between the feet.
     Take it from the capture's kinematic GRF (`gs3dx_kinematic_grf`) and
     split it by a centre-of-pressure rule.
   - The pelvis residual wrench (the "hand of god") that remains measures
     how dynamically inconsistent the reference is.
2. **Make the reference dynamically consistent before learning** (residual
   reduction).
   - Adjust the pelvis and trunk path slightly, and if needed the trunk mass
     split, until the pelvis residual wrench vanishes for the model's own
     masses.
   - An inconsistent reference forces feedback forever; the 2.63 BW spike
     came from exactly that.
3. **Anneal the gains (homotopy).**
   - Run ILC while scaling the PD gains and the balance gains by α = 1, 0.5,
     0.25, … 0.
   - At each α the learned F must absorb what the feedback did. The
     trajectory never jumps far from a stabilized one.
   - Record the α at which the swing first fails. That α is the
     instability boundary to attack.
4. **Solve the forward-dynamics problem where gradients are cheap, then
   transfer.**
   - MuJoCo, and MJX on GPU, or Drake can solve a torque-driven
     multiple-shooting problem with analytic or automatic gradients in
     seconds per iteration. The same model can be exported there (epic
     #11161 builds those pipelines).
   - Transfer the torques to Simscape and finish with 1–2 ILC iterations.
5. **Shoot in segments.**
   - Tipping grows as e^(t/τ) with τ ≈ √(L/g) ≈ 0.3 s. Over the 1.3 s swing
     that is about 4τ, or ~50× error growth.
   - Split the swing into phases: address to top, top to mid-downswing,
     mid-downswing to impact.
   - Match states at the joins (multiple shooting). Each segment stays
     inside a couple of time constants, so the problem is well conditioned.
6. **Few parameters.**
   - Parameterize τ(t) by B-splines (8–12 knots per joint) or by swing phase.
   - Optionally use joint synergies (a PCA of the learned torques).
   - This cuts the search from ~40 × 470 samples to a few hundred numbers.
7. **Warm starts across golfers.**
   - Scale a converged τ(t) to a new golfer by mass × height (dynamic
     similarity: τ ∝ m g L, time ∝ √(L/g)).
   - Retime it to the new capture's event markers (address, top, impact
     from `swing_comparison.events`).
   - `capture-O` is the first test: the owner is 1.956 m and 104.3 kg.
8. **Cheaper simulations.**
   - Fast restart and rapid accelerator for the learning loop.
   - Softer contact with a larger step where the accuracy allows.
   - Measure the time per simulated second as part of every run.
9. **Actuate the neck.** It is the last prescribed joint. Give it the same
   feedforward + PD chart as the other upper-body joints so the ladder
   applies to it too.

## Roadmap (Each Step With Its Acceptance)

1. **Measure the distance to the goal.** `gs3dx_contact_check(...,
feedback=true)` returns `.feedback` (`gs3dx_feedback_torque`): the
   per-axis RMS and peak of the upper-body PD torque, the leg servo PD torque
   and the balance loop's share, and the total. The prescribed neck is not
   yet included. Accept: a table for `GS3DX_Human` to impact.
   **Done 2026-09-30** (`test_gs3dx_human/feedback_torque_is_measured_on_every_driven_axis`).
   The baseline for every later step, `GS3DX_Human` from `TrackStart` to 1.814 s
   (31 min wall):

   | Group                                | RMS (N·m) | Largest axis (RMS / peak N·m)        |
   | ------------------------------------ | --------- | ------------------------------------ |
   | Upper body PD (21 axes)              | 20.7      | Spine X 47.5 / 228; Torso 41.5 / 513 |
   | Leg servo PD (12 axes)               | 327.6     | R knee 471.9 / 1229                  |
   | Balance loop share (12 axes)         | 314.3     | R hip Y 460.0 / 1106                 |
   | Leg servo + balance, summed per axis | about 101 | (derived from the total)             |
   | Total, every axis and sample         | 62.8      |                                      |

   Readings. The upper body is already close to feedforward-ready: 20.7 N·m
   RMS, with the forearm rotations below 2 N·m; only the trunk (spine, torso
   peak 513 N·m at the downswing) needs real work. The legs are the gap: each
   loop alone carries 300+ N·m RMS, peaks above 1 kN·m at the knees and hips,
   and the two largely cancel. The servo pulls the legs toward a reference
   that the balance loop rejects, so step 4 (a dynamically consistent leg
   reference) removes both at once and should come before gain annealing of
   the legs.

2. **Inverse-dynamics feedforward (idea 1).** Accept: from ID feedforward,
   RMS PD torque below the ILC iteration-2 value (10.3 N·m), in one
   simulation.
3. **Loop-aware learning (the drift section).** Accept: PD torque falls
   monotonically over 6 iterations.
4. **Dynamically consistent reference (idea 2).** Accept: pelvis residual
   wrench below 5 % of body weight RMS; the impact support peak within
   0.2 BW of the capture's kinematic GRF.
5. **Gain annealing (idea 3).** Accept: a swing to impact at α ≤ 0.1 with
   the balance and match metrics within their current bounds.
6. **Open-loop replay (L3/L4) via segments or an external optimizer (ideas
   4–6).** Accept: α = 0, no balance loop, neck actuated, the match metrics
   within bounds.
7. **Second golfer (idea 7).** Accept: `capture-O` converges from the scaled
   warm start in at most half the iterations `capture-A` needed.

**Next step for the next agent:** step 4, before step 2. The step-1 table shows
the leg servo and the balance loop cancelling (about 300 N·m each, about 101 N·m
summed), so an inverse-dynamics feedforward on today's leg reference would
feed that conflict forward. Cross-engine evidence (#11166, MuJoCo): summed
leg-task objectives fight each other, so build the consistent reference as ONE
whole-body solve (for example a QP over the pelvis wrench and the foot contacts),
not as extra weighted tasks.
