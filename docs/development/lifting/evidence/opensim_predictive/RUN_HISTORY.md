# OSL-7 Predictive Deadlift Run History (#13044)

Every ControlTower stage-1 (torque-driven) run of
`scripts/measure_opensim_predictive_deadlift.py`, 15 Hermite–Simpson mesh
intervals, forward finite differences and at most 3000 IPOPT iterations unless
noted. Values are copied from each run's `receipts.json`; runs without a
receipt say why. The generated [`README.md`](README.md) and
[`receipts.json`](receipts.json) in this folder are run R.

"Residual ratio" is the peak pelvis residual force divided by the lifter plus
bar weight (1765.2 N). The Hicks 2015 credibility limit is 0.05. "Violation" is
the scaled constraint violation of the final iterate; the solver tolerance
is 0.01.

| Run | Commit     | Change Under Test                                      | Status                                                                | Iterations | Objective | Violation | Residual Ratio |
| :-- | :--------- | :----------------------------------------------------- | :-------------------------------------------------------------------- | ---------: | --------: | --------: | -------------: |
| C   | f23d09a13e | Explicit dynamics, forward differences                 | Max iterations                                                        |       3000 |     0.996 |    0.0107 |          1.050 |
| D   | —          | (no receipt; no IPOPT exit line in the fetched logs)   | —                                                                     |          — |         — |         — |              — |
| E   | 0c81f1017e | Warm start from C                                      | Max iterations                                                        |       3000 |     0.835 |     0.628 |          0.852 |
| F   | —          | (no receipt; no IPOPT exit line in the fetched logs)   | —                                                                     |          — |         — |         — |              — |
| G   | —          | Six pelvis residuals                                   | IPOPT optimal; receipt crashed (`MocoProblemRep` has no `initSystem`) |          — |         — |         — |              — |
| H   | 353d42ed0f | Post-convergence path fixed                            | **Solve succeeded**                                                   |        646 |     0.187 |    2.2e-8 |          0.326 |
| I   | 353d42ed0f | Warm start from H                                      | Max iterations                                                        |       3000 |     5.992 |    1.8e-4 |          0.143 |
| J   | 353d42ed0f | Warm start from H                                      | Max iterations                                                        |       3000 |    36.511 |    1.7e-3 |          0.140 |
| K   | 96ff3d5b82 | Residual 1 mg, weight 100; duration fixed 2.0 s        | Max iterations                                                        |       3000 |   104.326 |    0.0995 |          0.695 |
| L   | 96ff3d5b82 | Residual 1 mg, weight 100; duration 1.5–3.0 s          | Max iterations                                                        |       3000 |    69.681 |    0.0038 |          0.584 |
| M   | —          | Earlier pose candidate                                 | IPOPT optimal; job cleaned before the receipt was written             |          — |         — |         — |              — |
| N   | 256cc9428b | Solved poses, free shoulders; residual 2 mg, weight 10 | Max iterations                                                        |       3000 |     7.685 |    0.0641 |          1.041 |
| O   | 256cc9428b | As N, duration fixed 2.0 s                             | Max iterations                                                        |       3000 |     5.862 |    0.0258 |          1.090 |
| P   | 9dca083f4f | Planted-feet goal; residual 1 mg, weight 100           | Max iterations                                                        |       3000 |   146.192 |     0.595 |          0.820 |
| Q   | 9dca083f4f | As P, duration fixed 2.0 s                             | Max iterations                                                        |       3000 |  1442.211 |      6.18 |          0.840 |
| R   | 9dca083f4f | As P, implicit multibody dynamics                      | Max iterations                                                        |       3000 |   216.199 |     0.301 |          0.923 |
| S   | 9dca083f4f | As P, warm start from P                                | Max iterations                                                        |       3000 |   484.500 |      30.6 |          0.746 |

## Findings

1. **Run H is not a credible deadlift.** It converged on the hand-built poses
   that preceded `compute_deadlift_poses`: duration 0.553 s against the bundle's
   2.0 s, bar rise 0.409 m against 0.477 m, and a pelvis residual carrying a
   third of the weight.
2. **The hand-built poses were infeasible.** With the shoulders welded, no
   setup pose put the bar over midfoot with the trunk at the bundle angle (pose
   residual 0.064 m). `compute_deadlift_poses` now solves both poses with free
   shoulder flexion, flat feet and the bar over midfoot (commit d2b3f2a067).
3. **Without a foot goal the optimizer floats the lifter.** Replaying run N's
   solution at 11 sampled nodes, the lowest foot sphere rises up to 56.7 mm and some spheres reach
   332.6 mm, while the pelvis vertical residual carries 1231–2356 N. The
   planted-feet goal (commit 9dca083f4f) holds both calcanei at their setup
   positions. Replaying run P, the lowest sphere stays within −9.4 to +0.2 mm.
   The feet still rock, with some spheres reaching 106.6 mm, and the residual
   still carries 491–1765 N.
4. **Stage 1 does not converge with foot contact in this formulation.** Explicit
   and implicit dynamics, fixed and free duration, and warm starts all stop at
   3000 iterations with violations from 0.03 to 30.6. Stage 2 (muscles) never
   ran.

## Open Options (Owner Decision)

- Replace the foot contact spheres with welded feet for the predictive solve and
  report the weld reaction as the ground reaction. A deadlift keeps the feet
  fixed, but this removes contact from the prediction.
- Keep contact and refine the mesh (30–50 intervals) with a contact-free
  warm start. Each run then costs several hours on ControlTower.
- Keep contact and tune the sphere stiffness and dissipation for a 180 kg
  load. The calibrated foot spheres sank 0.119 m before OSL-12.

Diagnostic replays (`contact_replay_n.log`, `contact_replay_p.log`) come from a
scratch script that sets each solution node's coordinates and reads the foot
sphere ground points with `msk_lift_model._ground_points`.
