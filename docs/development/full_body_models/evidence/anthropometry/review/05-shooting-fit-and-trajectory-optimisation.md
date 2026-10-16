# Anthropometry Review of Shooting Fit and Trajectory Optimisation

Sections 16 to 17 of the [anthropometry review](../REVIEW.md). Moved verbatim.

## 16. Contact-Aware Shooting Fit (FB-5, MM-7B, 2026-09-14)

The replay is the plant. `run_ground_support.py --shooting-fit N`
(`shooting_fit`, `replay`) iterates: replay the tracked reference with the
computed-torque controller on the compliant sole, measure the pelvis drift
of the replay from the reference pelvis path, move the pinned pelvis
command against that drift (iterative learning,
`command <- command - 0.7 (replayed - reference)` on the two horizontal
slides and the three root rotations; the vertical slide stays free so the
planted feet set it), re-solve the joints against the markers with the
command pinned (`solve_trajectory(locked_per_frame=...)`, consistency prior,
stance planted, human bounds), low-pass at 12 Hz, replay again. The
objective is the marker error of the replayed motion; the best iteration's
reference is what the dynamics stage then tracks, and every iteration is in
the receipt (`dynamics.shooting_fit`).

A plain fixed point (pin the pelvis where the replay drifted and re-solve)
was tried first and is anti-corrective: driver replays 74.6, 73.9, 90, 199,
266 mm over four iterations, 7-iron 112, 110, 151, 256, 335 mm. The
iterative-learning form:

| Capture            | Iteration | Replay markers | Root max | Root at 1.4 s | Weight fraction min | ZMP outside 1.0 to 1.5 s |
| ------------------ | --------- | -------------- | -------- | ------------- | ------------------- | ------------------------ |
| driver (gain 0.7)  | 0         | 74.6 mm        | 43 mm    | 38 mm         | 0.38                | 77 %                     |
| driver (gain 0.7)  | 1         | 103.9 mm       | 66 mm    | 39 mm         | 0.35                | 83 %                     |
| driver (gain 0.7)  | 2         | 126.6 mm       | 71 mm    | 62 mm         | 0.32                | 82 %                     |
| driver (gain 0.7)  | 3         | 130.3 mm       | 91 mm    | 72 mm         | 0.29                | 86 %                     |
| driver (gain 0.7)  | 4         | 134.8 mm       | 120 mm   | 68 mm         | 0.31                | 81 %                     |
| 7-iron (gain 0.7)  | 0         | 112.3 mm       | 165 mm   | 31 mm         | 0.29                | 76 %                     |
| 7-iron (gain 0.7)  | 1         | 128.8 mm       | 112 mm   | 27 mm         | 0.36                | 79 %                     |
| 7-iron (gain 0.7)  | 2         | 146.1 mm       | 138 mm   | 36 mm         | 0.32                | 81 %                     |
| 7-iron (gain 0.7)  | 3         | 174.5 mm       | 152 mm   | 38 mm         | 0.20                | 84 %                     |
| 7-iron (gain 0.7)  | 4         | 192.9 mm       | 136 mm   | 31 mm         | 0.06                | 87 %                     |
| driver (gain 0.25) | 0         | 74.6 mm        | 43 mm    | 38 mm         | 0.38                | 77 %                     |
| driver (gain 0.25) | 1         | 81.3 mm        | 43 mm    | 36 mm         | 0.37                | 81 %                     |
| driver (gain 0.25) | 2         | 87.4 mm        | 47 mm    | 36 mm         | 0.36                | 81 %                     |
| driver (gain 0.25) | 3         | 92.6 mm        | 48 mm    | 38 mm         | 0.36                | 80 %                     |
| driver (gain 0.25) | 4         | 96.9 mm        | 51 mm    | 40 mm         | 0.35                | 78 %                     |
| driver (gain 0.25) | 5         | 100.6 mm       | 54 mm    | 43 mm         | 0.35                | 79 %                     |
| driver (gain 0.25) | 6         | 103.9 mm       | 56 mm    | 45 mm         | 0.35                | 80 %                     |

Verdict: every gain diverges monotonically from iteration 0 on both captures (gain 0.7: driver 74.6 to 134.8 mm, 7-iron 112.3 to 192.9 mm; gain 0.25: driver 74.6 to 103.9 mm over six passes), so the best reference is always the unmodified one and the dynamics stage keeps it. The attribution receipted with the soft-sole replay explains why: substituting only the replayed pelvis rotation into the reference already gives 72.7 of the 74.6 mm replay error (translation alone 35.3 mm, the joints with the reference pelvis 34.6 mm); the pelvis yaw lags the reference by 15 deg through the downswing and 34 deg in the follow-through, and that lag is what the ground can supply in yaw moment for this reference, not something a displaced pelvis command can pre-compensate (a command displaced against the drift costs marker fit in the reference immediately and the replay does not recover it). Friction on the compliant sole is not the lever either (transition velocity 10 mm/s: identical; coefficient 1.5: yaw 14.9 to 9.1 deg at 1.3 s but 78.6 mm overall). The shooting fit therefore needs the whole-body swing as the decision variable with the contact dynamics as constraints: a trajectory optimisation with a differentiable simulator (the JaxSim differentiable backend, epic #6647, or MuJoCo MJX) over the arm-club angular momentum and pelvis rotation, where the composite reference is a soft target and the ground yaw-moment capacity is respected. The iterative-learning harness, `reference_zmp` and the downswing experiment harness stay as the evaluation tools; `--shooting-fit` remains available with its receipts (`anthro_driver_shoot`, `anthro_iron_shoot`, `anthro_driver_shoot_g025`).

## 17. Differentiable Trajectory Optimisation With MuJoCo MJX (FB-5, MM-7B, 2026-09-14)

The whole tracked reference is now optimised through the contact dynamics
with gradients. `export_mjx_package.py` writes a finished run as a
self-contained package (the MJCF with its grip site weld stiffened, the
tracked reference, the capture markers and validity, the marker attachments
in MJCF body frames, spheres, contact law, ground, controller gains);
`mjx_trajectory_optimisation.py` runs in an isolated environment (JAX
0.11, MuJoCo 3.13 with MJX; recipe in HANDOFF) and is a JAX port of the
replay: the shared Hunt-Crossley plus regularised-Coulomb sphere law applied
as world wrenches at the calcanei, the dual-grip weld as a stiff
spring-damper wrench (MJX's constraint solver is an iterative loop JAX
cannot reverse-differentiate, so every equality is removed from the model),
the root-free computed-torque controller, semi-implicit Euler at six
substeps per capture frame, a `lax.scan` over frames with checkpointing that
emits the replayed marker positions. Decision variables: knot values every
40 ms of a correction added to the 38 actuated coordinates (47 knots,
1786 parameters); cost: mean squared replayed marker error over valid
markers up to 1.65 s (the soft-weld plant departs from the rigid-weld one in
the last follow-through) plus a small regulariser; Adam on the gradient
through the whole rollout (about 52 s per iteration on CPU after a
697 s compile).

Port and pitfalls (all receipted in `anthro_driver/mjx_*`): the
near-massless hand standoff explodes on a spring weld without a 5e-3
kg m^2 armature floor on every dof; the previous substep's applied torque
sits in `qfrc_smooth` and must be cleared before the controller's forward
pass; a vector norm at zero tangential speed and the NaN of invalid markers
under a masked `where` both poison the gradient. With those fixed the MJX
replay of the unmodified reference is 41.0 mm to 1.65 s against the
shared simulator's 74.6 mm over the whole swing, with the same downswing
profile. A learning rate of 2e-3 rad per knot makes the objective wander
(41 to 107 mm between iterations; `mjx_opt_lr2e-3_log.txt`); 5e-4 descends
monotonically.

| Iteration (lr 5e-4) | MJX replay markers to 1.65 s |
| ------------------- | ---------------------------- | --------------------- | ------------------------ | -------------------------- |
| Iteration           | MJX replay to 1.45 s         | shared plant to 1.5 s | shared plant whole swing | pelvis yaw lag 1.3 / 1.4 s |
| ---                 | ---                          | ---                   | ---                      | ---                        |
| 0 (unmodified)      | 41.0 mm                      | 56.1 mm               | 74.6 mm                  | 14.9 / 12.3 deg            |
| 2                   | 34.2 mm                      | 48.0 mm               | 78.1 mm                  | 12.2 / 8.2 deg             |
| 4                   | 32.0 mm                      | 42.4 mm               | 79.6 mm                  | 10.6 / 5.4 deg             |
| 6                   | 30.5 mm                      | 42.8 mm               | 83.3 mm                  | 10.3 / 4.3 deg             |
| 8                   | 29.5 mm                      | 44.6 mm               | 195.5 mm                 | 10.4 / 4.0 deg             |
| 10                  | 28.8 mm                      | 43.7 mm               | 192.4 mm                 | 10.6 / 3.8 deg             |
| 12                  | 28.4 mm                      | 42.1 mm               | 190.0 mm                 | 10.3 / 3.4 deg             |
| 14                  | 27.9 mm                      | 41.6 mm               | 184.9 mm                 | 9.9 / 3.1 deg              |
| 16                  | 27.5 mm                      | 40.2 mm               | 182.3 mm                 | 9.4 / 2.6 deg              |

Validation in the shared-law simulator (`downswing_experiment.py
--reference`, the real plant with the rigid weld and RK4): every snapshot replayed in the shared-law simulator (`downswing_mjx_h145_iter*.json`), columns above.

Verdict: the differentiable optimisation works and transfers within its cost window. The replayed marker error to 1.5 s falls from 56.1 to 40.2 mm (28 %) and the pelvis yaw lag at 1.4 s from 12.3 to 2.6 deg with corrections below half a degree at the knots, which is the first method in this lane to move the pelvis yaw. It does not yet transfer to the whole swing: the follow-through after 1.5 s is uncosted (the soft-weld MJX plant departs from the rigid-weld one there, 187 and 322 mm at 1.7 and 1.8 s even at 12 substeps) and from iteration 8 the shared plant collapses after impact (whole-swing 182 to 195 mm), while iterations 4 to 6 keep it near the baseline (79.6 and 83.3 mm against 74.6). The transferable choice today is therefore iteration 4 to 6 for a whole-swing replay or iteration 16 for the swing to 1.5 s. Next: give the MJX plant a faithful follow-through. A five-times stiffer, more damped weld leaves the late window numerically identical (`mjx_diag_w1e6.txt`), so the weld is not the cause; the divergence after 1.6 s (small pelvis error, large marker error) has to be located joint by joint against the shared plant's late replay before a full-horizon solve, then the 7-iron follows. A joint-by-joint comparison of the two plants' replays of the unmodified reference (`mjx_diagnose.npz` against `downswing_soft_base.npz`) shows every joint tracked equally well; the difference is the pelvis rotation: MJX lags 7.6 to 8.7 deg through the downswing where the shared plant lags 14.9 to 15.1, and 50.6 deg in the follow-through (1.65 to 1.82 s) where the shared plant lags 34.4. The two plants differ in their ground yaw-moment response (contact evaluated at the substep start with semi-implicit Euler against RK4), which is where the next agent should look. The receipts (`mjx_optimisation_receipt.json`, `mjx_opt_*_log.txt`, the snapshot replays) and the two scripts are the harness for it.
