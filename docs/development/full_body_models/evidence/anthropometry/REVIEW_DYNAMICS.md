# Anthropometry and Posture Review: Downswing Dynamics and Trajectory Optimisation

Sections 15 to 17 of [REVIEW.md](REVIEW.md), moved verbatim by #12802 to keep each document under the documentation size budget. Section numbers are unchanged.

## 15. Downswing Dynamics: Compliant Sole, Tracked Reference, Zero-Moment Point (MM-7, 2026-09-14)

GS-4 left the forward-dynamics replay diverging after the top: with the
driver match of section 14 the root error was 21 mm at 1.2 s, 40 at 1.4,
164 at 1.75 s, the body airborne twice (weight fraction 0 at 1.31 and from
1.43 s) and the peak joint torque 2523 N m. `evidence/ground_support/
downswing_experiment.py` replays only the dynamics stage of a finished run
with one setting changed at a time and writes `<run>/downswing_<name>.json`;
`--free-wrists` runs are not needed since the reference is the run's own.

### What Was Ruled Out

- **Reference jerk.** The consistency re-solve steps 20 mm in one frame
  where markers drop out at impact (root acceleration 492 m/s^2, 218 000
  deg/s^2 in the joints). A second zero-phase low-pass of the tracked
  reference at 12 or 8 Hz removes the impact torque spike (2523 to 811 N m)
  but not the divergence (212 / 188 mm). Dropping the acceleration
  feedforward: 156 mm.
- **Friction creep.** The regularised Coulomb law (transition velocity
  50 mm/s) was suspected because the pelvis yaw lags the reference by 11 to
  13 deg through the transition. Transition velocities of 10, 5 and 2 mm/s
  (time steps down to 0.1 ms) leave the timeline unchanged (7/13/24/24/38
  mm at 1.0/1.1/1.2/1.3/1.4 s); friction coefficients 1.5, 2.0 and 3.0
  (spiked shoes) change nothing before impact either. The feet were not
  creeping: the lead foot's forefoot and toe spheres sit 1 to 2 mm above the
  plane from 1.0 s (the sim stands on its heels), so a root pitch error
  below one degree unloads a foot, and an unloaded foot slides freely
  (lead toe 35 mm by 1.2 s, 121 mm by 1.35 s) whatever the coefficient.
- **Root regulation through the legs.** Gains (100, 20) with legs at
  10 rad/s: 147 mm; (400, 40) and (900, 60): 1.0 m and 0.9 m with
  weight fractions of 100 to 200 (the planted-coupling pseudo-inverse
  amplifies once a foot is off the plane).

### What the Reference Demands

`full_body_simulation.reference_zmp` computes, from the whole-body momentum
rates of the tracked reference, the ground reaction and the zero-moment
point this model would need, against the hull of the spheres on the plane
at each frame. For the driver reference: vertical reaction 0.61 BW with
0.69 BW horizontal at 1.15 s (ratio 1.12, above any shoe), 1.69 BW with
0.82 BW horizontal at 1.3 s; the zero-moment point leaves the support
polygon on 77 % of the frames between 1.0 and 1.5 s, by
tens of centimetres. No controller can realise those frames on unilateral
feet. The tour-average capture is a composite of many swings, and averaged
kinematics are not dynamically consistent for any one body; the model's
de Leva mass distribution differs from the golfers' as well. The receipts
now carry this diagnostic (`dynamics.reference_zmp`).

### What Works: A Compliant Sole and a Band-Limited Tracked Reference

Contact stiffness (N/m), dissipation (s/m) and the tracked reference's
low-pass were varied on the driver run (columns: root error at 1.2 / 1.4 s
and at the end, maximum, weight fraction range, peak torque):

| name               | k (kN/m) | d   | mu  | v_t   | low-pass Hz | ff  | legs | root reg      | 1.2 s | 1.4 s | end  | max  | wf min | wf max | torque N m |
| ------------------ | -------- | --- | --- | ----- | ----------- | --- | ---- | ------------- | ----- | ----- | ---- | ---- | ------ | ------ | ---------- |
| base               | 200      | 1   | 0.9 | 0.05  | none        | 1.0 | 30   | off           | 21    | 40    | 164  | 178  | 0.0    | 5.59   | 2523       |
| ff0                | 200      | 1   | 0.9 | 0.05  | none        | 0.0 | 30   | off           | 27    | 29    | 144  | 156  | 0.0    | 5.36   | 1541       |
| k100               | 100      | 1   | 0.9 | 0.05  | none        | 1.0 | 30   | off           | 23    | 32    | 124  | 128  | 0.0    | 6.15   | 2777       |
| k25_d2             | 25       | 2.0 | 0.9 | 0.05  | none        | 1.0 | 30   | off           | 44    | 50    | 155  | 169  | 0.11   | 4.99   | 2615       |
| k40                | 40       | 1   | 0.9 | 0.05  | none        | 1.0 | 30   | off           | 11    | 39    | 90   | 93   | 0.0    | 5.43   | 2701       |
| k50                | 50       | 1   | 0.9 | 0.05  | none        | 1.0 | 30   | off           | 21    | 42    | 56   | 57   | 0.11   | 5.56   | 2684       |
| k50_d05            | 50       | 0.5 | 0.9 | 0.05  | none        | 1.0 | 30   | off           | 21    | 47    | 81   | 82   | 0.0    | 5.21   | 2642       |
| k50_d2             | 50       | 2.0 | 0.9 | 0.05  | none        | 1.0 | 30   | off           | 21    | 36    | 38   | 45   | 0.28   | 6.21   | 2693       |
| k50_d2_lp12        | 50       | 2.0 | 0.9 | 0.05  | 12.0        | 1.0 | 30   | off           | 23    | 38    | 33   | 43   | 0.38   | 2.37   | 476        |
| k50_d2_lp12_dt05   | 50       | 2.0 | 0.9 | 0.05  | 12.0        | 1.0 | 30   | off           | 23    | 38    | 33   | 43   | 0.38   | 2.37   | 473        |
| k50_d3             | 50       | 3.0 | 0.9 | 0.05  | none        | 1.0 | 30   | off           | 21    | 34    | 37   | 48   | 0.0    | 7.01   | 2766       |
| k50_dt05           | 50       | 1   | 0.9 | 0.05  | none        | 1.0 | 30   | off           | 21    | 42    | 56   | 57   | 0.1    | 5.56   | 2375       |
| k50_lp12           | 50       | 1   | 0.9 | 0.05  | 12.0        | 1.0 | 30   | off           | 23    | 41    | 51   | 53   | 0.08   | 2.73   | 508        |
| k50_mu15           | 50       | 1   | 1.5 | 0.05  | none        | 1.0 | 30   | off           | 23    | 47    | 85   | 87   | 0.04   | 6.15   | 3395       |
| k60                | 60       | 1   | 0.9 | 0.05  | none        | 1.0 | 30   | off           | 26    | 38    | 46   | 52   | 0.08   | 5.48   | 2649       |
| k60_d2             | 60       | 2.0 | 0.9 | 0.05  | none        | 1.0 | 30   | off           | 26    | 36    | 43   | 54   | 0.12   | 6.23   | 2685       |
| lp12               | 200      | 1   | 0.9 | 0.05  | 12.0        | 1.0 | 30   | off           | 22    | 26    | 183  | 212  | 0.0    | 2.98   | 811        |
| lp12_ff0           | 200      | 1   | 0.9 | 0.05  | 12.0        | 0.0 | 30   | off           | 25    | 37    | 169  | 195  | 0.0    | 3.1    | 828        |
| lp12_legs10        | 200      | 1   | 0.9 | 0.05  | 12.0        | 1.0 | 10.0 | off           | 20    | 73    | 47   | 78   | 0.0    | 4.6    | 758        |
| lp12_vt010_dt05    | 200      | 1   | 0.9 | 0.01  | 12.0        | 1.0 | 30   | off           | 24    | 24    | 176  | 207  | 0.0    | 2.91   | 729        |
| lp8                | 200      | 1   | 0.9 | 0.05  | 8.0         | 1.0 | 30   | off           | 22    | 51    | 151  | 188  | 0.0    | 3.08   | 783        |
| mu15               | 200      | 1   | 1.5 | 0.05  | none        | 1.0 | 30   | off           | 23    | 22    | 135  | 154  | 0.0    | 6.05   | 3393       |
| mu15_lp12          | 200      | 1   | 1.5 | 0.05  | 12.0        | 1.0 | 30   | off           | 23    | 17    | 183  | 217  | 0.0    | 3.33   | 1059       |
| mu20               | 200      | 1   | 2.0 | 0.05  | none        | 1.0 | 30   | off           | 24    | 15    | 126  | 145  | 0.0    | 5.59   | 3887       |
| mu30               | 200      | 1   | 3.0 | 0.05  | none        | 1.0 | 30   | off           | 31    | 18    | 107  | 129  | 0.0    | 5.12   | 4751       |
| rootreg_legs10     | 200      | 1   | 0.9 | 0.05  | none        | 1.0 | 10.0 | (100.0, 20.0) | 24    | 33    | 138  | 147  | 0.0    | 9.38   | 3691       |
| rr400_legs10       | 200      | 1   | 0.9 | 0.05  | none        | 1.0 | 10.0 | (400.0, 40.0) | 48    | 575   | 981  | 1032 | 0.0    | 196.77 | 71607      |
| rr400_legs10_nobal | 200      | 1   | 0.9 | 0.05  | none        | 1.0 | 10.0 | (400.0, 40.0) | 176   | 589   | 1101 | 1859 | 0.0    | 131.43 | 26948      |
| rr900_legs8_mu15   | 200      | 1   | 1.5 | 0.05  | none        | 1.0 | 8.0  | (900.0, 60.0) | 66    | 423   | 585  | 908  | 0.0    | 112.76 | 43585      |
| vt002_dt01         | 200      | 1   | 0.9 | 0.002 | none        | 1.0 | 30   | off           | 24    | 37    | 84   | 84   | 0.0    | 5.28   | 2653       |
| vt005_dt025        | 200      | 1   | 0.9 | 0.005 | none        | 1.0 | 30   | off           | 24    | 38    | 148  | 157  | 0.0    | 5.26   | 2565       |
| vt010_dt05         | 200      | 1   | 0.9 | 0.01  | none        | 1.0 | 30   | off           | 23    | 39    | 151  | 159  | 0.0    | 5.16   | 2329       |

Softening the sole from 200 to 50 kN/m (a 78 kg golfer sinks 16 mm at
rest instead of 4; shoe midsole plus turf) lets the body give the few
centimetres the composite reference demands instead of unloading a foot:
root error 57 mm maximum, never airborne. Dissipation 2 s/m damps the
impact rebound (45 mm), and the 12 Hz low-pass of the tracked reference
takes the peak torque from 2693 to 476 N m without costing tracking
(43 mm). Halving the time step reproduces the same numbers, so the result
is not an integration artefact. 25 kN/m sinks too far (169 mm), 100 kN/m
still hops (128 mm).

Adopted: `build_anthropometric_spec.py` writes 5.0e4 N/m and 2 s/m into the
anthropometric documents (`CONTACT_STIFFNESS_N_M`,
`CONTACT_DISSIPATION_S_M`; the qualified native v2 keeps 2.0e5, and the
driver's `add_toe_spheres` raises only native documents), and the driver
tracks the re-solved reference through one more 12 Hz zero-phase low-pass
(`TRACKING_CUTOFF_HZ`). Full-pipeline receipts: driver dynamics root error
38 mm, whole-run marker RMS 74.6 mm (was 77.3),
weight fraction 0.38 to 2.37, peak torque 476 N m; 7-iron root error
31 (to 1.5 s; 133 at 1.75 s in the follow-through) mm, marker RMS 112.3 mm, weight fraction 0.29 to 3.65, peak
torque 835 N m. The softer sole costs a little backswing sway
(driver root error to 1.0 s 20 mm, was 7 mm). Setup parity is
untouched (the contact law is in the shared document): Drake 5.9e-06 m
driver, 5.4e-06 m 7-iron.

### Cart-Table Dynamics Filter (MM-7B Experiment, Not Adopted)

`dynamics_filter.py` (model-free: `project_inside`, `cart_table_shift`)
and the driver's `--zmp-filter` implement the Kagami-style correction:
project the reference zero-moment point into the per-frame support polygon
shrunk by 2 cm, find the smallest smooth centre-of-mass shift whose
cart-table effect closes the gap, re-solve the IK against the markers with
that centre-of-mass path as per-frame rows (`solve_pose(com_target=...)`,
`solve_trajectory(com_targets_per_frame=...)`), low-pass, repeat. On the
driver (`ground_support/anthro_driver_zmp/receipt.json`) three passes take
the zero-moment point outside the polygon from 77 % to 62 %, 57 % and 57 %
of the downswing frames while the centre-of-mass shift grows to 112, 160
and 278 mm, the reference's marker error rises from 27 to 38, 55 and 98 mm,
and the replay is worse than without the filter (root error 129 mm at
1.4 s, 497 mm at 1.75 s; 7-iron 337 mm at 1.4 s, `anthro_iron_zmp/receipt.json`). Verdict: the single-mass cart-table relation
cannot describe this reference. Its zero-moment-point excursions are driven
by the angular momentum of the arms and club (the term the cart table
ignores), so the centre of mass would have to move tens of centimetres to
compensate, which the markers forbid. A consistent motion needs the
upper-body swing itself to change, not the centre-of-mass path. The flag
stays available as an experiment with its receipt.

### What Remains, Next Step MM-7B

The reference is still not dynamically consistent for this model (zero-moment
point outside on 77 % of the downswing frames), so the replay follows it to
within 3 to 4 cm at the pelvis rather than exactly. The next step is a
whole-body formulation rather than a centre-of-mass filter: the FB-5
contact-aware shooting fit (optimise the tracked reference of every
coordinate under the replay's own contact dynamics so that the marker error
of the replayed motion is minimised, with the zero-moment point as a
penalty), or an equivalent trajectory optimisation over the angular
momentum of the arm-club system. `reference_zmp` and the downswing
experiment harness provide the metrics; the 38 mm root error of the
compliant-sole replay is the baseline to beat.

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
