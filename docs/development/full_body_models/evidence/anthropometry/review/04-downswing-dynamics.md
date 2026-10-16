# Anthropometry Review of Downswing Dynamics

Section 15 of the [anthropometry review](../REVIEW.md). Moved verbatim.

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
