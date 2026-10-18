# Bench Press Same-Input Bundle (LIFT-5, #11745)

`bench_press_bundle.npz` packages a flat barbell bench press as a same-input
`InputBundle`, built by the back-squat pipeline (see
`../back_squat/README.md`) with the recipe in
`src/shared/python/lifting/lift_recipes.py`. The lifter lies supine on a
bench contact body.

## How It Is Built

1. **Bench contact body.** The shared Hunt-Crossley law
   (`src/shared/python/motion_matching/contact_law.py`) accepts raised
   contact supports: `contact.supports` names spheres that press on a plane
   parallel to the ground at the support's height (`support_planes`). The
   MuJoCo, Drake, Pinocchio and OpenSim same-input adapters all resolve the
   plane per sphere through that one helper, so the law is not duplicated.
   `bench_spec` (`src/shared/python/lifting/bench_press.py`) adds four back
   spheres that mirror OSL-2's `msk_lift_model._BACK_SPHERES`: posterior
   shoulder (RPSH, LPSH → spec markers `BackRight`, `BackLeft` on the spine
   frame) and posterior iliac (RPSI, LPSI → `WaistRBack`, `WaistLBack` on the
   pelvis). Each sphere has radius 0.04 m × stature / 1.70 m (0.0419 m for the
   1.78 m lifter) and sits one radius anterior of its skin site. The spheres
   press on the `bench_top` support; the feet stay on the floor. The
   stiffness is the spec's 1e5 N/m (MS-20), not OSL-2's 1e7 Pa.
2. **Supine hold.** `solve_bench_hold` solves the standing pose, turns the
   trunk until the back-sphere line is level, sets the legs to the parity
   standard's bench start (hip extension 0.27 rad, knee flexion 1.65 rad),
   and fits the full body with the feet flat on the floor and the back-sphere
   centres weighted 10×. As in OSL-2, the bench top is the lowest back-sphere
   point of the fitted hold: **0.501 m**. The other back spheres clear it by
   0, 0.21, 0.41 and 1.00 mm (RPSH, LPSH, RPSI, LPSI).
3. **Planar solution.** The Tools Movement Optimizer solves `bench_press`:
   80 kg, 1.78 m lifter, 100 kg bar, 2.0 s, 151 samples. Its bench model is
   an arm chain on a fixed shoulder. The bar is the arm tip relative to the
   shoulder, running from lockout (0, 0.696 m) to a chest via-point at 1.0 s
   (0.081, 0.021 m) and back to lockout.
4. **Press.** `lift_bench_to_full_body` holds every coordinate except the
   arm chain at the hold. It moves the left hand to the standard's 0.75 m
   grip and tracks the bar centre to the shoulder midpoint plus the
   optimizer's offset. That offset is scaled by 0.792: the model's
   straight-up reach at the grip width (0.551 m, found by bisection) divided
   by the optimizer's lockout offset.
5. **Window.** A 0.2 s MuJoCo tracking window, centred on the bottom (chest
   touch, 1.0 s), starts at 0.893 s. It runs 200 steps at 1 ms.

## Measured Numbers

| Quantity                                      | Value                                          |
| --------------------------------------------- | ---------------------------------------------- |
| Bench top height (lowest back-sphere point)   | 0.501 m                                        |
| Optimizer bar offset, lockout → chest         | 0.696 → 0.021 m above the shoulder             |
| Optimizer peak vertical bar velocity          | 1.42 m/s down (0.88 s), 1.42 m/s up (1.12 s)   |
| Optimizer horizontal bar travel               | 0.090 m (−0.009 to 0.081 m)                    |
| Model bar height (world), lockout → chest     | 1.161 → 0.633 m (0.527 m travel)               |
| Model bar horizontal travel (toward the feet) | 0.064 m                                        |
| Window bar: lowest point, net change          | −74.1 mm, −17.7 mm                             |
| Trunk support, window steps 50 / 100 / 200    | back 732 / 669 / 462 N, feet 305 / 401 / 206 N |
| Back-sphere penetration (mid-window)          | 0.9–2.6 mm                                     |
| Floating-root drift over the window           | ≤ 1.8 mm, ≤ 0.21°                              |
| RK4 stability (max amplification)             | 0.85                                           |
| Bundle size                                   | 229,353 bytes                                  |

IK residuals are the maxima over all 151 frames, in metres, rounded up to
0.1 mm. They are asserted in `tests/unit/lifting/test_bench_press_bundle.py`.
The held landmarks (ankles, knees, hips, shoulders) are 0 by construction;
the bar is 0.0148. The bar residual is 4.9 mm at lockout and 14.7 mm at the
chest touch: the arms cannot fold the bar down to the optimizer's chest
point. Grip closure is at most 2.9e-5 m, and the lowest foot sphere sits
0.69 mm into the ground.

The supine hold fit leaves residuals against its targets of 0.8–1.3 mm on the
back spheres, 1.1–2.8 mm on the hips and shoulders, and 2.5–12.3 mm on the
knees and ankles. Those targets are the standard's leg angles applied to the
model's segment lengths.

## Cross-Engine Bar-Path Parity

`bar_path_parity.json` (measured at commit `eb9839de`) uses the back-squat
protocol and the same fixed tolerances. Every engine replays the whole
bundle open loop and places the bar (`Grip` frame) with its own forward
kinematics, the back spheres pressing on the bench top.

| Engine    | Full Window Bar Max (m) | Full Window Coordinate Max (rad) | 20-Step Segments Coordinate Max (rad) | 20-Step Segments Frame Max (m) |
| --------- | ----------------------- | -------------------------------- | ------------------------------------- | ------------------------------ |
| MuJoCo    | 0 (bit exact)           | 0                                | 0                                     | 0                              |
| Drake     | 1.0e-11                 | 1.9e-9                           | 8.7e-13                               | 1.0e-13                        |
| Pinocchio | 1.2e-11                 | 2.2e-9                           | 8.7e-13                               | 8.8e-14                        |
| OpenSim   | 1.5e-13                 | 2.6e-11                          | 2.2e-14                               | 4.0e-15                        |

The bar stays within 1e-9 m over the whole 200-step window in every engine,
not only over the 20 CI steps. Every coordinate is inside the 1e-6 rad L2
bound (worst final coordinate `LSInputX`), so the bench press has no
coordinate xfail. (The squat and the deadlift windows miss that bound on
their foot-sphere coordinates; why the bench window does not was not
investigated.)

## Current Records (Whole-Row Lift Body, #13037)

The sections above were measured on earlier bodies and stay for history. The
bundle is now built on the AN-1 lift body regenerated on the whole de Leva
Table 4 rows (#13102) with the default 10 deg tibial torsion (#12814), and
every number below is read from the committed bundle and
`bar_path_parity.json` (source commit `858f4c03`).

Bench top 0.500 m. Held landmarks 0 by construction, bar residual 14.59 mm
(bound 14.8 mm). Grip closure at most 3.0e-5 m (2.998e-5); the
regression-snapshot bound in `tests/unit/lifting/test_bench_press_bundle.py`
is re-derived from 2.9e-5 to 3.0e-5 m (#13041, issuecomment-6243170509). Lowest foot sphere 0.68 mm, RK4 amplification
0.85, bundle 229,631 bytes. The supine fit leaves 0.70-1.20 mm on the back
spheres, 1.37-2.66 mm on the hips and shoulders and 2.11-12.36 mm on the
knees and ankles. The support forces, penetration and root drift rows above
were not re-measured.

| Engine    | Full Window Bar Max (m) | Full Window Coordinate Max (rad) | 20-Step Segments Coordinate Max (rad) | 20-Step Segments Frame Max (m) |
| --------- | ----------------------- | -------------------------------- | ------------------------------------- | ------------------------------ |
| MuJoCo    | 0 (bit exact)           | 0                                | 0                                     | 0                              |
| Drake     | 5.4e-12                 | 7.6e-10                          | 8.2e-13                               | 2.4e-13                        |
| Pinocchio | 6.4e-12                 | 9.2e-10                          | 8.3e-13                               | 2.4e-13                        |
| OpenSim   | 4.2e-13                 | 5.2e-11                          | 2.7e-14                               | 1.1e-13                        |

Every coordinate is inside the 1e-6 rad L2 bound (worst final coordinate
`LSInputX`).

## Literature Comparison (Evidence, Not Tuning Targets)

These sources were checked on 2026-10-17. Each entry says how.

- **Bench height.** IPF Technical Rules Book 2026 (updated 1 January 2026),
  "Bench" under equipment: 42–45 cm from the floor to the top of the
  uncompressed pad. I read this in the text of the rulebook PDF (a
  third-party copy at benchpresschampion.com; the IPF site link returned
  404). The model's bench top is 0.501 m, 5–8 cm above that range. It is set
  by the hold, not by the rule: the supine root height comes from the
  standing ankle height and the standard's leg angles. It is not tuned to
  the rule.
- **Bar velocity.** González-Badillo & Sánchez-Medina (2010), _Int J Sports
  Med_ 31(5):347–352, doi:10.1055/s-0030-1248333, PMID 20180176. The
  abstract, read through Europe PMC, reports a mean velocity at 1RM of 0.16 ±
  0.04 m/s in the bench press for 120 strength-trained men, with mean
  propulsive velocity falling with %1RM (R² = 0.98). The model's press is
  much faster: 0.527 m in 1.0 s (about 0.53 m/s mean, 1.42 m/s peak). That
  is a submaximal-speed lift for a 100 kg bar, and a minimum-effort planar
  solve need not reproduce a maximal lift.
- **Range of motion, path and timing.** Larsen, Gomo & van den Tillaar
  (2021), _Front Sports Act Living_ 2:637066, doi:10.3389/fspor.2020.637066,
  Table 1, 1-RM bench press, 14 trained men. With a medium grip the bar rises
  0.345 ± 0.050 m from the chest to the second velocity peak, reached
  1.22 ± 0.61 s into the ascent, and its horizontal position changes by about
  0.13 m. I read these values from the open-access article through a
  summarising fetch, not from the PDF. The model's 0.527 m vertical travel is
  about 0.15–0.18 m longer, because the optimizer's chest point is only
  0.021 m above the shoulder joint centre, i.e. inside the chest. Its 1.0 s
  ascent and 0.064 m horizontal travel are within or below the reported
  ranges.
- **Not verified.** Madsen & McLaughlin (1984), _Med Sci Sports Exerc_
  16(4):376–381, is widely cited for a non-vertical bench bar path and the
  sticking region. I could not retrieve its text or abstract, so this record
  does not rely on it.

## Limitations

- Only the arm chain moves. The trunk, legs, scapulae and head are held at
  the hold, so there is no leg drive, arch change or scapular retraction.
- The chest has no contact: the bar is not stopped by it. The optimizer's
  chest position lies inside the model's chest (see above).
- The bar mass enters only the planar optimizer, not the full-body dynamics
  (LIFT-4, #11744). The bar is the club's `Grip` frame with the left hand
  moved to the grip width.
- The bench is a plane, not a finite pad: it has no edge and no width limit.
  The 1e5 N/m sphere stiffness gives 1–3 mm penetration rather than OSL-2's
  stiffer seat.
- Pinocchio's analytic contact derivatives do not model raised supports.
  `contact_effort_derivatives` raises for a spec with supports; replay and
  forward dynamics are unaffected.
- The bench height is 0.501 m, above the IPF 0.42–0.45 m range (see above).

## Regenerate

```bash
MUJOCO_GL=egl python3 scripts/generate_lift_bundle.py --lift bench_press
MUJOCO_GL=egl python3 scripts/measure_lift_bar_path_parity.py --lift bench_press
```
