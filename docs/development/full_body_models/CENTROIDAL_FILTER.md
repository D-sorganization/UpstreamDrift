# Centroidal Feasibility Filter v2

Issue #11669, epic #11667. Source: `src/shared/python/motion_matching/pipeline/centroidal_filter.py`. Opt-in pipeline stage `--centroidal-filter` (runs after `--zmp-filter`).

## Problem

The tracked reference demands a ground reaction the feet cannot deliver in the finish. The full zero-moment point

$$p = c_{xy} - \frac{(c_z - z_g)\,F_{xy} + \hat{z}\times M_{xy}}{F_z}, \qquad F = \dot P - m g,\quad M = \dot L$$

(`c` centre of mass, `P` linear and `L` centroidal angular momentum, `z_g` ground height) leaves the support polygon, the vertical force dips, and the horizontal force leaves the friction cone. The cart-table `--zmp-filter` ignores the `M = \dot L` term.

## Method

Frames before `t_on = 0.9 s` are untouched. Over the finish the joint trajectory becomes `q + dq`, with `dq` a cubic B-spline in time (spacing 0.05 s, value and slope zero at `t_on`). Each outer iteration linearises every frame at the current trajectory and solves one bounded least-squares problem.

- Objective: `|L^T dq|^2` per frame, where `L L^T` is the marker-Jacobian Gram matrix plus `0.05^2 I` (stay close to the markers), a ridge, and a small acceleration penalty.
- Soft equalities: loop closures hold (`J_c dq = -r_c`, weight 1e4) and stance feet do not move (contact-site Jacobian times `dq` is zero, weight 1e3).
- Inequalities from `dC = J_c dq`, `dF = m J_c \ddot{dq}`, `dM = H \ddot{dq}` (`J_c` the centre-of-mass Jacobian, `H` the centroidal momentum matrix):
  - linearised ZMP inside every edge of the support polygon shrunk by 0.03 m (central-difference sensitivity of `p` to `(c, F, M)`);
  - `F_z >= 0.5 m g`;
  - `|F_h| <= 0.6 F_z` on an inscribed octagon.
  They enter through an active-set quadratic penalty (`penalty = 100`).
- Trust region: every spline coefficient satisfies `|x| <= bound` (0.03 rad initially).
- Acceptance: the exact (nonlinear) merit, the sum over `t >= 1.0 s` of squared ZMP distance outside the polygon, squared vertical-force shortfall and squared friction-cone excess, must fall. Otherwise the step is rejected and `bound` halves. Ten iterations.

Nothing is smoothed afterwards, no tolerance is changed, and the replay controller is unchanged.

## Results

Runs on `main` at the Balance-1 commit, `--static-seeds`, driver without and iron with `--zmp-filter`. "Before" is the same run without `--centroidal-filter`. The reference ZMP columns are the 1.0 to 1.5 s outside fraction (acceptance below 0.2). Marker RMS is the simulated dynamics RMS against the capture (acceptance at most 90 mm).

| Club   | Reference ZMP outside before | after | Marker RMS before | after   | Acceptance |
| ------ | ---------------------------- | ----- | ----------------- | ------- | ---------- |
| 7-iron | 0.400                        | 0.028 | 57.9 mm           | 65.5 mm | met        |
| Driver | 0.756                        | 0.739 | 83.3 mm           | 87.1 mm | not met    |

Finish metrics of the simulation, which is what the plant actually delivers (before to after): iron friction saturated fraction 0.30 to 0.39, foot slide 340 to 354 mm, pelvis yaw error 19.4 to 18.4 degrees; driver 0.25 to 0.31, 502 to 538 mm, 34.5 to 38.3 degrees. The filter makes the reference dynamically feasible in the ZMP sense for the iron. It does not change the foot slide, foot pivot or pelvis yaw lag of the replayed simulation, so the replay is not more physical because of it. The iron reference also reaches `F_z` up to 3.85 body weight.

Driver: apart from one 0.015 rad step, only steps of at most 0.004 rad pass the exact merit test, so ten iterations reduce the merit from 234 to 115 but not the outside fraction. The driver reference needs a larger correction than the linear model supports; the ZMP excursion is dominated by frames where the reference vertical force is close to zero.

## Failed Experiments

A dense Kronecker QP over all frame corrections took about 15 minutes per step and diverged. A per-frame joint-space QP without a time basis or trust region left several hundred linearised constraints violated, produced 8 degree joint steps and raised the outside fraction from 0.71 to 0.97. The B-spline basis, trust region and exact-merit acceptance above replaced both.

## Limitations

- Linearisation uses acceleration-only sensitivities; the velocity-dependent terms of `dF` and `dM` are neglected, which is why large steps fail.
- The correction is kinematic: it is re-checked only through the reference ZMP and the same replay, never through an independent open-loop dynamics replay.
- Unloaded frames (`F_z` near zero) have no defined ZMP; their rows are dropped, and the merit counts them as one metre outside.

## Reproduction

```bash
python3 docs/development/full_body_models/evidence/ground_support/run_ground_support.py \
  --spec docs/development/full_body_models/full_body_spec_anthro_iron7.json \
  --skip-hip-calibration --static-seeds --zmp-filter --centroidal-filter \
  --capture iron --out <dir>
```

The receipt carries `dynamics.centroidal_filter` (config, per-pass merit, accepted flag and zmp summary, before and after). Side-by-side finish renders live in `/home/dieterolson/Videos/Parity Audit/balance/`.
