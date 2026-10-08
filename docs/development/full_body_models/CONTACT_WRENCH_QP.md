# Contact Wrench Tracking QP (Negative Result)

Issue #11670, epic #11667. Source: `src/shared/python/motion_matching/contact_wrench_qp.py`; opt-in `--tracking wrench-qp`. The default tracker is unchanged.

## Model

A quadratic program per control step on the centroidal momentum balance, solved with Drake `MathematicalProgram` and Clarabel (no other QP backend is installed).

- Variables: joint acceleration $\ddot q_j$, root acceleration $a_r$, one wrench $w_i$ per foot (about the support centre), slack $s$.
- Equality (written as a deviation about the achieved nominal point): $A_G \ddot q + \dot A_G \dot q = \sum_i w_i + m g$.
- Cone per foot (11 rows): a 4-face inscribed friction pyramid with $\mu/\sqrt2$, 4 centre-of-pressure bounds over the polygon of touching spheres (shrunk by `cop_margin_m`), 2 torsion bounds and $f_z \ge 0$. Slack is penalised with `slack_weight`.
- Cost: tracking of the computed-torque joint acceleration target, root-hold, wrench regularisation and slack.
- Torque: $\tau = A_{act}^{+}(a_{des} - b_{act})$ from `sim.affine_dynamics`, with $a_{des}$ the QP joint acceleration.

Units SI, ground normal z. Parameters in `WRENCH_QP_SETTINGS` (`pipeline/dynamics.py`): friction 0.6, CoP margin 10 mm, root weight 1e3, slack weight 1e6.

## Acceptance

| Criterion | Result |
| --- | --- |
| Reproduces the baseline within 1 mm on a feasible reference | Met only with the plant's own cone (no margin): replay difference below 1 mm. Not met with the conservative cone (friction 0.6, 10 mm CoP margin). |
| Post-impact pelvis yaw error below 8 degrees on both clubs | **Not met.** The QP leaves the yaw error at the baseline value on both clubs (driver 34.5, iron 19.4 degrees); torsional friction (#11671) is the lever that moved it. |

## Why It Had No Effect

The torque is a function of the joint acceleration target only. The QP changes the planned wrench and root acceleration, but the plant's contact reacts physically to whatever torque is applied, so changing the plan does not change the realised wrench. A conservative cone activated the slack on the infeasible finish ZMP (reference driver ZMP-inside fraction about 0.2) without changing the realised foot motion. Raising the root gain to 1 was unstable.

## Failed Experiments

- Contact moments about ground contact points: wrong, forces act at sphere centres; the QP activated everywhere.
- Target the commanded `wanted` acceleration: did not reproduce the baseline; the achieved acceleration is the target.
- Conservative cone, margin 10 mm, friction 0.6: no improvement in slide or yaw.
- Root gain 1: unstable.

## Status and Next Step

The QP is kept as an opt-in, tested component and not made the default. A QP that matters must feed the wrench into the actuation (contact-consistent torques, e.g. whole-body inverse dynamics with the wrench as a constraint on the realised contact), not only into the plan. Unit tests: `tests/unit/motion_matching/test_contact_wrench_qp.py` (cone rows, supports, activation, equals baseline on a hold, backend registration).

## Reproduction

```bash
python3 -m src.shared.python.motion_matching.pipeline.cli \
  --spec docs/development/full_body_models/full_body_spec_anthro_driver.json \
  --skip-hip-calibration --static-seeds --capture driver --tracking wrench-qp --out <dir>
```
