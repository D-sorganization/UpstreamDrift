# Articulated Golfer Model and the Continuous-Motion Fit

Epic #9709. Code: `src/motion_capture/reconstruct/model/`. Commands:
`rig fit-model`, `rig export` (adds the Simscape CSV). Tile: _Fit model_.

## Why a Model, and Which One

The multi-camera reconstruction ends with fifteen 3-D joint centres per
frame held together by fixed segment lengths. That is a good estimate of
where the joints are; it is not a description of how the body moved. An
articulated model turns positions into joint angles that a physics model
can be driven with, and its constraints filter what the capture may claim:
a joint cannot bend past its range, a segment cannot change length, and,
above all, no angle may jump between frames.

The topology follows the MATLAB 3-D golf model
(`src/engines/Simscape_Multibody_Models/3D_Golf_Model`), which is the
model the kinetics will later be matched on:

```
World -[Hip 6-DOF]- Pelvis -[Spine universal]- Spine -[Torso revolute, axial]- Hub
Hub -[Scapula universal, centred at the hub]- strut (HubtoSLength 0.254 m)
    -[Shoulder gimbal 3-DOF]- Upper arm -[Elbow]- Forearm -[Pronation]- -[Wrist universal]- Hand
```

Sources: `matlab/MATLAB_GOLF_MODEL_GUIDE.md` lines 52-66 (chain and the three
joint sub-models), `matlab/src/functions/model/readJointStateTargets_GolfSwing3D.m`
lines 12-50 (every joint block path), `matlab/motion_matching/shared/compute_skeleton_fk.m`
lines 36-47 and 159-165 (`fk.lscap = fk.hub`; `LSCAP_TO_LS_BODY = [0 0 -0.254]`),
model workspace `HubtoSLength`, `UpperTorsoLength`, `UpperArmLength`.

**The scapula.** In the MATLAB model the scapula is a two-degree-of-freedom
joint whose rotation centre sits at the hub (base of the neck, top of the
torso) and a fixed-length strut from there to the glenohumeral centre. The
shoulder point therefore moves relative to the trunk, which is exactly the
motion a shoulder-only model cannot represent. The Python spec keeps this
verbatim: `left_scapula` is a co-located joint at `hub` with axes `xy`
(elevation about the target axis, protraction about the vertical) and
`left_shoulder` sits at the end of the `hub_to_shoulder` strut with a
three-axis gimbal.

**Extensions.** The MATLAB model has no legs and no head; the detectors
observe hips, knees, ankles and the nose. The spec adds hips (3 DOF),
knees (hinge) and ankles (landmark only), and a head segment, and marks them
as additions. 27 of the model's 35 DOFs carry the Simscape start-position
names (`LScapStartPositionX`, `TorsoStartPosition`, ...).

**Axis conventions.** Simscape's world is z-up; ours is y-up (x toward the
target, z to the golfer's right). Spine Rx Ry becomes our x and z, Torso Rz
(axial) becomes our y. Names are mapped; signs and orders are not yet
validated against `GolfSwing3D_Kinetic`, and every export says so (#9714).

## The Fit

`fit_trajectory` solves all frames at once. Unknowns: the joint-angle state
q(t) and, optionally, a few segment lengths. Residuals, all in physical units:

| Term        | Form                                      | Default                           |
| ----------- | ----------------------------------------- | --------------------------------- |
| Landmark    | (model - observed) / σ, Huber-weighted    | σ = 10 mm                         |
| Continuity  | (q(t-1) - 2 q(t) + q(t+1)) · fps² / σ_acc | 300 rad/s² (30 m/s² for the root) |
| Joint limit | overshoot beyond [lo, hi] / σ             | σ = 0.01 rad                      |
| Rest        | q / σ_rest for every rotation             | σ = 3 rad                         |
| Length      | (L - L_measured) / σ                      | σ = 3 mm                          |

The continuity term is the filter the user asked for. A landmark that the
model can only reach by a discontinuous change of some angle is out-voted
by the continuity of every other frame; its residual exceeds the gate (5 σ)
after convergence, it is rejected and listed, and the fit is repeated without
it. The weak rest term fixes the one direction a point landmark cannot see,
rotation about the segment's own axis, so a redundant angle settles rather
than wandering (a wander would look like motion).

Outputs per session (`<session>/model/`): `joint_angles.json` (series per
DOF, lengths, rejections, peak speeds), `fit_report.json` (RMS per landmark,
rejections, velocity violations), `landmarks_fit.npy`; after `rig export`,
`joint_angles_simscape.csv`.

## Evidence

Synthetic chains in `tests/motion_capture/reconstruct/model/`:

- forward kinematics keeps every segment length to machine precision and the
  finite-difference Jacobian matches an independent difference;
- a 90-frame motion on a trunk-scapula-arm-leg chain is recovered with 6 mm
  RMS while a 25 cm one-frame jump injected into the wrist is rejected and
  the model wrist stays within 2 cm of the truth; five unobserved scapula
  frames are bridged by the prior; hinges stay inside their limits;
- a tape reading 2 cm off is refined to within 1 cm when the length is fitted;
- on the golfer spec, raising one shoulder relative to the hub is recovered
  as left scapular elevation within 4° with the right scapula still (< 6°).

Real-data evidence needs the three-view take; single-camera sessions skip
this step.

## Any Model, the Same Fit

`reconstruct/model/registry.py` names the fittable models; each carries its
spec, the map from its landmarks to the reconstruct joints (a landmark may be
the mean of several joints: the pendulums' _hands_ are both wrists) and the
lengths the fit may learn. Registered today:

| name               | DOF | what it is                                                             | learnable lengths                   |
| ------------------ | --- | ---------------------------------------------------------------------- | ----------------------------------- |
| `golfer` (default) | 35  | the scapula golfer above                                               | strut, torso halves, hip half, head |
| `double_pendulum`  | 6   | pivot at the shoulders (plane + arm angle), one rigid arm to the hands | arm                                 |
| `triple_pendulum`  | 7   | pivot, upper-arm angle, elbow hinge, forearm to the hands              | upper arm, forearm                  |

`rig fit-model --model NAME [--fit-lengths]` runs the same continuous fit on
any of them, so every model's constraints are enforced and its dimensions
learned from the take. `rig compare-models` fits several and ranks them in
`model/comparison.md` by a DOF-penalised score (log RMS plus DOF per
observed coordinate), with RMS, rejections and peak joint speeds beside it.
Register another model with `register_model`; the fit, the comparison, the
kinetics and the tile pick it up without further code.

## Kinetics (#9714, first slice)

`rig kinetics --model NAME --body-mass KG` computes, for the fitted
trajectory of any registered model, the generalised torques
`tau = M(q) qdd + C(q, qd) qd + G(q)` with the inertia matrix from point
masses at the segment centres (de Leva mass fractions), Coriolis terms from
the motion of `M`, and gravity, all from the model's own forward kinematics.
A forward replay integrates those torques back from the fitted initial state
and reports its drift from the fitted angles; that drift is the acceptance
number for "kinematics and kinetics agree". Output: `model/kinetics.json`
(torques per DOF, peak torques, replay error per DOF, assumptions stated;
Simscape names attached for the golfer).

What this is not yet: rod inertia is not modelled, the replay is linearised
about the fitted configuration rather than a free multi-second integration,
and the Simscape axis conventions still need validation before the torques
can drive `GolfSwing3D_Kinetic` directly.

## Toward Kinetics (#9714)

The joint-angle series in the Simscape vocabulary is the input the MATLAB
model's kinetic drive needs: inverse dynamics on the fitted kinematics gives
torques consistent with the motion, and a forward replay that reproduces the
fitted angles within tolerance closes the loop. That step, and the axis
validation it requires, is the next child of the epic.
