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
World -[Hip 6-DOF]- Pelvis -[Torso revolute, axial]- lower trunk -[Spine universal]- upper trunk = Hub
Hub -[Scapula universal, centred at the hub]- strut (HubtoSLength 0.254 m)
    -[Shoulder gimbal 3-DOF]- Upper arm -[Elbow]- Forearm -[Pronation]- -[Wrist universal]- Hand
```

(The guide's prose lists the spine universal before the torso revolute; the
model's own logs show the torso revolute sits at the hip and the spine
universal above the lower trunk, see the validation section below.)

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

**Axis conventions.** Since `golfer-scapula/2.0` every body frame of the
Python model is the corresponding Simscape sensor frame: each joint carries
the constant pre/post rotations, primitive axes and signs identified from
the MATLAB model's own logs, so the 27 shared DOFs are the Simscape
start-position angles up to a recorded sign and the export is a table
lookup, not a conversion. See "Simscape Axis Validation" below (#9714).

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
it. The gate never removes more than half of the observations: when most of
them are beyond it the model cannot represent the motion (a rigid pendulum
on a bent-arm golfer), and the honest RMS must say so rather than a fit to
nothing. The weak rest term fixes the one direction a point landmark cannot see,
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

## Variants: Any Subset of the Cameras (#9790)

Capture once with every camera, then match with any subset. A **variant**
is a named match of the same recordings: `variants/<name>/` holds its own
`reconstruct/` and `model/` trees while the recordings and observation sets
stay shared; the session root is the default variant (`""`). One resolver,
`src/motion_capture/variants.py::variant_dir`, is used by every command
that reads or writes those trees, and `variants/index.json` records each
variant's views, observation set and source.

```
rig reconstruct --session S --cameras start.json --anchor shank=0.42 \
    --views face_on,down_line --variant pair_fd
rig fit-model   --session S --variant pair_fd
rig fit-model   --session S --variant cam_face --from-views face_on --cameras-from ""
rig overlay     --session S --view overhead --variant "" --variant pair_fd --out o.mp4
rig compare-variants --session S
```

`--views` restricts and orders the cameras (two or more for triangulation);
`--observations` picks the detector set (`observations`,
`observations_openpose_dnn`, `observations_manual`, `observations_edited`).

## Image-Space Fit: One Camera and Up (#9794)

`reconstruct/model/fit2d.py` fits the articulated model to the 2-D
keypoints of one or more views directly, with the data term
`(project_v(landmark) - keypoint_px) / sigma_px` and the same continuity,
limit, rest and length priors as the 3-D fit (`_Problem2D` overrides only
the landmark block and the reporting hooks of `fit._Problem`). The cameras
come from a previous multi-camera variant (`--cameras-from`) and that
dependency is recorded in the provenance. With one view the in-plane motion
is observed directly; depth comes from perspective and the model's known
segment lengths, so this is the seam for the single-camera research:
`rms_px` and per-view `residual_px` are reported, `rms_m` is NaN.

Synthetic evidence (lab rig, 1 px noise): landmark RMS against the truth
under 10 mm from three views and under 20 mm from two; from one view the
in-plane error stays under 30 mm while depth is only weakly observed
(`tests/motion_capture/reconstruct/model/test_image_space_fit.py`).

## Overlays and the Cost of Fewer Cameras (#9795, #9796)

`reconstruct/overlay3d.py` projects a variant's reconstructed joints and
fitted model landmarks onto **any** view, including views the variant never
used (_held out_; the camera comes from the variant's own reconstruction,
else from the variant it borrowed cameras from, else from the default).
`src/tools/capture_rig/overlay_render.py` draws one or several variants on
the recording in distinct colours with a legend (`rig overlay`, and the
_Model overlay_ checkboxes in the player). `rig compare-variants` writes
`variants/comparison.{json,md}`: per variant the reprojection RMS on every
view with held-out views flagged, the 3-D joint RMS and the per-DOF angle
RMS against a reference variant. `docs/motion_capture/evidence/camera_subsets.md`
tabulates the synthetic 3 / 2 / 1-camera experiment.

## Provenance (#9792)

Every JSON the pipeline writes carries `schema_version` and a `provenance`
block (`src/motion_capture/provenance.py`): creation time, the generating
package/module/version/git SHA, the hashed inputs, the parameters (views,
observation set, anchors, camera source, detector, fit options, model,
source kind) and `derived_from`. `rig lineage --session S --path FILE`
walks a model fit back to the reconstruction, the observation files and the
recordings; the tile's _Provenance_ tab shows the same when a result row is
clicked. The detector's own record inside an observation file is kept and
folded into the parameters, so the plug-in and its options are visible at
every hop.

## Manual Annotations as a Source (#9791)

`src/motion_capture/annotate/` holds the Qt-free logic: a sparse
`annotations/<view>.json` store (clicks, skips for occluded joints, an
optional `base_set` when it corrects a detector), a guided cursor that walks
frames and joints, and `rig annotations-to-observations`, which writes a
`view-observations/1.0.0` set (estimator `manual`) or, with
`--merge-with SET`, the detector set with the corrections applied (clicks
replace, skips reject) and the counts in the provenance. The Capture Rig
tile's _Annotate / edit points_ dialog drives the cursor over the player's
view; opened with an observation set selected it becomes the outlier
editor. The reconstruction and the model fit treat confidence 0 as
unobserved and carry the continuity prior across gaps, so sparse clicks fit
like any other set; `evidence/sparse_annotations.md` records the accuracy
against frame stride.

## Kinetics (#9714, First Slice)

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

What this is not yet: rod inertia is not modelled and the replay is
linearised about the fitted configuration rather than a free multi-second
integration. The axis conventions are validated (next section), so the
exported angles can drive `GolfSwing3D_Kinetic` as start positions; driving
it with the torques still needs the free integration.

## Simscape Axis Validation (#9714)

`src/motion_capture/reconstruct/model/simscape.py` identifies, for every
joint the MATLAB model logs, which logged angle rotates about which axis
with which sign and which constant frames sit either side, by fitting
`R_parentᵀ R_child = A · R_axes(±angles) · B` over every axis order, sign
pattern and column order on the 18 dataset-generator trials (620 frames).
Result and method: `evidence/simscape_axes.md` and `.json`. Every joint is
explained to ≤ 2.7e-4 rad, which is the logs' own precision.

What the logs settled:

- the torso revolute is at the hip and shares the hip's third axis; the
  spine universal is between the two trunk sensors (`TorsoLogs` is the
  lower trunk, `SpineLogs` the upper);
- the hub is a rigid point of the upper trunk at (0, -0.0508, -0.2438) m in
  its frame, 0.061 m + 0.249 m above the lower-trunk origin, and both
  scapula joints are centred on it;
- the strut is 0.254 m along the scapula frame's ∓z, the upper arm 0.3047 m
  along the shoulder frame's +x, and the forearm sensor sits 0.1778 m past
  the elbow on its +z (so the elbow is on that axis, 0.3047 m from the
  shoulder);
- no wrist angles are logged; the wrist universal is named only.

`golfer-scapula/2.0` encodes all of this: joints now take `pre_rotvec` /
`post_rotvec` constant frames (`kinematics.Joint`), the model exposes body
frames (`frames()` / `forward_frames()`), and `SIMSCAPE_NAMES` maps each DOF
to its Simscape variable and sign. The replay test drives the model with
the logged angles and lands every body frame on its sensor to ≤ 1.7e-3 rad
and the joint positions to ≤ 0.6 mm. The remaining assumption is the
Simscape world orientation (z up, x toward the target, y to the golfer's
right), which only affects the root's hip angles and translation.
