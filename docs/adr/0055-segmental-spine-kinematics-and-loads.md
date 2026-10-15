# ADR-0055: Segmental Spine Kinematics, Load Observables and Golf-Term Definitions

- Status: Accepted
- Date: 2026-10-15
- Decision Makers: Repository owner (directed implementation of epic #12735 on 2026-10-15); reviewed by the Astra, Fable and Argon agent panel
- Related Issues/PRs: Epic #12735, child #12736; consumers #12739–#12774

## Context

Golf instruction describes the backswing trunk as "extension, lateral flexion and
rotation". Those words mix lab-frame and pelvis-relative quantities, treat Cardan
angles as if they added, and say nothing about which intervertebral joints
produce the motion. UpstreamDrift models the spine as one 3-DOF `back` joint
(`golf_humanoid.osim`, `lumbar_extension/bending/rotation`, ZXY with
ISB-aligned axes), estimates spinal load heuristically
(`src/shared/python/injury/spinal_load_analysis.py`), and computes "X-factor" in
several incompatible ways. Epic #12735 integrates segmental OpenSim spine models
(Bruno, Bouxsein and Anderson 2015; Senteler et al. 2016). Every child of that
epic needs one fixed set of conventions so that low-level implementers do not
invent their own.

The repository also contains two lab-frame statements that disagree
(`swing_comparison/turn.py` cites ADR-0041 as "world Z up, golfer faces −X,
target −Y", whereas ADR-0041 proposes "x toward the target, y up"). This ADR
therefore defines spinal quantities **relative to body frames**, and every
quantity that needs the lab takes the gravity direction and the target
direction as explicit inputs.

## Decision

### 1. Vertebral Frames and the Per-Level Transform

Each vertebra (and the sacrum, pelvis and the named terminal body) carries an
ISB frame (Wu et al. 2002, J Biomech 35:543): origin at the vertebral body
centre, **X anterior, Y cranial, Z right**.

A spinal level `i` (for example `L4_L5`) joins an **inferior** (caudal)
vertebra and a **superior** (cranial) vertebra. Its homogeneous transform maps
coordinates expressed in the superior frame into the inferior frame:

```
T_i = T_i0 · T(R(θ_i), d_i)
```

- `T_i0` is the neutral offset (resting lordosis or kyphosis and the
  inter-centre offset), read from the model's parent and child
  `PhysicalOffsetFrame`s.
- `θ_i = (θ_FE, θ_LB, θ_AR)` are the ISB joint-coordinate angles, intrinsic
  sequence **ZXY**, exactly the `isb_spine` profile of
  `src/shared/python/biomechanics/joint_conventions.py`:
  - `θ_FE` about Z: **positive = extension** (the superior Y axis tilts
    posteriorly);
  - `θ_LB` about the floating X axis: **positive = right lateral bending**;
  - `θ_AR` about Y: **positive = left axial rotation** (the superior X axis
    turns toward the left). This is the same sense as `isb_thorax`
    (`left_axial_rotation`); the two profiles differ only in name.
- `d_i` is the translation, zero for rotational models and non-zero for the
  Senteler 30-DOF assembly. Contracts are SE(3); rotational results use the
  SO(3) projection.

Angles are extracted with `anatomical_joint_angles(R, "isb_spine")`, whose
`OrientationResult.singular` flags the gimbal-lock branch. The neutral offset
is excluded from the reported angles and included in every composition.

### 2. Composition Invariant

For a chain from the pelvis to a **named terminal body** (`T1` for the Bruno
model, `thorax_proxy` for the synthetic fixture, `torso` for `golf_humanoid`):

```
T_pelvis→terminal = T_sacrum · T_L5S1 · T_L4L5 · … · T_terminal
```

The caudal factor comes first because each `T_i` maps superior into inferior.
The relationship between the terminal vertebra and the ISB thorax
(`isb_thorax`, IJ/C7/PX/T8 landmarks) is a separate, model-specific calibration
offset `T_terminal→thorax`; it is never assumed to be the identity.

Tolerances: 1e-9 for analytic and synthetic tests; native models report their
measured composition error and state their tolerance in the receipt.

### 3. Angles Never Add

Per-level Cardan angles do not sum to the global angles. Worked example (three
levels, ZXY degrees, `(FE, LB, AR)`):

| Level | FE | LB | AR |
|---|---|---|---|
| L3_L4 | −10 | 5 | −15 |
| L2_L3 | −10 | 8 | −20 |
| L1_L2 | −15 | 10 | −25 |
| **Sum** | −35 | 23 | −60 |
| **Composed** `anatomical_joint_angles(R_L3L4·R_L2L3·R_L1L2)` | **−25.36** | **31.83** | **−57.57** |

The 9.6° flexion difference is far larger than measurement noise. The angle
sum may be reported only as a labelled diagnostic (`angle_sum_diagnostic`),
never as a definition of global trunk motion.

### 4. Sign and Frame Map

- **OpenSim bodies** (Rajagopal, Bruno and the fixture) use X anterior, Y up
  and Z right in the anatomical neutral, so ISB axes coincide with body axes
  up to `T_i0`. Every model's actual axes are verified by the inventory
  (#12753) with an alignment tolerance of 1°; a misaligned model needs an
  explicit axis map, never a silent pass.
- **`golf_humanoid` `back` joint**: ZXY TransformAxes `(0 0 1)`, `(1 0 0)` and
  `(0 1 0)` on `lumbar_extension`, `lumbar_bending` and `lumbar_rotation`, so
  `R_pelvis→torso = R_ZXY(lumbar_extension, lumbar_bending, lumbar_rotation)`
  composed with the `pelvis_offset` and `torso_offset` frames.
- **Lab quantities** take explicit unit vectors `up` (opposite gravity) and
  `target` (toward the target along the target line) in the same frame as the
  body poses. No lab axis convention is assumed.
- **Handedness**: for a right-handed golfer the lead side is the left and the
  trail side is the right; for a left-handed golfer they swap. Golf-oriented
  quantities are trail- or lead-signed:
  - `trail_axial_rotation = −θ_AR` (right-handed), `+θ_AR` (left-handed), so
    that **backswing trunk rotation is positive** for both.
  - `lead_lateral_flexion = −θ_LB` (right-handed), `+θ_LB` (left-handed).
  - Extension is not handed.

### 5. Load Observables

Three observables are produced and never conflated:

| Observable | Definition | Includes |
|---|---|---|
| `GENERALIZED_EFFORT` | Inverse-dynamics generalized force on each spinal coordinate | Net effect of everything crossing the level; no split between muscles and joint |
| `JOINT_REACTION` | OpenSim `JointReaction` resultant at the level, computed with static-optimization muscle forces applied | Intervertebral (bone-on-bone/disc) resultant after muscles carry their share |
| `PASSIVE_ELEMENT` | Bushing, ligament and disc-stiffness element forces where the model has them | Passive structures only |

For every observable:

- the wrench is the force and moment exerted **on the superior body by the
  inferior body**;
- it is expressed in the **inferior vertebra ISB frame**;
- its reference point is the joint centre (mid-disc). Moving it uses
  `src/shared/python/force_overlay/conversions.py::move_wrench_point`, which
  transports the moment `M' = M + r × F`;
- **compression positive = +F_y** (a compressed disc pushes the superior body
  cranially). With `child = superior`, this is what
  `JointReaction(apply_on_bodies=child, express_in_frame=parent)` returns
  before the axis map;
- AP shear = `F_x` (positive anterior) and ML shear = `F_z` (positive right);
- tension-positive consumers (`body_part_viz/axial_loads.py`) receive
  `tension = −compression`.

### 6. Coupling

Coupling constraints (`CoordinateCouplerConstraint`) **prescribe kinematics
only**. Following Bruno et al. 2015, all spinal DOFs are unconstrained during
inverse dynamics, static optimization and joint-reaction analysis, so
constraint multipliers never enter disc loads. Static optimization is mandatory
for `JOINT_REACTION` results. The cost exponent `p ∈ {2, 3}` is recorded; Bruno
uses `p = 3`.

A coupling law maps independent coordinates `φ` to per-level angles:

- linear: `θ_i = C_i · φ` with `C_i` of shape `(3, n_φ)`. Cross-axis terms are
  allowed, for example axial rotation inducing lateral bending with a
  level-dependent sign (Pearcy and Tibrewal 1984; Fujii et al. 2007);
- polynomial or piecewise (state-dependent), for example the lumbopelvic
  rhythm scheduled on flexion (Bruno 2015, 2.5 to 1.0).

Every law carries an id, a version and a citation (DOI with table or page).

### 7. Distribution Problem

Distributing a measured or matched pelvis→terminal rotation over the levels is
underdetermined. It is defined as:

```
φ* = argmin_φ ‖ log( R_targetᵀ · R(φ) ) ‖      subject to the coupling law and joint bounds
```

where `R(φ)` is the composed chain of item 2 under the law and `log` is the
SO(3) logarithm (rotation vector).

- The coupling law is **never altered** to close the residual.
- The geodesic residual (radians) and a convergence flag are outputs.
- An infeasible problem is flagged, not repaired.
- Every per-level kinematic output carries `distribution_source`, one of:
  - `coupling:<law_id>@<version>`;
  - `ik:<observation set>;priors:<…>;reg:<…>` for marker-based IK (endpoint IK
    alone does not identify individual vertebrae);
  - `measured:<source>`.

### 8. Golf-Term Registry

Terms are versioned functions with explicit inputs. Existing implementations
keep their numeric meaning under legacy ids; new ids are new measurands.

| `term_id` | Definition | Inputs | Units / sign |
|---|---|---|---|
| `x_factor_jcs_axial` | `trail_axial_rotation` of `R_pelvis→thorax` (ZXY) | pelvis→thorax rotation, handedness | deg, backswing positive |
| `x_factor_line_turn` | Upper-trunk line turn minus pelvis line turn in the horizontal plane (`swing_comparison/turn.py`) | marker or model points, `up` | deg, backswing positive |
| `x_factor_position_proxy` | Legacy joint-position proxy (`api/services/analysis_service.py`, `analysis/swing_metrics.py::compute_x_factor`) | shoulder and hip joint positions | as implemented |
| `x_factor_euler_yaw_legacy_v1` | Thorax yaw minus pelvis yaw from `[roll, pitch, yaw]` (`injury/spinal_load_analysis.py`) | thorax and pelvis Euler angles | as implemented |
| `trunk_extension_jcs` | `θ_FE` of `R_pelvis→thorax` | pelvis→thorax rotation | deg, extension positive |
| `trunk_lateral_flexion_jcs` | `lead_lateral_flexion` of `R_pelvis→thorax` | pelvis→thorax rotation, handedness | deg, lead-side positive |
| `side_bend_tilt` | Lab-frame trunk-axis tilt split into components (`motion_matching/posture_metrics.py::SpineBend`) | trunk axis, `up`, `target` | deg |
| `crunch_factor_legacy_v1` | `\|lateral°\| · \|rotation°\| / 100` (existing analyzer) | angle series | deg² / 100 |
| `crunch_factor_rate` | Lateral flexion angle × axial rotation angular velocity (Sutcliffe et al. 2008 sense; confirmed in #12737) | `trunk_lateral_flexion_jcs`, `x_factor_jcs_axial` series | deg·deg/s |
| `early_extension` | Change in pelvis-to-target-line distance (or pelvis anterior translation) from address, with `trunk_extension_jcs` change reported alongside | pelvis position, `target`, `up`, events | m, toward the ball positive |
| `reverse_spine_angle` | Trunk-axis tilt toward the target at the top, in the vertical plane containing the target line | trunk axis, `up`, `target`, top event | deg, toward target positive |

`early_extension` and `reverse_spine_angle` are not functions of intervertebral
rotations alone. Their extra inputs are mandatory and missing inputs raise.

For JCS terms, the registry also provides a per-level decomposition: the
contribution of level `i` is the change in the term when the chain is
truncated after level `i` versus before it. This is a decomposition, not an
additive identity; it sums to the global value only to first order, and the
remainder is reported.

### 9. Scope of Claims

- Model loads are **model estimates** under declared assumptions (model,
  coupling law, cost exponent, included external loads).
- Heuristic injury classifications (the existing analyzer) are labelled
  heuristic and are never applied automatically to model outputs.
- Golf literature magnitudes (Hosea et al. 1990, L3–L4 about 8 BW; Lim et al.
  2012, L4–L5 above 6 BW) are **context bands, not acceptance**.
- Acceptance tiers stay separate: (a) analytic and synthetic verification; (b)
  reproduction of the integrated model's own published outputs; (c) golf
  context.
- A missing licensed asset means `unqualified`; a skipped test is never
  acceptance.

## Alternatives Considered

1. **Scale global Cardan angles by per-axis fractions.** Rejected. The result
   does not compose to the global rotation (item 3), and it cannot express
   cross-axis or state-dependent coupling.
2. **Post-hoc "exact-composition correction" of scaled angles.** Rejected. It
   is non-unique and silently alters the published coupling ratios.
3. **Report loads in each body's own frame, or on the inferior body.** Rejected
   for comparability: the clinical and golf literature reports loads on the
   disc in the inferior-vertebra frame, with compression positive.
4. **One global lab frame for every quantity.** Rejected because the
   repository holds two conflicting lab conventions. Body-relative definitions
   plus explicit `up` and `target` inputs are unambiguous under either.

## Consequences

- Positive:
  - One contract (`src/shared/python/spine/`) for every engine and tool.
  - Per-level outputs always state the assumption that produced them.
  - Loads are comparable to the literature by construction.
- Negative:
  - Legacy X-factor and crunch values keep their old meanings under legacy
    ids, so consumers must choose an id deliberately.
  - Static optimization is required for compression, which costs runtime on
    the 620-actuator Bruno model.
- Follow-ups: implemented by the children of #12735. The cervical spine,
  intra-abdominal pressure and disc finite-element analysis are out of scope.

## Validation

- `tests/shared/python/spine/` holds the composition invariant (1e-9), the
  worked example above as a regression test, the sign tests for every term and
  load component, and the handedness flips.
- `tests/opensim/spine/` holds native checks: fixture composition equals the
  OpenSim body transforms, coupler round trips, and analytic gravity and
  inertia load cases.
- Integration fidelity against Bruno 2015 is tracked in #12763 and is reported
  separately from golf context bands (#12764).
