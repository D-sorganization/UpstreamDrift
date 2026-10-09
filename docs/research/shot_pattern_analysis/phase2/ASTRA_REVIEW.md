# Astra Review: Matched Delivery and Strike Variation

## Incremental Contact Review: Current Disposition

The impulse-incremental Coulomb/energetic-restitution implementation supersedes
the historical stick-first primary discussed below. Its tangential disk solve
is the backward implicit friction increment; refinement resolves sliding and
admissible sticking. Compression and rebound normal work are integrated separately,
with rebound limited to e² times compression work. This does not impose a Newton
velocity restitution ratio in coupled collisions. The shared raw Python/Numba
kernel uses identical equations without fast-math.

Independent seeded checks covered 100 spatial contact states with rotated inertia,
nonzero initial ball/head angular velocities, restitution 0/0.4/0.83/1 and friction
0.02/0.2/0.4/1/5. All completed. Maximum linear/angular momentum residuals were
1.82e−15 kg·m/s and 1.36e−15 kg·m²/s; normal-work restitution residual was at most
8.89e−16 J. No energy increase or deficit in required normal dissipation occurred.
Twelve raw Python/compiled comparisons agreed exactly in impulse; rotated-frame
impulse covariance differed by at most 8.0e−15 N·s. All non-roundoff 128→256→512
step comparisons improved; largest fine ball-velocity difference was 0.000616 m/s.
This establishes sampled numerical/invariant evidence, not empirical collision
accuracy or a universal convergence proof.

Reproduction scripts and receipts are in `validation/independent_contact_audit.py`
and `validation/independent_contact_audit.json`. The separate deterministic landing
and bounded-matching preflight is `validation/independent_preflight.py`. Its
predeclared practical gate is 0.05 m for fine landing changes, with 128→512 primary
changes also recorded; unsupported outcomes remain explicit. The deterministic preflight has now passed as detailed below; the end-to-end
pilot and final source freeze remain required before the final matrix.

Integration findings sent to owners: inconsistent scoring radius, incomplete
input/output covariance and launch/spin angles, matrix source drift between cells,
and rejection of admissible zero-spin flight. Owners are adding regression fixes.
Zero angular norm ≤1e−10 rad/s is acceptable as explicit numerical zero with
undefined axis, null serialized tilt, and zero Magnus force. Empirical qualification
remains deferred under issue #11984; numerical gates do not establish it.

## Implementation Review: First Independent Pass

### Follow-Up on the Consistent QP Candidate

The subsequent uncommitted equality-minimizer candidate was checked specifically
for the reported gear-sign regression. For a +12 mm toe strike, default driver
geometry with a flat face produced +10.1024° native draw tilt, +0.3515° rightward
launch, and −7.802 m lateral carry. With 0.3 m bulge/roll radii, the same strike
produced −3.14854° tilt, +2.21758° rightward launch, and +9.57349 m lateral carry.
Bulge radius 1 m restored +6.1775° tilt; radius 10 m gave +9.7137°. Heel results
mirrored. This demonstrates competition between gear response and local face
orientation, not a reversed flat-face gear sign. A test requiring net draw for
every convex face is not a valid universal physical contract.

However, the consistent QP law leaves tangential post-contact slip even when
the friction cone is inactive. It materially changes gear strength relative to
the sticking rule. Mechanical passivity is not validation of a Coulomb grip law.
For a physical slipping/gripping primary model, an impulse-incremental contact
law with static-friction constraints and energetic restitution is preferable;
the QP remains a legitimate explicitly approximate sensitivity comparator.

With normal impulse p as the independent variable, integrate
`dc/dp = −K(n + T q)`. Sliding uses `q = mu c_t/||c_t||`. At zero tangential
slip, the sticking derivative is `q = −(T^T K T)^-1 T^T K n`, if its norm is
within the friction limit. Resolve slip-stop and maximum-compression events.
An energetic restitution parameter may specify rebound normal work as e² times
the stored compression work; frictional work then makes the total energy loss
nonnegative. This e is an energetic restitution coefficient, not a guarantee
of normal velocity ratio e in every coupled impact. The uncoupled limit recovers
the usual ratio. Impulse-step refinement and event accuracy require new tests.
Do not silently change the meaning of restitution or loosen the energy gate.

Reviewed the new `rigid_contact.py` implementation introduced at `d9698cf` and
the current `club_geometry`, `matched_config`, `matched_physics`,
`nominal_matching`, and `noise` modules. This pass is **not yet accepted for the
large matrix**. The blocking items below were sent to their implementation owners.

| Finding | Independent Evidence | Disposition |
| --- | --- | --- |
| Stick-First Is Not Global Maximum Dissipation | With n=(1,0,0), r_c=(0.04,0.02,0.01) m, I_c=diag(0.00012,0.00018,0.00020) kg·m², m_c=0.2 kg, c=(45,3,1) m/s, e=0.78, μ=1, the returned sticking impulse has an energy objective 0.16905438 J above an admissible interior minimizer. | Choose a consistently declared law. The source currently claims maximum dissipation only for the sliding branch; an overall QP claim would be false. |
| Valid Elastic Case Rejected by Stick-First | The same geometry with e=1 and c=(45,−3,−1) m/s causes `contact passivity postcondition failed`. | Do not weaken the energy guard. Correct the chosen law and add a regression before matrix acceptance. |
| Admissible Sliding Case Fails SLSQP | One of 100 deterministic random 3D contact cases failed `sliding contact optimization did not converge`; exact state sent to the owner. It used e=0.83 and μ=0.02. | Add the reproducer and improve numerical robustness; do not count optimizer failure as physical unsupported-shot probability. |
| Local Impact Orientation Missing From Output | The adapter returns center-face angles. Driver high-face offset 6.35 mm changes the local normal through roll, while reported center loft remains 10°. | Add distinct local-contact face/loft fields for covariance, preserving center orientation as a separate quantity. |

For the equality-constrained energy objective, the unconstrained minimizer is
`J* = K^-1 c + [e c_n / (n^T K n)] n`. If it lies in the friction cone, it is the
global constrained minimizer without an iterative solve. It differs from the
stick-first candidate when normal/tangent response is coupled. Its final
tangential slip is not generally zero; regime naming must reflect the chosen law.

The 99 successful randomized cases included arbitrary rotations, anisotropic
inertia, nonzero initial ball translation/spin and club angular velocity, and
friction 0/0.02/0.4/1. Maximum linear momentum error was 2.17×10⁻¹⁵ kg·m/s,
angular momentum error about an offset origin 1.26×10⁻¹⁵ kg·m²/s, and normal
restitution residual 4.27×10⁻¹⁴ m/s; none increased energy. These checks support
the impulse-update algebra, while the two failures above prevent unconditional
acceptance. They are independent sampled identities, not empirical qualification.

The local parabolic contact displacement and normalized surface normal are
geometrically consistent. Face axes form a proper frame, inertia rotates with
the head, and toe-positive offset maps to negative heel-axis displacement.
Default driver checks at ±6.35 mm yielded mirrored native spin-axis tilts
±2.33087° and mirrored endpoints. The toe shot initially launches right due to
bulge and still lands 1.9603 m right while curving left; endpoint side alone is
not a gear-effect sign test. High-face launch was 9.4054° versus centered 8.0852°,
and spin 2303.7 versus 2460.2 rpm; low-face contact gave the opposite trend.
Driver static 10°/lean 10° correctly raised `UnsupportedShot(nonairborne_launch)`.
A 0.04 s horizon correctly raised a flight-time error instead of fabricating a landing.

Nominal matching uses actual face/path/speed inputs and raw endpoints; no output
rotation appears in that module. Its default speed box is ±15%, whereas the draft
study specifies ±25%; the runner must pass the declared bounds explicitly and
record them. Same-step final reevaluation is present; the independent tighter-step
replay and wider feasibility evidence remain final acceptance gates. Geometry and
correlated-noise modules are consistent with the design at source-review level.
Primitive shaft/path correlation must remain distinct from realized face/path
correlation in GUI labels and reports.

## Review Status and Scope

This is a design and source review of the shot-pattern project at initial phase-two
commit `10c7d61aa686e456b7962f1d9aa9ecfda4e8405f`, not acceptance of the new implementation.
The prior 24-cell central-contact study remains its own conditional experiment.
The proposed extension changes the collision model, orientation geometry, random
inputs, and matching criterion; prior numerical acceptance does not transfer automatically.
No production files were edited and no large flight experiment was rerun for this review.

AGENTS, CLAUDE, agent-context usage, shared infrastructure, public impact exports,
and current shot-pattern consumers were inspected. Agent-context search for impact
returned no registered boundary; direct source inspection was therefore necessary.
The Tools pin and runtime matched `3678409fc51024150ab28970b72e3b468935f345`.

## Blocking Findings and Acceptance Tests

| ID | Finding and Source | Required Acceptance |
| --- | --- | --- |
| P2-A01 | Existing off-center impact is a scalar effective-mass correction followed by additive spin. `impact_model/models.py`, `solver.py:solve_pre_impact_state`, and `utils.py:compute_gear_effect_spin` do not implement coupled rotational contact. | Use the shared impact boundary for a declared two-body contact law. Update ball and club translation and rotation from the same impulse. Verify full momentum, contact restitution, passivity, and rotation covariance before strike Monte Carlo. Do not add the old gear term afterward. |
| P2-A02 | Toe-positive helper spin has the wrong horizontal sign for the experiment's native Magnus convention. At +0.00635 m and 45 m/s, the helper gives world-z spin −50.00625 rad/s; its cross product with forward velocity points world-right. Existing `test_impact_physics_value_assertions.py` calls negative z “draw,” thereby preserving the error. | A flat-face, rearward-CG toe strike must produce the expected draw spin in the native frame, with heel mirrored. Test spin-to-aerodynamic-force and actual flight signs; do not use assertions that merely copy implementation signs. Distinguish initial launch direction from curvature. |
| P2-A03 | Impact offset is inconsistently described as from face center and from CG; scalar MOI supplies no CG depth or tensor orientation. | Define a rigid club frame, face-center-to-CG vector, toe/high axes, full positive-definite inertia tensor in that frame, and transformations to world coordinates. Test rotations, depth-zero limit, high-inertia limit, and dimensions. Clearly identify assumed values. |
| P2-A04 | `delivery_from_face_angle` samples a requested face error and inversely solves twist. It is not Gaussian shaft twist. | For shaft-rotation SD, draw twist directly and compute face/loft from the rotated normal. Keep separate variable names, units, receipts, and plots for twist SD and resulting face SD. Test the zero-error limit and local Jacobian. |
| P2-A05 | Existing shaft-lean control holds nominal delivered loft fixed and changes only the assumed axis. It is not forward pitching a club of fixed static loft. | Rigidly rotate face, shaft, CG vector, inertia, and face tangent axes together for 0°/5°/10° lean. Then apply nominal yaw and stochastic twist in a documented order. Confirm zero-yaw, zero-twist delivered loft equals static loft minus forward pitch. |
| P2-A06 | Current `core.py` rotates completed landings with `_rotate_to_aim`. That violates the new request for matched nominal delivery without post-flight adjustment. | Solve actual nominal delivery inputs against target x/y, then freeze them. Export and score raw simulated endpoints. Assert no per-pattern output rotation, translation, or rescaling is used in primary phase-two results. Keep any equal-range diagnostic separate. |
| P2-A07 | Low-loft/high-lean cases may have no airborne solution. Current configuration also rejects nominal loft ≥45°, excluding a 46° static PW before geometric pitching. | Add explicit static-versus-delivered input contracts, high-loft tests, bounded target feasibility, and nonairborne/timed-out statuses. Never silently widen bounds, alter attack, resample failed shots, or report an airborne final sample as a landing. |
| P2-A08 | Face-only scatter is not a probability-density depiction; adding two strike inputs changes dimension but does not guarantee a Gaussian landing population. | Export empirical equal-area density or probability-per-bin with shared bounds and normalization, plus transparent sample scatter if useful. Retain every shot in accounting; disclose plotted clipping and invalid mass. Label covariance ellipses as descriptive, not calibrated player confidence regions. |

## Contact Mechanics Design

Let impulse J act on the ball and −J on the club. Let r_b and r_c be center-to-contact
vectors, with ball vector r_b = −R n and normal n directed from club into ball. Use
world-frame inertia tensors and contact relative velocity

`c = v_c + omega_c × r_c − v_b − omega_b × r_b`.

The effective inverse mass is

`K = (1/m_c + 1/m_b) I − [r_c]× I_c^-1 [r_c]× − [r_b]× I_b^-1 [r_b]×`,

so `c_post = c − K J`. Both rotational updates follow the same contact impulse:
`omega_b_post = omega_b + I_b^-1 (r_b × J)` and
`omega_c_post = omega_c − I_c^-1 (r_c × J)`.
Positive masses and positive-definite inertias make K positive definite.
The off-diagonal terms couple normal and tangential response: computing normal
impulse first and independently clipping a tangential vector is not generally valid.

A full sticking candidate solves `K J = c + e (c·n) n`. Accepting it requires
nonnegative normal impulse, friction-cone admissibility, and energy consistency.
Do not assume an arbitrary Newton-restition/friction combination is passive.
One stable alternative is a *declared maximum-dissipation impulse approximation*:
minimize `0.5 J^T K J − c^T J`, subject to
`n^T K J = (1+e) c_n`, `J_n >= 0`, and `||J_t|| <= mu J_n`.
The pure-normal impulse is feasible and non-energy-producing for 0≤e≤1; the
minimizer therefore cannot have a larger energy change. This is an explicit
impact law, not exact time-resolved Coulomb slip history. With coupled K its
constrained minimizer need not produce zero final tangential slip, so it must
not be called a sticking solver merely because the cone constraint is inactive.
Choose and document one law rather than silently mixing incompatible criteria.
An impulse-integrated compression/restitution friction model is another option,
but is a materially larger validation task.

Tests must include arbitrary 3D orientation, nonzero pre-impact angular velocities,
noncentral lever arms, anisotropic inertias, friction zero/low/high, restitution
zero/one, zero and separating approach, and solver convergence/residual checks.
Use consistent center positions to test total angular momentum about an arbitrary
origin. Test total kinetic energy including club and ball rotation. Add central
limits, frame-rotation covariance, and toe/heel/high/low symmetry with the declared
face geometry. High-face driver contact should reduce backspin in the appropriate
rearward-CG reference case; do not demand this as a universal statement independent
of face roll, attack, and CG position.

Cross and Nathan experimentally connect gear effect to tangential contact motion
and friction during eccentric impact. Their experiments support the mechanism,
not these proposed club parameters or a calibrated golf-shot predictor.
[Cross and Nathan, 2007](https://www.physics.usyd.edu.au/~cross/PUBLICATIONS/37.%20GearEffect.pdf).

## Static Loft, Lean, and Shaft Twist

Treat static loft as the reference head geometry, not as delivered loft. Starting
with a square face and a zero-forward-lean shaft, positive world-Y pitch in the
native x-forward/y-left/z-up frame lowers face loft. It also changes shaft
elevation. The rigid pitch angle and projected forward shaft lean agree in this
specified square reference; general yawed configurations need their own frame
definition. Apply the same rotation to all body-fixed quantities, including inertia.

If R_y is pitch and R_z is nominal yaw, one consistent sequence is first
`R_z R_y` on the reference body, followed by Rodrigues twist about the resulting
shaft direction. Rotation order is significant. Shaft twist is an orientation
perturbation here, not a prediction of torsional spring dynamics or shaft angular
velocity. A static angular error does not supply an impact angular rate.

For the earlier zero-lean geometry, `dF/dtheta = −sin(lie)` at square nominal face.
Direct calculation at 24° loft gives face changes −0.85094°, −0.88946°, and
−0.89729° for a +1° shaft twist at lies 58.5°, 63°, and 64°. A one-degree twist
distribution therefore cannot be silently substituted for one-degree face SD.
With lean, the sensitivity changes further. Record actual delivered face, loft,
shaft direction, twist, and path for every shot.

Published shaft work distinguishes grip orientation and shaft deformation;
this supports describing rigid shaft rotation as a geometric hypothesis, not
measured player covariance or a substitute for shaft dynamics.
[MacKenzie et al., 2018](https://people.stfx.ca/smackenz/Publications/MacKenzie%202018%20The%20influence%20of%20golf%20shaft%20torque%20on%20clubhead%20kinematics%20and%20ball%20flight.pdf).

Static presets of driver 10°, 7-iron 34°, and PW 46° have primary manufacturer
support. The cited GT2 offers a 10° option; historical T100 specifications give
34°/46° and lie 63°/64°. These do not source head mass, CG, full inertia, friction,
COR, or delivered dynamics, nor establish a complete GT2/T100 model.
[Titleist GT2](https://www.titleist.com/golf-clubs/drivers/gt2),
[Titleist Historical T100](https://www.titleist.com/golf-clubs/irons/720-t100).

The face should initially be described as flat if bulge and roll are omitted.
That assumption particularly limits driver mishit claims. McNally et al.'s study
explicitly treats CG depth, inertia, and face bulge together; it is a suitable
reference for parameter sensitivity, not empirical validation of this addon.
The publisher full-text endpoint returned HTTP 403 during this review; its
abstract and authorship were verified through publisher search and the authors'
university record. No uninspected table values were adopted.
[McNally et al., 2016](https://doi.org/10.1016/j.proeng.2016.06.329),
[University of Waterloo Record](https://uwaterloo.ca/motion-research-group/references/effect-clubhead-inertial-properties-and-driver-face-geometry).

## Nominal Matching and Feasibility

For each club/lean stratum, use the central, zero-error straight endpoint as its
initial target. Preserve intended face-to-path offsets (0, ±1.5°, ±3°) and solve
nominal yaw and speed against target x/y. Path must change consistently with the
specified offset. Freeze the calibrated input before drawing the stochastic sample.
Record objective residuals, bounds, convergence, chosen branch, and all adjusted
delivery values. Speed variation across shapes is part of this estimand: it asks
about matched-range delivery, not the isolated effect of curvature at equal speed.

Primary within-lean matching and cross-lean common-target matching answer different
questions and require separate tables. Use declared speed bounds, such as the
proposed ±25% of baseline, without silent widening. A numerical optimizer success
flag is insufficient: independently replay each zero-error nominal solution and
verify physical validity and target residual with a tighter integration step.
Matching residual should be small relative to endpoint differences and numerical
error; choose and record that tolerance before runs.

At driver static loft 10° and forward lean 10°, square delivered loft is zero.
With the proposed negative attack angle, the central impulse can produce a
nonairborne launch. Speed cannot generally repair a negative launch direction.
Retain this as an infeasible condition if confirmed. A 12° driver is another
equipment hypothesis, not a legitimate silent repair, and still may produce
invalid Gaussian tails. Nonairborne or outside-model outcomes must count in the
population. Do not resample them or conceal their probability. If resting position
requires ground physics, whole-population SG is unavailable; conditional airborne
statistics may be reported only with their condition and denominator prominent.

## Sampling, Covariance, and Statistical Reporting

Use heel/toe and high/low strike SD of 0.25 inch = 0.00635 m **per component**,
not radial SD. With independent equal Gaussian components the radial RMS is
sqrt(2) times that value. Define zero relative to face center and transform the
contact offset to the CG lever vector. A Gaussian distribution has unbounded
support: declare finite face extents or an infinite-plane approximation. Outside
face/ground-invalid events must not disappear through clipping or resampling.

Use separate seeded standard-normal columns for twist, toe offset, and high
offset. Reuse the same rows across compared patterns; do not reuse a single
random column for all variables. In the absence of measured covariance,
independence is an explicit assumption. Do not manufacture correlations from
“realistic golfer” intuition. A user-specified correlation matrix must be symmetric
positive semidefinite, with matching units/order and deterministic edge-case tests.

Report both input and output sample covariance/correlation matrices, including
the deterministic face/loft relationship induced by twist. Include twist, strike
coordinates, delivered face/loft, launch angles, speed, spin-vector components,
carry/downrange/lateral, and SG where valid. Constants should have undefined/null
correlation, not zero, and singular deterministic relations are not errors.
Angles require circular treatment if any range can wrap; current small local
delivery angles should explicitly remain in a nonwrapping domain.

Use paired row resampling for differences, preserving the full random-input row.
Report target bias, covariance, lateral SD, radial RMSE/quantiles, nominal residual,
carry, and SG together. A narrower projection does not imply better target accuracy
or scoring. Separate centered-strike, strike-only, twist-only, and combined cases
to reveal interaction; variance components need not add under nonlinear impact.
Monte Carlo intervals exclude uncertain CG/inertia/contact parameters and physical
model error. Parameter sensitivity is a separate experiment, not another bootstrap.

For density plots, use common spatial bins and color normalization for patterns in
one comparison; annotate units as shots/m² or probability/m², not arbitrary opacity.
If displaying contour regions, derive enclosed empirical probability and show actual
coverage rather than calling a Gaussian covariance ellipse a 95% landing region.
Keep equal axis scales for geometric dispersion. Singular/near-singular populations
need explicit handling; KDE must not turn a one-dimensional locus into fake precision.

## Scoring and Release Gates

Reuse the V2 unholed-putting baseline and exact public-API cache; preserve the ≥1
unholed bound and direct parity. Driver remains tee/fairway/rough, irons/PW remain
approach/green/rough. Nominal range matching does not make expected SG identical:
the full endpoint distribution and lie boundary still matter. Carry is still being
treated as final rest, so realistic wedge stopping, rollout, hazards, and turf
interaction remain outside scope. Trackman's club-data definitions also refer to
the impact-location orientation near maximum compression, whereas this model is
an instantaneous rigid pre-impact construction; avoid claiming direct measurement
equivalence. [Trackman Club Data Definitions](https://www.trackman.com/blog/club-data-definitions).

Before the final large matrix: pass meaningful impact/geometry/matching tests;
complete deterministic feasibility; run a small end-to-end matrix with invalid
cases; benchmark the new collision path; check flight-step refinement on the
expanded launch/strike domain; then freeze source/native/dependency hashes.
Revalidate resume manifests against scientific inputs. Preserve all prior-phase
artifacts and their original provenance. Export invalid counts, random-input
digests, matching receipts, units, head assumptions, and scoring-source hashes.
Only after those gates should the new results and shareable graphics receive a
second independent review. Empirical accuracy remains unqualified throughout.

## Frozen-Source Preflight Disposition

The deterministic preflight executed against source commit `ce60b19a90` with both
receipts confirming unchanged scientific sources. The 243-delivery grid yielded
225 supported cases and 18 consistently nonairborne cases at all resolutions.
There were no solver errors or support-status inconsistencies. Maximum Euclidean
landing changes were:

| Comparison | Maximum Change | Predeclared Gate | Disposition |
| --- | ---: | ---: | --- |
| Contact 128→256 | 0.000014892 m | 0.05 m | Pass |
| Contact 256→512 | 0.000006701 m | 0.05 m | Pass |
| Contact 128→512 | 0.000020123 m | 0.05 m | Pass |
| Flight dt 0.02→0.01 s | 0.036837 m | 0.05 m | Pass |
| Flight dt 0.01→0.005 s | 0.010916 m | 0.05 m | Pass |
| Flight dt 0.02→0.005 s | 0.042923 m | 0.05 m | Pass |

The proposed 128 contact steps are accepted for this sampled geometric domain.
The largest flight differences occur in very low driver launches at 10° lean.
These checks do not establish bounds for every Gaussian tail or uncertain head
parameter; numerical failures must still abort final execution.

Of 90 bounded nominal cases, 72 matched and every fine replay remained within
0.002184 m of its target. Ten driver/10° lean cases were nonairborne. Eight failed
the stated bounded search: driver/5° lean across-lean ftp ±3° (0.635088 m residual),
PW/0° lean within- and across-lean ftp ±3° (0.351838 m), and PW/5° lean across-lean
ftp ±3° (0.103616 m). Retain these unavailable cases explicitly. Do not widen
bounds, adjust aim after flight, or describe optimizer failure as a global proof
of physical impossibility. The matching target, search residual, and fine replay
numerical displacement are separately recorded.

The independent 100-state contact audit was rerun with frozen provenance and
passed its declared invariant gates with no failures. Both raw receipts and the
derived concise summary are under `validation/`. Reproduction commands are in
that directory's README. This accepts the numerical preflight, not empirical
accuracy, production-population completeness, final scoring, or final graphics.
The end-to-end pilot and final result audit remain open gates.

### Receipt Hashes

- `independent_contact_audit.json`: `ca4c5b91d3113d55677ab8923af8cf567077d9a0493963aa8280e3d7bf72ca0e`
- `independent_preflight.json`: `b472dd91bfea058737f6bc441f29d903bc2718d4354438ba65d3a2c9ff108ed9`

## Head-Assumption and Contact-Law Sensitivity

A further frozen-source deterministic study held delivery fixed (no rematching):
three clubs, lean 0/5/10°, and center/toe/heel/high/low strikes at 12.7 mm. One factor
at a time changed CG depth or all principal inertias by ±20%, driver bulge/roll
radii together by ±20%, or replaced the energetic incremental law with the
Newton-QP comparator. The same numerical restitution 0.83 has different physical
semantics in the two laws. This comparison is not an isolated measurement of
friction-law uncertainty.

All 300 evaluations completed: 268 supported and 32 consistently nonairborne,
with no numerical failure or support-status change between nominal and variant.
Both contact and flight source hashes remained unchanged. The largest endpoint
displacements over this deterministic grid were:

| Club | CG Depth ±20% | Principal Inertias ±20% | Driver Radii ±20% | Newton-QP Comparator |
| --- | ---: | ---: | ---: | ---: |
| Driver | 5.144 m | 7.105 m | 22.628 m | 11.331 m |
| 7-Iron | 0.543 m | 1.764 m | Not Applicable | 1.224 m |
| PW | 0.184 m | 1.402 m | Not Applicable | 0.325 m |

The largest driver radius response occurs at 5° lean and a low strike, with both
radii reduced by 20%. It reflects a low-launch sensitivity, not a population
mean or a typical player result. Largest absolute spin-rate changes across the
CG/inertia/radius/law comparisons were respectively 180/225/164/394 rpm for the
driver, 97/109/not-applicable/194 rpm for the 7-iron, and
63/69/not-applicable/120 rpm for PW. Raw launch angles, spin tilt, ball speed,
and target-frame endpoints are retained for every supported case.

These changes substantially exceed numerical refinement error. They require
conclusions to remain conditional on the assumed head geometry and impact law.
The ±20% choices are deliberate perturbations, not measured parameter ranges;
the table is neither a statistical interval nor an empirical error bound and
does not include interactions among simultaneous parameter changes. A successful
large Monte Carlo run cannot remove this physical-model limitation.

Reproduce with `validation/independent_head_sensitivity.py --output` followed by
the receipt path, using the same root/runtime/environment command as the other
preflight scripts. The raw receipt is `independent_head_sensitivity.json`; the
concise extrema and their complete case states are in
`independent_head_sensitivity_summary.json`.

Receipt SHA256: `3bd6acb1118b6cbedd03171744a09a1b810f8a77cd7e4dc72c1d609181edb558`.
