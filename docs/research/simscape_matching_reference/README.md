# Simscape Matching Reference

Start with the [standalone LaTeX reference](simscape_matching_reference.tex). It
documents the model ladder, coordinate frames, joint conventions, capture
processing, geometry and inertia assumptions, inverse kinematics, controllers,
contact diagnostics, qualification gates and reproducible exports.

The [refinement record](MATCHING_REFINEMENT.md) and
[current turnover](../../development/HANDOFF.md) identify tested experiments,
rejected candidates and outstanding work. The engine's
[reference pointer](../../../src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/docs/REFERENCE.md)
links here so the modeling account has one maintained LaTeX source.

This is a separate research and experiment record. The canonical engineering
design manual remains governed by the QMD sources under
[manuals/upstreamdrift](../../../manuals/upstreamdrift); its generated artifacts
must not be edited directly. The repository's
[modeling documentation policy](../../../AGENTS.md) requires source references,
equations, assumptions, units, reproduction steps and evidence whenever the
modeling process changes.

Capture A is the public tour reference; capture O is the owner evaluation.
Avatar distortions cannot establish playing ability. Current exports use IK
poses and are labeled **IK / DYNAMICS UNQUALIFIED**. Historical feedback-assisted
simulations and new IK exports must be identified separately. Issues #11156,
#11160 and #11173 remain open scientific gates; successful software execution
does not establish physical qualification.

Future shareable matches use the Human ellipsoid model with capture-specific
longitudinal geometry and calibrated foot orientation. Artistic transverse
proportions, incomplete head orientation constraints and subject inertia
personalization remain limitations. Position RMS excludes the foot orientation
term, which is reported separately. Flat soles at address are a calibration
assumption, not measured ground contact.

The reviewed baseline is `144e81188dd7bb106f81d89b7a7330dd20cce511`.
Execution provenance must identify the actual capture and model hashes, all
solve/render dependencies, fitted geometry, sampling and native receipt. A
scratch mirror may report an unknown Git commit; exact file hashes must then
identify its executed inputs. Public records use capture aliases and hashes;
raw private captures, absolute private paths and pose caches remain outside Git.

The `.tex` source is standalone and can be opened in Codex's LaTeX editor or
compiled with an existing LaTeX installation:

```sh
pdflatex -interaction=nonstopmode -halt-on-error simscape_matching_reference.tex
pdflatex -interaction=nonstopmode -halt-on-error simscape_matching_reference.tex
```

Structural documentation checks:

```sh
python3 -m pytest -q docs/research/simscape_matching_reference/test_simscape_matching_reference.py
```

These checks verify source presence, standalone structure, the governance
notice and obvious private-path leakage. They do not validate equations or
physical claims. Native Simscape probes, mathematical tests, compiled PDF
inspection and complete video decoding provide distinct evidence.

## Current Evidence and Selection

The selected Desktop package is `Best_Human_Matches_20261002.zip`. Tour clips
retain the earlier Human model (13.188/40.121 mm mean/max frame position RMS);
owner clips use the current model (16.643/36.865 mm). Each capture records its
own model/runtime identity. The owner foot-orientation changes are small and
mixed. These are exploratory position-tracking selections, with no anatomical
or physical acceptance claim.

| Evidence                                                       | What It Establishes                                                                                      | Remaining Boundary                                                        |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| [Current Exports](main_video_verification_20261002.json)       | Native receipts, separate position/foot metrics and complete MP4 decoding                                | IK visualization only                                                     |
| [Runtime Comparison](capture_runtime_comparison_20261002.json) | Canonical SI positions, validity masks, labels and rates agree under recorded importers                  | Full solver/runtime equivalence unresolved                                |
| [Native Helper Review](native_helper_review_20261002.json)     | 18 pose-mapping and 8 cluster tests; both native target-expression/FK roundtrips pass after mask refresh | Priorities, controller references and physical initialization unqualified |
| [Head-Track Audit](head_track_audit_20261002.json)             | Cluster data available in both captures, with explicit pair-distance variation                           | Body-frame calibration and head-constrained fit under review              |

The controlled tour comparison used current source/runtime with both model
binaries and produced identical poses. It does not explain the difference
from the earlier selected execution. All physical gates remain open.

The latest `.tex` source remains uncompiled: the built-in compiler reports
`Unable to find standard directories for platform`. The previously reviewed
16-page PDF is a prior revision and does not validate these newer sections.

### Reviewed Native Helper Checkpoint

The combined suite passed 213 MATLAB R2025b checks with zero failed or
incomplete tests; the serialized process exited naturally with code zero at
10:27:31Z. The 18 mapping and 8 cluster tests are included in that total.
See `native_helper_integration_tests_20261002.json` in the research reference
directory. These are software/parameter checks, with no physical acceptance.

The unfinished head prototype was withheld after review found insufficient
gap/coverage validation and an unproven baseline-preservation claim. Its
source, tests and partial execution evidence are retained privately for
continuation; the baseline IK source was restored before the combined check.
No head-constrained candidate or new selected video is claimed. Protected CI
still requires a fresh run after regenerating the monolith register. The
leaderboard runner's missing local action remains unexplained: its checkout
log already records sparse-checkout disable, so an additional cleanup patch
was not accepted on the proposed explanation alone.

### Head and Quiet-Reference Review Checkpoint

The reviewed head residual shares the normalized SO(3) chordal helper with
feet and validates complete head sequences and masks before model setup.
The combined MATLAB R2025b suite passed **267 checks**, zero failed or
incomplete, with natural exit zero at **12:34:19Z**. Two added quiet-reference
guards first failed (19 passed, two failed), then passed in the combined
suite: integer right-ankle inputs preserve other leg fractions, and malformed
neck-unit metadata raises the contract error. Neck prescribed profiles use
**radians**; native initial neck targets use **degrees**. Zero feedforward
does not establish gravity compensation or equilibrium.

The corrected four-frame A/O native comparison retains personalized geometry
and preserves original joint/position outputs exactly when head tracking is
disabled. An earlier private probe lost seven owner geometry values across
an unsaved close/reload boundary and is excluded from selection evidence.
Both tour head candidates failed screening. Owner weight 0.03 passed only
the coarse screen, with left-foot maximum worsening 1.9971 degrees.

The completed **46-frame owner** cold-start, fixed-offset comparison at weight
0.03 improves mean/max head error from 75.517/129.297 to 27.456/54.163 degrees,
and mean/max position RMS from 38.524/81.119 to 26.612/37.606 mm. All native
statuses are one and the process exited naturally at **12:33:07Z**. It does
not replace the selected 16.643/36.865 mm owner clip. Fixed-offset solving
skips calibration's fitted starting pose; keyed warm-start review is ongoing.

Current owner FK contact-sphere clearances are **-64.462 to -48.262 mm**, a
16.199 mm spread, under the validated stored native ground transform. That
audit exited naturally at **12:06:25Z**, without simulation or model save.
The ground transform is parameter-derived, not a KinematicsSolver output.
The assembled 20 ms diagnostic exited naturally at **12:38:27Z**. Scalar
joints at t0 match requested targets (zero translation error, maximum scalar
rotation error 5.43e-9 degrees); all five spherical rotation matrices also match (maximum matrix error 1.34e-10, native postprocess natural exit zero at 12:47:09Z). The contact clearance mismatch survives actual assembly. Initial
left/right normal force is 32.205/24.407 kN, falling to 106.684/0 N at 20 ms;
maximum pelvis displacement is 38.370 mm. This uses constant references,
stored primitive flags, upper tracking enabled and balance correction disabled.
It does not qualify equilibrium or open-loop full-swing dynamics. No force
magnitude or collapse prediction is accepted from FK alone.

See [Sanitized Review Evidence](head_and_quiet_review_20261002.json) and
[Combined Native Test Inventory](native_head_and_quiet_integration_tests_20261002.json).
Desktop selections are unchanged. All physical gates remain open. Latest
LaTeX compilation still fails with `Unable to find standard directories for platform`;
the prior PDF does not validate the new sections. Protected CI and review remain required.

### Keyed Seed and Contact Checkpoint

The keyed initial-pose contract and integration passed **275 native MATLAB
R2025b software checks**, with zero failed or incomplete tests and natural
exit zero at **13:08:09Z**. The full output structure, with explicit Human
selection, matched the frozen pre-extension solver at three sampled frames
for each capture. The full seeded comparison completed naturally at
**13:26:50Z**, all 55 tour and 46 owner native statuses one.

At head weight 0.03, tour mean/peak position RMS is **13.349/42.116 mm**,
head post-address mean/peak **27.331/50.315 degrees**; owner position RMS is
**16.596/36.746 mm**, head **27.667/54.064 degrees**. Both missed the
prospective 30% head-improvement criterion. Tour peak position RMS also
exceeds the selected Desktop clip by 1.995 mm. **Neither is selected.**
Further weights 0.06 and 0.10 are a separate prospective experiment with
unchanged acceptance limits against the same-source seeded baseline and
selected clips. Current native loop success does not establish physical
acceptance or anatomical calibration.

The tangent-plane contact diagnostic completed naturally at **13:30:55Z**.
It changes only the ground normal translation by **-64.462 mm**, retaining
the actual assembled pose. Native clearances become **0 to 16.199 mm**.
Initial left/right normal forces are **0/0 N**, final 20 ms forces
**588.495/0 N**, maximum pelvis displacement **1.505 mm**. Upper tracking
remains enabled, balance correction disabled and upper feedforward zero;
this is a unilateral minimum-touch diagnostic, not bilateral equilibrium
or open-loop replay. Both-foot support, capture-consistent geometry/pose,
COM/balance references and gravity-support torques remain required.

See `seeded_head_and_contact_review_20261002.json` in the standalone research
reference directory for sanitized aggregates, direct source inventory and
actual receipts. The updated LaTeX source remains in the same editor; its
built-in compiler still reports `Unable to find standard directories for platform`.
No new rendered PDF is claimed. CI Standard run **37010261028** on published
79d3b97 failed MyPy/core tests; exact-log review is active, protected review
remains required. The Human-default migration is a separate policy change
under source review, not covered by explicit-Human parity. Physical gates
#11156, #11160 and #11173 remain open.

### Human Default and Reviewed Head-Tracked Delivery

New direct whole-body IK calls default to **GS3DX_Human**. Historical Fit
builders, neck-injection harnesses and reproduction examples explicitly
select Fit. This intentional policy change leaves solver mathematics
unchanged. Native RED exposed the wrong default and Human-only seed
rejection; native GREEN passed both policy tests at **13:58:47Z**. The full
reviewed suite passed **277 software/parameter checks**, zero failed or
incomplete, with natural exit zero at **14:07:07Z**. These do not qualify
physical replay or anatomical calibration.

The prospective **head weight 0.10** screen passes both captures. Mean/peak
body RMS is **13.540/40.368 mm tour**, **16.754/36.987 mm owner**. Mean/peak
post-address head error is **10.671/18.918 degrees tour**, **11.318/22.169
degrees owner**; mean head error improves 69.8%/68.8% against their seeded
weight-zero baselines. Each foot remains within the recorded peak limits.
Tour 0.06 fails the peak body-RMS criterion against the prior Desktop clip.

Both-view rendering completed naturally at **14:02:59Z** after preserving
the first attempt's failed provenance-write receipt. All four H.264 MP4s
fully decoded at 800x600, 30 fps, with 55 tour and 46 owner frames. Every
sampled frame was reviewed in ordered contact sheets for both views; no
obvious projected limb/head flips or scene clipping were observed at that
scale. The new selections are saved on the **local user Desktop** in
`Best_Human_Matches_20261002_HeadTracked`, with a matching ZIP and sanitized
provenance/verification. ZIP SHA-256:
`48c34d60a50610879f9f9f12ee90526c9fe091d45c72dc1d1c7744cb93089108`.
Earlier packages are preserved. These are IK videos; the floor is decorative,
head targets cluster-relative, and contact/balance/forward dynamics remain
unqualified. The videos cannot establish player skill.

See `head_weight_followup_review_20261002.json`,
`human_default_policy_review_20261002.json`,
`native_human_policy_integration_tests_20261002.json` and
`desktop_head_tracking_delivery_20261002.json` in the research reference.
The same LaTeX source/editor is updated; built-in compilation remains
unverified with the platform-directory error. CI run 37010261028 failed
dispatch-context coverage/base-ref checks and a whole-repo MyPy baseline;
all 2,563 executed core tests passed. Source and workflow-context review
does not establish passing protected CI. Current main reconciliation,
fresh checks and protected review remain required. Bilateral contact,
capture-consistent balance/gravity support and full native dynamics remain
active requirements; physical gates are open.

### Leg Orientation Contract and Selected Contact Geometry

The analytical leg IK previously accepted a 180-degree orientation mismatch because
its skew residual vanished. Native R2025b TDD reproduced false success (RED: one
pass, five failures, zero incomplete), then passed all six contract tests after
reusing the shared SO(3) chordal residual with a 12x6 Jacobian and separate final
position/orientation bounds. The existing welded-foot native FK test, reachable
IK and trajectory tests also pass. The combined suite passed **287 native
software/parameter checks**, zero failed/incomplete, natural exit zero at
**2026-10-02T14:57:32Z**. This does not qualify Human ankle-to-foot-solid frame
correspondence, anatomical limits, or independent forward dynamics.

Fresh native FK evaluated BOTH promoted head0.10 addresses without simulation
or model save. Per-foot lowest-contact heights differ **29.081 mm tour** and
**15.097 mm owner**; within-foot spreads are below 1.5 mm. With the ground normal
fixed, one plane translation cannot remove this two-foot discrepancy. The bounded
native correction passes both captures: each minimum clearance 0.250 mm, sole spreads below 1.5 mm, foot XY/orientation retained, leg rotations at most 8.731 degrees tour / 4.229 degrees owner. Root/upper coordinates, passive midfoot and fitted geometry are unchanged. The accepted address candidates are staged for assembled-state/contact diagnostics;
equilibrium, gravity torques and forward replay remain open. The selected Desktop
MP4s are unchanged IK visualizations. See the standalone LaTeX reference and
`leg_orientation_contract_review_20261002.json`,
`selected_contact_geometry_review_20261002.json`, and
`native_leg_contact_integration_tests_20261002.json` for equations and receipts.

Latest LaTeX source is maintained in the same editor. Built-in compilation still
fails with `Unable to find standard directories for platform`; no new PDF is
claimed. Changelog duplicates for PR #11256 were consolidated. Subsequent CI
code-quality failed a GitHub fetch because of runner certificate verification;
no certificate validation was disabled and protected review remains required.

### Bilateral Address and Assembled Gravity Diagnostics

Both selected Human address candidates pass their prospectively fixed native
geometry gates. Each foot minimum clearance is 0.250 mm against one plane;
horizontal foot-solid position/orientation, root/trunk/upper coordinates,
passive midfoot coordinates and fitted geometry are retained. Maximum leg
rotation changes are 8.731 degrees tour and 4.229 degrees owner. These are
address corrections, not a new measured whole-swing fit or Desktop promotion.

Separate 20 ms R2025b simulations verify the actual assembled scalar and
spherical pose plus all ten contact clearances. Both feet develop support:
at 20 ms, tour left/right normal force is 443.969/407.322 N, owner
502.848/407.399 N. Maximum pelvis displacement is 1.513/1.482 mm. Both
native processes exited naturally with zero status; geometry/physics were
reapplied before logged-pose FK and the Human binary was not saved. The
tour body mass is model-default 80 kg, owner 104.3 kg; total mechanism
masses include unchanged equipment. Zero initial force reflects 0.25 mm
clearance. Endpoint support does not establish standing equilibrium.

A separate one-second constant-reference hold is registered before results:
each foot >=0.05 BW and summed normal force 0.8-1.2 BW over 0.5-1 s;
whole-run sum peak <=2 BW, pelvis displacement <=5 mm and rotation change
<=1 degree. It retains upper PD tracking, prescribed neck and zero upper
feedforward, with balance correction off. Even a passing hold is not
independent open-loop replay. Actual gravity and native mass define BW.

Read `bilateral_stance_geometry_review_20261002.json` and
`bilateral_assembled_contact_review_20261002.json` alongside the same
standalone LaTeX reference. Its abstract now identifies the latest head0.10
Desktop selections; historical force trials are explicitly attributed and
zero support is not treated as proof of airborne geometry. PDF compilation
remains unavailable. Full-swing physical gates and protected review stay open.

The tour one-second hold is rejected: all three force screens pass, but pelvis displacement 246.153 mm and rotation 14.989 degrees violate the fixed 5 mm / 1 degree bounds. Native process exits naturally with zero at 15:18:18Z; successful execution does not establish physical success. Owner hold and reference/torque/COM diagnosis remain separate active work. See `tour_constant_hold_review_20261002.json`.

The owner one-second hold also rejects the fixed pose limits: force screens
pass but pelvis displacement is 118.109 mm and rotation change 7.866 degrees.
Native exit is naturally zero at 15:23:43Z; final left/right forces are
531.659/499.520 N. Both captures require controller/reference/COM diagnosis;
neither hold qualifies standing stability or independent open-loop motion.
Read `owner_constant_hold_review_20261002.json`. Gates remain unchanged.

Saved native endpoint diagnosis completed naturally in R2025b at 15:54:34Z.
COM horizontal displacement is 230.802 mm (tour) and 110.766 mm (owner);
terminal projected COM lies outside the contact-point hull by 167.030 and
13.899 mm. Native ankle rotations change 12.485/12.476 degrees (tour L/R)
and 4.069/4.525 degrees (owner L/R). This is endpoint motion, not continuous
contact-slip measurement or a causal diagnosis. Initial forces are zero;
there is no initial active support hull. Native workspace and hold scripts
confirm zero leg feedforward and unchanged servo gains. A preliminary probe
rejected its incorrect rigid five-sphere constellation assumption; the
corrected probe measures ankle followers directly and preserves midfoot
articulation. No new simulation or model save occurred. See
`saved_native_hold_diagnosis_review_20261002.json` and the updated LaTeX.
Next: verify Human ankle FK/gain compatibility before a same-stance
balance-enabled hold with the unchanged registered force/drift gates.

Native Human ankle/gain interface checks passed for both exact fitted address
stances (R2025b natural exit zero at16:03:31Z). Native/analytical Jacobian
differences are below2.3e-13 m/degree and gain differences below7.5e-9 degree/m.
Same-stance balance-on holds retain all original gains, zero feedforward,
prescribed neck and fixed force/drift gates. Tour pelvis displacement/rotation
is 20.041 mm / 5.582 degrees;
owner is 14.818 mm / 4.408 degrees.
Hold screens: tour REJECT, owner REJECT.
Improvement is not physical acceptance. See
`human_ankle_gain_interface_review_20261002.json` and
`balance_enabled_hold_comparison_review_20261002.json`. The existing upper-only
learning API hardcodes FitTrack and overwrites starts; a private Gemini TDD
proposal is not accepted code or full forward-dynamics evidence.

### Configured Human Upper-Body Learning and Native Lifecycle Evidence

`gs3dx_track_learn` accepts an explicit already-loaded model with
`initialization="configured"`. This mode preserves the caller workspace and
initial-state configuration, bypasses legacy drive/TrackStart replacements,
and leaves the caller model loaded. Historical FitTrack behavior remains
available through default legacy initialization. References, gains,
feedforward, timing, filter support and enabled tracking are checked before
native dynamics. The returned feedforward is the best profile actually
simulated, with explicit upper-only qualification and first-reference-sample
state provenance. The original Q-filter learning law is retained.

Native R2025b TDD progressed from **1 pass, 13 failed, 13 incomplete** on the
old API to **19 passed, zero failed/incomplete** on the strengthened contracts.
MATLAB marks assertion-aborted RED tests both failed and incomplete. The
valid configured Human probes completed two 0.1 s constant-reference
iterations: tour angle RMS **0.104749 -> 0.021832 degrees**, PD RMS
**5.092613 -> 1.456887 N m**; owner **0.152201 -> 0.022887 degrees**,
**8.803216 -> 1.482412 N m**. Both preserved all **837 workspace values**,
including **620 independently snapshotted parameter values**, caller dirty
state and initial upper angles/rates within the registered 1e-5 bounds.
Neither saved the model binary. Parent mask velocity priority controls the
primitive target; native tracing explains why a child-only override failed.
Three unsuccessful probes and their natural-exit receipts remain recorded.

These are short upper-learning integration checks with balance/feedback,
zero leg feedforward and prescribed neck. They do not qualify full-swing
learning or independent forward dynamics. The physical hold gates still
reject both captures (tour 20.041 mm / 5.582 degrees; owner 14.818 mm / 4.408
degrees, against 5 mm / 1 degree). Separate actual servo and balance torques
and verify settling before treating any empirical leg offset as gravity
feedforward; a rejected transient is not a unique static solution.

See `configured_human_learning_review_20261002.json` and the updated LaTeX
equations/contracts/reproduction account. The combined native regression suite completed naturally at 17:26:58Z:
**311 passed, zero failed/incomplete**, including the five legacy FitTrack
checks. Its 0.3 s historical replay remained within original limits
(rounded angle RMS 0.28 degrees, worst joint 0.49 degrees, PD 16.7 N m).
See `native_configured_learning_integration_tests_20261002.json`; this
short legacy replay does not qualify the configured Human full swing. Built-in LaTeX compilation remains
unavailable: `Unable to find standard directories for platform`. PDF review,
current-head protected CI and full-body replay remain open.

### Saved Leg Servo Effort and Late-Window Motion

Read-only native R2025b analysis completed naturally at **17:47:04Z**.
The 2 ms diagnostic grid reconstructs baseline leg servo feedback from the
saved balance-enabled holds and actual gains, excluding balance correction
and total actuation. Initial reference closure is below 1e-6 degrees.
Whole-run baseline servo RMS is **226.583 N m tour**, **236.384 N m owner**.
The 0.5-1.0 s analysis window still moves: pelvis maximum displacement and
rotation relative to its first window sample are **10.439 mm / 3.147 degrees
tour**, **5.893 mm / 1.283 degrees owner**. Largest per-axis leg-rate RMS is
**7.691 / 7.275 degrees/s**. This is not verified static equilibrium; no
mean torque is promoted to gravity feedforward and the fixed hold gates
remain rejected. Large baseline effort cannot alone establish harmful
cancellation because balance deliberately shifts the servo target.

The reader loaded its own unchanged Human model only for saved block-path
resolution and closed it without saving. No new dynamics/FK ran. Two failed
path-resolution attempts are retained. Raw traces remain private; see
`saved_leg_servo_diagnosis_review_20261002.json` and the same updated LaTeX
source. Inspect nested joint datasets/controller signals to recover actual
net actuation and balance terms before a bounded support-control change.
Current PDF remains unverified because the built-in compiler is unavailable.
CI Standard run37042119516 executed published3187eccd0 and exposed an owned
duplicate #11256 SPEC row; it is consolidated and the duplicate/version
checks pass locally. Fresh current-head protected checks remain required.
