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
