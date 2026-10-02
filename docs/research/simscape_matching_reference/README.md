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

| Evidence | What It Establishes | Remaining Boundary |
| --- | --- | --- |
| [Current Exports](main_video_verification_20261002.json) | Native receipts, separate position/foot metrics and complete MP4 decoding | IK visualization only |
| [Runtime Comparison](capture_runtime_comparison_20261002.json) | Canonical SI positions, validity masks, labels and rates agree under recorded importers | Full solver/runtime equivalence unresolved |
| [Native Helper Review](native_helper_review_20261002.json) | 18 pose-mapping and 8 cluster tests; both native target-expression/FK roundtrips pass after mask refresh | Priorities, controller references and physical initialization unqualified |
| [Head-Track Audit](head_track_audit_20261002.json) | Cluster data available in both captures, with explicit pair-distance variation | Body-frame calibration and head-constrained fit under review |

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
