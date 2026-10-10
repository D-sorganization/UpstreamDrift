# Analysis Issue Register

These are reviewable issue records requested by the user. The user explicitly
authorized the personal GitHub identity for this task: “use my personal identity -
I authorize this.” This is a task-specific exception to the earlier user-provided
GOV-1 restriction; credentials and repository settings remain unchanged.
Gemini 3.8 Flash High through `agy` is filing these records and retaining actual
issue URLs in publication receipts. No issue is claimed closed without a merged
implementing PR or an explicitly permitted disposition.

| Record | Finding | Implementation State | Acceptance Evidence |
| --- | --- | --- | --- |
| SPA-001 | Friction Changed Spin Without Tangential Translation | Corrected Locally | Three Regression Tests Failed Before the Fix; Broad Impact Suite: 117 Passed |
| SPA-002 | Sticking Impulse Assumed Infinite Club Mass | Corrected Locally | Finite-Mass Slip and Momentum Tests; Independent 3D Conservation Review: Six Passed |
| SPA-003 | Earlier Results Used the Defective Impact Model | Corrected 24-Cell V2 Matrix Complete | Four Historical Bundles Marked Superseded; All 24 Corrected V2 Manifests Verified |
| SPA-004 | Fixed Loft Omitted Face–Loft Geometry | Geometric Mode and Full Matrix Complete | 17 Geometry Tests and 24 Core/Physics Tests Passed |
| SPA-005 | Dispersion Summary Omitted Long/Short Covariance | Covariance and Report Integration Complete | Eight Tests Passed, Including Mirror and Zero-Variance Cases |
| SPA-006 | Driver Approach Scoring Did Not Match Tee Context | Tee/Approach Contexts and V2 Scoring Complete | Historical Table 9 Tee/Fairway/Rough; Dense API Parity; 24 V2 Manifests Verified |
| SPA-007 | Presets Could Be Mistaken for Measured Club/Player Data | Assumptions Qualified in Final Reports | Astra Identified Assumed 7-Iron Loft, Optimizer PW Example, and Static-Lie Assumptions |
| SPA-008 | Physics Qualification Could Be Overstated | Scope and Deferred Physical Validation Explicit | Central Contact Only; Off-Center Club Rotation and Player Validation Remain Unqualified |
| SPA-009 | Separating Contact Produced an Attracting Normal Impulse | Corrected Locally | Separating-Contact Regression Failed Then Passed; Zero Approach Remains No Impulse |
| SPA-010 | Comparison Accepted Different Delivery Baselines | Corrected Locally | Three New Tests Failed Before the Baseline Contract; Eight Comparison Tests Passed |
| SPA-011 | GUI Custom Inputs Passed a Rejected CLI Preset | Corrected Locally; Custom CLI/GUI Regressions Passed | Permit Explicit Custom Mode and Preserve User Delivery Overrides |
| SPA-012 | New GUI/Core Boundaries Failed Static Type Checks | Local Corrections and Focused Mypy Passed; Canonical Gate Pending | Guard Qt Headers; Compatible Resize Signature; Typed Physics Calls and Numeric Refinement Rows |
| SPA-013 | Export-Time Hashes Could Misidentify Loaded Experiment Sources | Matrix and Standalone Provenance Corrected; Resume Hash Checks Tested | Freeze Worker Sources Before Simulation; Verify and Link Source/Native Receipts |
| SPA-014 | Extended Scoring Table Retained the Abbreviated Baseline Version | V2 Baseline and Separate Postprocess Provenance Complete | Content SHA Is Authoritative; Version and Postprocess Receipts Must Identify Full Table |
| SPA-015 | Mirror Symmetry Was Overgeneralized to Coupled Delivery | Documentation Corrected; Astra Verified Geometry | Fixed Loft Is Symmetric; Same Right-Handed Axis Coupling Can Differ Structurally |
| SPA-016 | Putting Baseline Awarded Partial Hole-Outs at Positive Distance | Corrected; All 24 Saved-Flight Bundles Re-Scored | Unholed Expected Strokes Must Be at Least One; Terminal Holed State Must Be Separate |
| SPA-017 | Shared SG Legend Repeated Entries and Overlapped a Panel Title | Corrected; Regenerated V2 Overview | Red–Green Legend Test and Visually Verified 1080p PNGs |
| SPA-018 | Git Line-Ending Normalization Changed Evidence Hashes | Corrected and Committed | All 24 Indexed CSV Hashes Match Frozen Manifests |
| SPA-019 | Bare Test Module Name Collided During Broad Collection | Corrected and Committed | Red Canonical Collection; Green Combined Collection of Both Test Modules |

| SPA-020 | Provenance Required an Ignored Generated Lockfile | Corrected in the Canonical Run/Export Snapshot | Red Clean-Clone CLI/Export; Optional Lock Presence Explicit; Required Sources and Native Binary Retained |
| SPA-021 | Native Provenance Ignored Windows `.pyd` Layouts | Corrected and Committed | Ten Native-Discovery Tests Cover Direct/Packaged Layouts and Ambiguity |

## Regression and Resolution Requirements

SPA-001 and SPA-002 must preserve equal-and-opposite impulses, linear and angular
momentum, normal restitution, nonincreasing kinetic energy, and sticking contact
when friction permits. Initial spin must enter contact slip. Their corrections
must remain covered by independent arbitrary-3D and rotation-covariance tests.

SPA-003 requires rerunning every final comparison with corrected physics and
exact source/build provenance. Historical graphics must not appear in the final
shareable package or support corrected-model conclusions.

SPA-004 is a geometric sensitivity analysis, not a measured player covariance.
Each pattern must share its zero-error delivered loft; face error rotates around
the assumed shaft axis. Shaft-lean axis sensitivity and fixed-club pitching must
be distinguished. Report actual loft and carry; do not assume less loft always
increases carry.

SPA-005 requires descriptive endpoint covariance, signed lateral/carry
correlation, target-error percentiles, and explicit quadrant conventions. A
face-only random experiment does not qualify a two-dimensional golfer error
distribution or confidence ellipse.

SPA-016 additionally requires the unholed putting lower bound of one; API
parity alone is insufficient to validate baseline physics.

SPA-006 requires driver tee scoring and iron/PW approach scoring. Do not replace
a tee starting baseline with a fairway baseline. Unsupported baseline distances
must be unavailable rather than extrapolated. Any exact cache must reconstruct
public-API outputs, pass parity tests, and disclose API evaluations separately
from shots scored.

SPA-007 and SPA-008 are resolved by truthful assumptions, bounds, and deferred
validation, not by inventing measurements. Numerical tests establish internal
consistency within the stated model; empirical accuracy remains unestablished.

## Published Issue Links

SPA-001–014: #11840–11853. SPA-015: [#11862](https://github.com/D-sorganization/UpstreamDrift/issues/11862). SPA-016: [#11863](https://github.com/D-sorganization/UpstreamDrift/issues/11863). SPA-017: [#11864](https://github.com/D-sorganization/UpstreamDrift/issues/11864). SPA-018: [#11865](https://github.com/D-sorganization/UpstreamDrift/issues/11865). SPA-019: [#11930](https://github.com/D-sorganization/UpstreamDrift/issues/11930). SPA-020: [#11936](https://github.com/D-sorganization/UpstreamDrift/issues/11936). SPA-021: [#11937](https://github.com/D-sorganization/UpstreamDrift/issues/11937). Exact receipts are in `publication_receipt.json`.

## Publication and Closure

Publish these records under the expressly authorized identity, check for existing
duplicates, and link their regression evidence and implementing PR. Keep issues
open until a merged PR implements their acceptance criteria, or an explicitly
permitted disposition applies. Physical measurements are deferred validation,
not a completed simulation acceptance claim.

## Separate Baseline Release Follow-Ups

These are not model-analysis defects and are not claimed fixed by this feature:

- [Drake Offscreen Native Abort #11934](https://github.com/D-sorganization/UpstreamDrift/issues/11934): isolated reproduction aborts in unchanged Drake/Qt widget construction.
- [Development-Log Policy Debt #11935](https://github.com/D-sorganization/UpstreamDrift/issues/11935): unchanged shared log fails field-provenance, WIP and size policies. Other live owners' state must not be rewritten to make this feature gate green.

The ready PR carries a merge hold while baseline validation remains unresolved;
no protection bypass or false full-suite pass is permitted.

| FOLLOWUP-003 | [#11940](https://github.com/D-sorganization/UpstreamDrift/issues/11940) | Unchanged Research Atlas Provenance | Open Baseline Blocker; Separate Governed Correction |
