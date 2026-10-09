# Analysis Issue Register

These are reviewable issue records requested by the user. GitHub publication is
pending: the CLI and connector authenticate as `dieterolson`; GOV-1 requires
the agent's GitHub App identity. No configured App key or installation-token
route was found on this host. No issue is claimed to have been filed or closed.

| Record | Finding | Implementation State | Acceptance Evidence |
| --- | --- | --- | --- |
| SPA-001 | Friction Changed Spin Without Tangential Translation | Corrected Locally | Three Regression Tests Failed Before the Fix; Broad Impact Suite: 117 Passed |
| SPA-002 | Sticking Impulse Assumed Infinite Club Mass | Corrected Locally | Finite-Mass Slip and Momentum Tests; Independent 3D Conservation Review: Six Passed |
| SPA-003 | Earlier Results Used the Defective Impact Model | Historical Bundles Marked Superseded; Replacement Pending | Status in All Four Summary, Scoring, and Receipt Files |
| SPA-004 | Fixed Loft Omitted Face–Loft Geometry | Geometric Mode Implemented; Expanded Results Pending | 17 Geometry Tests and 24 Core/Physics Tests Passed |
| SPA-005 | Dispersion Summary Omitted Long/Short Covariance | Pure Statistics Implemented; Report Integration Pending | Eight Tests Passed, Including Mirror and Zero-Variance Cases |
| SPA-006 | Driver Approach Scoring Did Not Match Tee Context | Replacement Tee Scoring Pending | Requires Source-Verified Tee/Fairway/Rough Baselines and API Parity |
| SPA-007 | Presets Could Be Mistaken for Measured Club/Player Data | Qualifications Required in Final Reports | Astra Identified Assumed 7-Iron Loft, Optimizer PW Example, and Static-Lie Assumptions |
| SPA-008 | Physics Qualification Could Be Overstated | Scope Limits Required in Final Reports | Central Contact Only; Off-Center Club Rotation and Player Validation Remain Unqualified |

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

SPA-006 requires driver tee scoring and iron/PW approach scoring. Do not replace
a tee starting baseline with a fairway baseline. Unsupported baseline distances
must be unavailable rather than extrapolated. Any exact cache must reconstruct
public-API outputs, pass parity tests, and disclose API evaluations separately
from shots scored.

SPA-007 and SPA-008 are resolved by truthful assumptions, bounds, and deferred
validation, not by inventing measurements. Numerical tests establish internal
consistency within the stated model; empirical accuracy remains unestablished.

## Publication and Closure

Publish these records under the agent bot once authenticated, check for existing
duplicates, and link their regression evidence and implementing PR. Keep issues
open until a merged PR implements their acceptance criteria, or an explicitly
permitted disposition applies. Physical measurements are deferred validation,
not a completed simulation acceptance claim.
