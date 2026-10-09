# Matched Shot Distributions: Scientific Design and Verification

## Purpose and Estimands

This extension compares straight, draw, and fade patterns with a common **physical zero-error landing point**, followed by prescribed delivery noise. No simulated endpoint or trajectory is rotated to aim at the target. Nominal face direction, path, and speed are inputs solved before sampling. Matching two endpoint coordinates does not match the entire trajectory, launch state, peak height, descent angle, or energy. Curved trajectories remain different by design.

The study estimates changes in landing spread, bias, supported-shot probability, and conditional course scoring under explicit equipment and delivery assumptions. It does not identify a universal preferred shape, a measured human variability distribution, or the performance of a particular commercial club.

This is an editable research reference, separate from the governed engineering manual. Its [Standalone LaTeX Version](matched_delivery_reference.tex) is regenerated from this research Markdown with Pandoc, not from the engineering manual. The [Astra Design Review](ASTRA_REVIEW.md), [Phase-One Reference](../README.md), issue records in [issues/](issues/), executable tests, and immutable result receipts form its evidence chain. Implementation and results remain under development until their acceptance evidence is recorded.

## Experimental Controls

For each club, use static loft and lie from the cited specification, fixed attack angle, and explicit assumed head properties. Pitch the complete club by forward lean of 0°, 5°, or 10°. Then solve nominal face and speed, with path constrained by face-to-path difference: 0° for straight, −1.5° for draw, +1.5° for fade; the stronger curves use ±3°. The target line remains the fixed +X direction throughout.

Two target protocols distinguish the questions:

1. **Within-Lean Matching:** Straight, draw, and fade land at the same target for a given club and lean. The target is the zero-error straight shot at that club's reference speed and lean. Distances may differ across lean settings.
2. **Across-Lean Matching:** Attempt the same physical target distance as the zero-lean straight reference. Report all infeasible conditions with their fixed bounds and reason. Reachability is not guaranteed by an optimizer's termination.

Speed search bounds are fixed at 75–125% of the reference speed; nominal face bounds remain local to the intended shape. The accepted raw landing residual must be at most 0.05 m, checked independently after optimization. The frozen experiment plan records actual bounds and solved inputs. Infeasible cases remain in the matrix; neither attack angle nor static loft is silently changed to populate them.

## Equipment and Provenance

| Club | Static Loft | Lie at Zero Lean | Reference Speed | Attack Angle | Head Mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| Driver | 10° | 58.5° | 45 m/s | −0.9° | 0.200 kg |
| 7-Iron | 34° | 63° | 36 m/s | −4° | 0.272 kg |
| Pitching Wedge | 46° | 64° | 32 m/s | −5° | 0.300 kg |

The [Titleist GT2 specification](https://www.titleist.com/golf-clubs/drivers/gt2) provides the selected driver loft/lie. The [historical T100 specification](https://www.titleist.com/golf-clubs/irons/720-t100) provides the iron/wedge lofts and lies. These references establish geometry only. Speed, attack, mass, CG, inertia, usable face dimensions, restitution, friction, and curvature are research assumptions. The hybrid presets are **not** reconstructions or performance predictions for GT2 or T100 clubs.

| Club | CG Depth | Principal Inertias: Normal, Heel, Up | Face Half-Width / Half-Height | Bulge / Roll Radius |
| --- | ---: | --- | --- | --- |
| Driver | 35 mm | 0.00025, 0.00040, 0.00045 kg·m² | 50 / 27 mm | 300 / 300 mm |
| 7-Iron | 10 mm | 0.00012, 0.00020, 0.00025 kg·m² | 35 / 23 mm | Flat / Flat |
| Pitching Wedge | 8 mm | 0.00010, 0.00020, 0.00022 kg·m² | 35 / 25 mm | Flat / Flat |

Restitution is 0.83 and friction coefficient 0.4 in the primary experiment. Inertia and CG sensitivity must accompany off-center conclusions. [McNally et al. (2016)](https://doi.org/10.1016/j.proeng.2016.06.329) establishes the importance of CG depth, inertia, and face geometry for off-center trajectory sensitivity; it does not validate these assumed parameter values. Manufacturer loft alone is insufficient to identify gear effect.

## Frames and Rigid Delivery Geometry

The native frame is right-handed: +X toward target, +Y toward the right-handed golfer/left, +Z upward. Reported lateral coordinates and face/path angles are positive right. Contact horizontal offset is positive toward the toe (−face-heel axis); vertical offset is positive toward the top of the face.

Starting with the static face normal and zero-lean shaft axis, apply a rigid rotation about +Y for forward lean, then a negative world-Z rotation for positive-right nominal face direction. Apply the stochastic shaft rotation using Rodrigues' formula about that delivered shaft axis. Positive shaft rotation closes the right-handed face. Zero twist gives delivered loft = static loft − forward lean. These rotations preserve the shaft–face angle and orthonormal face frame.

Unlike phase one's constant-delivered-loft sensitivity, forward lean here changes both face loft and shaft orientation. A 10° static driver at 10° lean can have zero delivered loft and a nonairborne nominal launch; that is a physical-domain limitation to report, not a missing value to interpolate.

## Two-Body Contact Law

The previous off-center path combined a scalar effective mass with additive heuristic spin and omitted the corresponding club rotational impulse. It is unsuitable for claiming mechanically consistent strike dispersion. The new research path uses a full club inertia tensor and contact lever, without additive gear spin.

Let c be club-minus-ball relative contact velocity, n the unit outward face normal, r_c and r_b the CG-to-contact levers, and [r]× the cross-product matrix. For impulse J applied to the ball:

K = (1/m_c + 1/m_b) I − [r_c]× I_c⁻¹ [r_c]× − [r_b]× I_b⁻¹ [r_b]×,

c_after = c_before − K J.

The primary law integrates Coulomb contact in normal-impulse coordinates. For each positive normal increment dP, solve the tangential maximum-dissipation disk problem using the current relative contact velocity and the full coupled K. Compression ends at zero normal approach velocity. Store compression work E_c = ∫ c_n dP; restitution returns e_energy² E_c through ∫ −c_n dP. The coefficient 0.83 is therefore **energetic restitution**, not a prescribed normal velocity ratio in coupled off-center contact. Resolution is explicit and must pass refinement checks. The optional Numba implementation uses no fast-math transformations and is checked against the Python reference.

A separate maximum-dissipation/Newton comparator minimizes ½JᵀKJ − cᵀJ subject to nᵀKJ = (1+e)c_n and the impulse friction cone. This comparator does not claim to reproduce the full slip/grip history. An earlier stick-first candidate passed simple centered checks but violated passivity in Astra's coupled elastic counterexample; that failed experiment motivated the corrected comparator and incremental primary law. It is not used in the final primary pathway.

All four body velocity states receive equal-and-opposite linear and angular impulse updates. Validate work recovery, cone bounds, energy passivity, linear/angular momentum, frame covariance, and numerical refinement; do not replace failed numerical solutions with heuristic launches. These rigid contact laws are not measured tangential-compliance models. Conservation and convergence establish implementation consistency, not empirical golf-ball/club accuracy. Shaft flexibility, face compliance, speed-dependent restitution, local friction, grooves, and turf interaction remain unqualified. The external measurement obligation remains open as [#11984](https://github.com/D-sorganization/UpstreamDrift/issues/11984) and in the [Deferred-Validation Catalog](../../../development/planning/DV-11984.md).

For curved faces, a local parabolic surface provides a declared geometry approximation. Surface displacement along the center normal is −½(h²/R_b + v²/R_r), and its normal is proportional to n + (h/R_b)t_toe + (v/R_r)t_up. A zero radius explicitly selects a flat direction; it is not evaluated as a division by zero.

## Stochastic Delivery and Covariance

The primitive error vector is ordered as shaft rotation (degrees), path (degrees), speed (m/s), toe/heel contact (m), high/low contact (m), forward lean (degrees), and attack angle (degrees). A multivariate Gaussian is a transparent local sensitivity assumption, not an established distribution for all human motion. Without player data there is no justified empirical tail or covariance fit.

Default comparisons include shaft-rotation SD 1° and 2°, path SD 0.5°, speed SD 1 and 2 mph (exact conversion 0.44704 m/s per mph), and strike SD 0.25 inch = 0.00635 m on **each** contact axis. The radial strike SD is not 0.25 inch. Baseline and progressively combined noise profiles separate contributions before their interactions are considered.

Users can edit all SDs and the complete symmetric correlation matrix. Require a unit diagonal, finite entries, and positive semidefiniteness. Zero-SD dimensions are exactly constant. Draw one seeded standardized sample matrix and reuse corresponding shot indices across patterns. Preserve the requested covariance and report the realized input covariance. Actual face angle is derived nonlinearly from shaft rotation and lean; a shaft/path correlation must not be labelled an exact face/path correlation. Report realized face/path covariance explicitly.

Do not clip, retry, or resample off-face, nonairborne, or otherwise unsupported deliveries. Preserve their input rows and failure categories. Supported-endpoint statistics and density are explicitly conditional when any shots are unsupported; report the valid fraction and interval alongside them. Full-population strokes gained is unavailable if an unsupported outcome has no defensible terminal lie. Conditional paired scoring uses the intersection of supported indices across all patterns, with the reduced population stated.

## Density, Uncertainty, and Course Scoring

Academic figures use common axes, units, and density normalization across patterns. Show points, binned probability density, marginal distributions, and empirical 50%, 80%, and 95% highest-density regions. Report bin resolution and achieved probability mass; tied bins can exceed the requested threshold. Do not manufacture a two-dimensional Gaussian width for a nearly one-dimensional input/output manifold. Export publication PNG, SVG, and PDF; inspect rendered output for clipping, overlaps, and readable annotations.

Report lateral and longitudinal SD, full covariance, mean bias, radial RMSE and quantiles, target containment, and paired differences with sampling uncertainty. The same approximate historical scoring baseline as phase one applies: driver tee/fairway/rough; iron/wedge approach/green/rough. Landing is a hypothetical stopping point. Roll, hazards, and wedge stopping are not inferred. Sampling intervals exclude model, equipment, and course uncertainty. Course-dimension and off-center head-parameter sensitivities remain separate from Monte Carlo error.

## Pilot, Final Matrix, and Acceptance

First run small deterministic and Monte Carlo pilots. The final 10,000-shot-per-pattern matrix begins only after software contracts, physical identities, matching residuals, numerical refinement, population accounting, plot review, and an independent Astra review pass. Save unavailable cells as explicit records rather than omitting them.

| Requirement | Executable Evidence | Acceptance |
| --- | --- | --- |
| Static Loft and Lean | `test_club_geometry.py`, `test_matched_config.py` | Rotation identities, physical ranges, sourced static values |
| Correlated Noise | `test_noise.py` | PSD rejection, reproducibility, zero variance, empirical covariance |
| Raw Nominal Matching | `test_nominal_matching.py` and Real Pilot Receipts | Physical raw X/Y residual ≤0.05 m; fixed bounds; no aiming rotation |
| Off-Center Contact | Independent Rigid Contact Tests | Momentum, passive energy, restitution, frame covariance, central limit, gear direction |
| Unsupported Samples | Matched Runner Tests | Every input retained; no resampling; explicit conditional estimands |
| Density Figures | Density/Plot Tests and Rendered Review | Normalization, ties, degeneracy, common scale, readable exports |
| Final Results | Source-Bound Manifests and Independent Recalculation | Exact reproducibility, complete matrix accounting, no stale provenance |

Final reports must link exact source revisions, native binary identity, input specifications, seeds, solver bounds, numerical tolerances, result hashes, pilot evidence, and review dispositions. Historical phase-one outputs remain labelled with their original model and controls.
