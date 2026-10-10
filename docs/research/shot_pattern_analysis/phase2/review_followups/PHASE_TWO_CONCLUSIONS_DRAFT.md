# Draft Phase-Two Conclusions

Draft for final report integration after complete source-bound publication inventories qualify. These are conditional simulation results, not empirical player accuracy or graphical acceptance. Positive relative SG favors the named curve over Straight; positive interaction means that relative benefit increased in condition B versus A. All intervals below are paired 95% Monte Carlo sampling intervals, not model-uncertainty intervals.

## What Changes With Shaft Variability and Curve Strength

The zero-lean selected inventory does not support a single preferred shot shape across clubs and noise assumptions. In all eight selected 7-iron conditions (two curve scales × two shaft SDs × two profiles), Fade has positive SG benefit and Draw has negative benefit, with the respective intervals excluding zero. This consistency is confined to this specified eight-condition selection and hypothetical approach context.

Driver Draw is positive with an interval excluding zero in six of eight selected conditions; the two ordinary-curve, one-degree cases have intervals spanning zero. Driver Fade becomes negative in all four two-degree cases, whereas its one-degree evidence is mixed. There is no uniform Driver Draw/Fade improvement across every condition.

PW shaft-only conditions favor Draw and penalize Fade in all four selected cases. Adding combined path/speed/strike noise weakens that simple pattern: one-degree Draw intervals span zero, while the strong-curve one-degree Fade has a small positive interval; both two-degree combined conditions favor Draw and penalize Fade. The shaft-only result is therefore not a general PW preference.

Doubling primitive shaft-twist SD changes both the endpoint spread and its bias; it is not a prescribed doubling of delivered face-angle SD. The following direct interactions quantify changes in relative scoring, not changes in absolute performance. A curve can gain relative to Straight even if both patterns become worse. Curve-strength comparisons include nominal delivery rematching and, for strong PW, the explicitly adapted bounds protocol.

| Club / Profile | Ordered Change | Common Six n / 10,000 | Draw Interaction, Strokes [95% CI] | Fade Interaction, Strokes [95% CI] |
|---|---|---:|---:|---:|
| driver / shaft_only | SD 1→2°, curve ×1 | 10000 | +0.00281801 [+0.00234573, +0.00328781] | -0.00129338 [-0.00170755, -0.000895627] |
| driver / shaft_only | Curve ×1→×2, SD 2° | 10000 | +0.00490503 [+0.00438298, +0.00545231] | -0.000235419 [-0.000574032, +0.000112352] |
| driver / combined2 | SD 1→2°, curve ×1 | 10000 | +0.00216825 [+0.00160011, +0.00270617] | -0.000969873 [-0.00150214, -0.000502799] |
| driver / combined2 | Curve ×1→×2, SD 2° | 10000 | +0.00448715 [+0.00393767, +0.0050661] | +9.29854e-05 [-0.000273522, +0.000464163] |
| seven_iron / shaft_only | SD 1→2°, curve ×1 | 10000 | +6.44855e-06 [-7.08911e-05, +4.9342e-05] | +1.99997e-05 [-5.97948e-05, +0.000135795] |
| seven_iron / shaft_only | Curve ×1→×2, SD 2° | 10000 | +0.000210886 [+9.16361e-05, +0.000357793] | +0.00151468 [+0.00143332, +0.00162877] |
| seven_iron / combined2 | SD 1→2°, curve ×1 | 9996 | -6.0275e-05 [-0.000296613, +0.00015399] | +0.00021754 [-7.96529e-05, +0.000511338] |
| seven_iron / combined2 | Curve ×1→×2, SD 2° | 9996 | -0.000247592 [-0.00049481, +2.78039e-05] | +0.0011829 [+0.000981745, +0.00136468] |
| pitching_wedge / shaft_only | SD 1→2°, curve ×1 | 10000 | +0.00109268 [+0.00102641, +0.00116357] | -0.0010371 [-0.00110479, -0.000971134] |
| pitching_wedge / shaft_only | Curve ×1→×2, SD 2° | 10000 | +0.00559049 [+0.00552875, +0.00564324] | -0.00547902 [-0.00553037, -0.00541849] |
| pitching_wedge / combined2 | SD 1→2°, curve ×1 | 9999 | +0.000389315 [+0.000218487, +0.000550562] | -0.00042761 [-0.000545245, -0.00031499] |
| pitching_wedge / combined2 | Curve ×1→×2, SD 2° | 9999 | +0.000562086 [+0.000416443, +0.000716681] | -0.000293463 [-0.000491058, -8.72342e-05] |

Intervals were taken directly from same-index six-population interaction records. None were formed by subtracting marginal confidence limits. An interval spanning zero does not prove equivalence or no effect.

## Distance and Fixed-Target Controls

Matching the nominal raw landing removes the starting target mismatch, but noisy mean carry can still change. In straight zero-lean shaft-only cases, increasing SD from 1° to 2° changes Driver mean downrange error from −0.505 to −2.005 m, 7-iron from −0.055 to −0.220 m, and PW from −0.023 to −0.095 m. These descriptive differences have no newly computed interval. Their opposite club-dependent lateral/downrange correlations also show why closing/delofting alone cannot guarantee long-left/short-right.

The separately qualified fixed-target, curve ×2, SD 2°, combined-noise controls compare lean 0°→5°. Driver uses the reviewed speed-bound extension and n=9,940 common supported indices: its Draw-relative interaction is +0.00598008 [+0.00483146, +0.00699576] strokes, while Fade is −0.00139745 [−0.00233796, −0.000480746]. PW uses the reviewed face-bound protocol and the genuine zero-lean alias, n=9,999: Draw is −0.000351324 [−0.000560261, −0.000142443], Fade +0.000686801 [+0.000485119, +0.000885867]. These ordered changes do not say the later condition is better in absolute SG.

Holding the target fixed is a useful distance control, but the matched face/path/speed and supported population also change. It does not isolate a pure causal lean, curvature or distance contribution. Read lateral spread together with downrange spread, bias, target error and scoring context. Driver uses tee/fairway/rough scoring; irons use approach/green scoring; carry is treated as final lie.

## Exact Evidence and Integration Gate

- Selected 24-cell/72-pattern extraction: [review.json](../validation/selected_headline_traceability_v1/review.json); independent field-by-field qualification: adjacent `astra_review.json`. Individual endpoint support and common-three SG support are separately recorded.
- Interaction values: [REPORT_SUPPLEMENT_COMPARISONS.json](../presentation_preview_rejected_supplement_v3/supplement_report_v1/REPORT_SUPPLEMENT_COMPARISONS.json) (SHA256 `c46738a38bb35ab86ba2163260e31a225aca07c807e021225d3c38fd09f0cf46`). This is the preserved numerical output of an intentionally retired graphical run, not an accepted final publication. Numeric parity and the 56 newly added across-lean interactions were independently checked in [review.json](../validation/supplement_v3_numeric_comparison_review/review.json).
- Actual fixed-target producer qualification and full-precision objects: [actual_control_effects.json](../validation/across_lean_interaction_review_v1/actual_control_effects.json) and `independent_qualification.json`.
- The [V5 Numerical Reconciliation](../validation/supplement_v5_numeric_review_v1/numeric_review.json) now confirms exact parity of every numerical file and these selected interaction objects. The replacement pipeline is still rendering; before publication, cite its successful final receipt and qualify the complete graphics. Preserve historical numerical provenance; do not describe the retired graphics run as accepted.
- No primary-manifest protection is inferred from the recovery wrapper’s self-equality assertion. The completed primary manifest requires independent validation against its accepted receipt.

Finite-sample intervals exclude head-property uncertainty, contact-law discrepancy, course-baseline approximation and unmeasured golfer delivery distributions. These model-conditional patterns are hypotheses for measurement, not a universal strategy recommendation.

## Bound Numeric Provenance

Independent numeric-parity review receipt SHA256: `b1f8666e324af98fd3f733c9ad7786943aba51550fb9ebdb843f2d7dd80c6509`. Selected interaction producer `matched_lean_control.py` SHA256: `0c57f3c1a2535f63ba059b5c6fb5c3414f83cc6b9a1169a429cf14a19bbd12c0`. The per-pair `analysis_source_sha256` maps in the cited comparison JSON bind the remaining scoring, bootstrap and validation dependencies. The retired v3 run did not finish a pipeline receipt; no final producer receipt is claimed for it. Its exact numerical file SHA above and independent review are the draft provenance, pending reconciliation with the successful V5 receipt once available (the intermediate v4 run terminated without completion).
