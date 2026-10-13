# Final Report Readiness Review

## Confirmed Precision Defect

The current four-decimal formatter loses information in actual saved results. Across original and supplemental per-cell paired SG records, 18 nonzero estimates display as zero, 13 nonzero intervals collapse to identical bounds, and 16 one-sided intervals display a zero boundary (47 affected rows; counts overlap and retain protocol duplicates). The selected report table itself reproduces the defect: 7-iron, zero lean, curve 1, shaft SD 1°, shaft-only Draw is −0.000505606 [−0.000513691, −0.000496835], rendered as −0.0005 [−0.0005, −0.0005]. The Driver combined-noise example also prints two small nonzero effects as zero.

`actual_selected_table.md` was produced by the current report table function using saved summaries. `precision_scan.json` contains exact values, source paths and hashes. This does not imply practical significance: the correction preserves the distinction between a very small effect and exactly zero, without overstating model accuracy.

After active rendering releases the frozen sources, use one pure interval-aware formatter for report intervals and effect tables, with actual-value regressions preserving interval width/sign and approximately 1% numerical display accuracy. Reuse the GUI formatter's numerical policy without introducing Qt dependencies into reporting. Keep full-precision machine-readable outputs. No source change was made during this review.

## Scientific Narrative Completion

The builder already distinguishes individual-supported geometry, common-three scores and common-six interactions. Its warnings against empirical, universal and causal interpretations are appropriate. Matching nominal endpoints removes nominal range mismatch; it does not equalize realized carry or identify a causal distance component. The report must retain this distinction.

The original across-lean table intentionally uses the original control inventory; the supplemental section currently links the full comparison inventory, while representative interactions select zero-lean curve/SD contrasts. The final prose should explicitly explain the resolved Driver 5° speed-bound and PW 5° face-bound controls, naming their adapted protocols and common-six counts. This is a completion requirement, not permission to rewrite original unavailable cells.

Existing deterministic head-parameter/contact-law sensitivity is materially relevant to off-center claims and should be summarized or directly linked, separately from Monte Carlo uncertainty. Its saved summary contains 268 valid and 32 unsupported deterministic cases; the deliberately chosen ±20% settings are not measured parameter confidence intervals. No new flight experiment is required for this disclosure.

## Editable LaTeX and Publication Gates

The user-open `shot_pattern_analysis.tex` correctly labels phase one historical and phase-two final conclusions pending. `phase2/matched_delivery_reference.tex` contains calculation-level design and acceptance requirements, but no final result synthesis. The final Markdown/PDF builder does not create that missing editable LaTeX results deliverable. After the complete result/figure review, update the existing editable reference or link an explicitly editable phase-two results source from it, preserving historical model separation and exact result provenance.

The recorded built-in compiler bundle-fetch failure remains an unresolved compilation limitation. Do not call a separately generated ReportLab PDF LaTeX compilation acceptance, or overwrite prior failure evidence. Final PDF inspection, final graphical inventories and the final narrative review remain pending while producers run.

This is a read-only readiness review. It does not revoke the completed numerical/data audits or accept unfinished publication artifacts.
