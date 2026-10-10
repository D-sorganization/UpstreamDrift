# Phase-Two Shot Pattern Analysis: Numerical Expansion and Verification Review

## Executive Summary

This document records the independent Gemini 3.8 primary numerical expansion and Astra review of the canonical research article `docs/research/shot_pattern_analysis/shot_pattern_analysis.tex`. In accordance with user governance instructions, all work was performed strictly in place within the existing editable LaTeX source document without touching frozen publication producer scripts or graphic assets in the presentation worktree (`4023111d2a`). No staging, commits, or pushes were performed.

The phase-two numerical findings were inserted prior to `\section{Historical Phase-One Status and Scope}`, preserving all historical phase-one methods and baseline results. All newly added tables, captions, and headings were audited and confirmed to meet Title Case conventions (0 violations via `scripts/check_document_title_case.py`). All tables were formatted to fit within standard letter/A4 margins with zero overfull horizontal box warnings.

## Document Checksums and Integrity

- **Article Path**: `docs/research/shot_pattern_analysis/shot_pattern_analysis.tex`
- **Pre-Edit SHA256**: `9b136fb6d73ff240ceceb250b3ba701a2c0d1db7b8b607c8febb60e8bf91d4eb`
- **Post-Edit SHA256**: `6b586441c0f094b9f077e49203f17fdf782d0f26c1d0727dd344fd3c080306ad`
- **Title Case Violations**: `0`
- **Tectonic Compiler Exit Code**: `0`

## Source-Bound Evidence and Audit Reconciliations

All reported numerical values trace directly to accepted, byte-verified source artifacts:
1. **Selected Headline Traceability** (`phase2/validation/selected_headline_traceability_v1/review.json`):
   - All 52 referenced simulation summary and manifest SHA256 hashes independently verified.
   - Covers 24 unique zero-lean cells across driver, 7-iron, and pitching wedge, yielding 72 pattern rows.
2. **Supplement Comparisons Ledger** (`phase2/supplement_report_v1/REPORT_SUPPLEMENT_COMPARISONS.json`):
   - Reconciled byte-for-byte to `phase2/validation/protocol_name_recovery_v6/expected_numeric_hashes.json` (all 7 reference files matched).
   - Contains 384 qualified comparisons: 164 shaft rotation SD, 164 curve strength, 56 across-lean.
   - Fully cross-checked against accepted independent audits `root_review.json` and `astra_review.json`.
3. **Public API Selectors Reused**:
   - `select_representative_interaction_pairs()` in `scripts/research/shot_pattern_analysis/final_report_inputs.py` (12 interaction contrasts).
   - `select_bounds_control_pairs()` in `scripts/research/shot_pattern_analysis/final_report_inputs.py` (2 bounds controls).
   - Pure error-budget formatting via `format_sg_interval_record()` and `format_sg_value()` in `src/tools/shot_pattern_analysis/matched_formatting.py`.
   - Synthesis lines logic from `scripts/research/shot_pattern_analysis/report_synthesis.py`.

## Numerical Expansion Inventory and Findings

### 1. Condition Keys (C01--C24)
A compact key convention was established to prevent wide, overflowing identifiers:
- **Driver (C01--C08)**: Target $X_T = 178.949$\,m. C01--C04 curve $\times 1$ (face $\pm 2.029^\circ$, path $\pm 3.529^\circ$, speed $45.063$\,m/s); C05--C08 curve $\times 2$ (face $\pm 4.033^\circ$, path $\pm 7.033^\circ$, speed $45.253$\,m/s). Straight nominal speed $45.000$\,m/s. Protocol `original_primary/1`.
- **7-Iron (C09--C16)**: Target $X_T = 127.882$\,m. C09--C12 curve $\times 1$ (face $\pm 0.444^\circ$, path $\pm 1.944^\circ$, speed $36.014$\,m/s); C13--C16 curve $\times 2$ (face $\pm 0.885^\circ$, path $\pm 3.885^\circ$, speed $36.054$\,m/s). Straight nominal speed $36.000$\,m/s. Protocol `original_primary/1`.
- **Pitching Wedge (C17--C24)**: Target $X_T = 77.402$\,m. C17--C20 curve $\times 1$ (face $\mp 0.130^\circ$, path $\pm 1.370^\circ$, speed $32.011$\,m/s, protocol `original_primary/1`); C21--C24 curve $\times 2$ (face $\mp 0.260^\circ$, path $\pm 2.740^\circ$, speed $32.046$\,m/s, verified supplemental protocol `pw-primary-face-bounds/1` with relaxed $\pm 6.0^\circ$ face bounds). Straight nominal speed $32.000$\,m/s.

### 2. Seventy-Two Pattern Rows (Tables 2–4)
Club-separated tables report individual supported endpoints ($n$), lateral SD ($\sigma_Y$), downrange SD ($\sigma_X$), target RMSE, absolute strokes gained against the historical Tour baseline, and paired relative $\Delta\text{SG}$ versus Straight with 95% Monte Carlo intervals.
- **Population Accounting**: Driver has 10,000 / 10,000 supported shots across all patterns. 7-Iron `combined2` has 9,996 / 10,000 supported shots (4 low-strike ground scrapes). Pitching Wedge `combined2` has 9,999 / 10,000 supported shots (1 low-strike scrape). Common-three scoring intersection strictly matches individual supported counts.

### 3. Direct Paired Common-Six Interaction Contrasts (Table 5, E01--E12)
The 12 representative interaction contrasts were evaluated directly on the six-population supported-index intersection with normalized primitive noise pairing:
- **Driver (E01--E04)**: Doubling shaft SD ($1^\circ \to 2^\circ$) increases Draw's relative advantage over Straight ($+0.00282$ strokes in shaft-only; $+0.00217$ in combined-2). Doubling curve strength adds an even larger advantage ($+0.00491$ in shaft-only; $+0.00449$ in combined-2). Fade is penalized.
- **7-Iron (E05--E08)**: Doubling shaft SD produces negligible effect modifications (intervals span zero). Doubling curve strength significantly enhances Fade's relative advantage ($+0.00151$ in shaft-only; $+0.00118$ in combined-2).
- **Pitching Wedge (E09--E12)**: Doubling shaft SD favors Draw ($+0.00109$ shaft-only; $+3.89\times 10^{-4}$ combined-2) and penalizes Fade. Doubling curve strength strongly favors Draw ($+0.00559$ shaft-only; $+5.62\times 10^{-4}$ combined-2) and penalizes Fade ($-0.00548$ shaft-only; $-2.93\times 10^{-4}$ combined-2).

### 4. Fixed-Target Across-Lean Controls and Rematching Physics (Tables 6–7)
Evaluating across forward shaft lean ($0^\circ \to 5^\circ$) at identical target distances requires solving new nominal delivery inputs:
- **Driver Curve $\times 2$ (Control Bounds)**: Target 178.949 m. Straight speed increases from $45.00$ to $55.41$\,m/s ($+23.1\%$) to compensate for delofting from $10^\circ$ to $5^\circ$. Curved speed reaches $56.35$\,m/s with face $\pm 4.89^\circ$. High kinetic energy causes 60 low-launch shots to fail support ($n=9{,}940$). Draw relative benefit changes by $+0.00598$ [$+0.00483, +0.00700$]; Fade changes by $-0.001397$ [$-0.00234, -4.807\times 10^{-4}$]. Protocol: `original_control/1` $\to$ `driver-control-speed-bounds/1`.
- **PW Curve $\times 2$ (Face Bounds)**: Target 77.402 m. Forward lean reduces loft from $46^\circ$ to $41^\circ$, moving launch closer to optimal distance trajectory. Straight speed *decreases* from $32.00$ to $29.12$\,m/s ($-9.0\%$). Curved speed decreases from $32.05$ to $29.16$\,m/s with face $\mp 0.08^\circ$. Common-six $n=9{,}999$. Draw relative change is $-3.513\times 10^{-4}$ [$-5.603\times 10^{-4}, -1.424\times 10^{-4}$]; Fade is $+6.868\times 10^{-4}$ [$+4.851\times 10^{-4}, +8.859\times 10^{-4}$]. Protocol: `pw-primary-face-bounds/1` $\to$ `pw-control-face-bounds/1`.
- **PW Curve $\times 1$ (Original Control)**: Target 77.402 m. Rematched Straight speed $29.12$\,m/s, Curved speed $29.13$\,m/s, face $\mp 0.04^\circ$. Common-six $n=9{,}999$. Draw change $-2.001\times 10^{-4}$; Fade $+3.139\times 10^{-4}$. Protocol: `original_control/1`.

### 5. Cross-Condition Synthesis and Reversals (Table 8)
The 8 selected conditions per club demonstrate that relative scoring behavior reverses across clubs:
- 7-Iron Fade is favorable in 8 of 8 conditions (+8/0/0) and Draw is penalized (0/-8/0).
- Pitching Wedge Draw is favorable in 6 conditions and neutral in 2 (+6/0/2), while Fade is penalized in 6, positive in 1, neutral in 1 (1/-6/1).
- Driver Draw is favorable in 6 conditions and neutral in 2 (+6/0/2), while Fade is negative in 4, positive in 1, neutral in 3 (1/-4/3).
This proves that there is no universal 'best shape' in golf modeling; optimal patterns depend on club loft, aerodynamic regime, delivery profile, and target context.

## Independent Reviews

### Root Engineering Perspective
- **Source Fidelity**: All numbers, intervals, denominators, and identifiers match the source data exactly without recomputation or rounding small intervals into equality.
- **Repository Safety**: Only `shot_pattern_analysis.tex` was modified. No Python code, manifests, or graphic outputs were touched. All untracked intermediate build artifacts were purged.
- **Compiler Health**: Tectonic compiles the document cleanly with zero overfull horizontal box warnings in the new material.

### Astra Reviewer Perspective
- **Scientific Rigor**: The narrative strictly adheres to `FINAL_NARRATIVE_ACCEPTANCE.md`. Causal claims are explicitly disclaimed; fixed-target distance control is explained through physical rematching; primitive shaft-twist SD is distinguished from delivered face SD; Monte Carlo intervals are identified as simulation sampling uncertainty on correlated draws.
- **Population Clarity**: Individual supported populations, common-three scoring intersections, and common-six contrast intersections are kept distinct with explicit denominators.
- **Governance**: Full matrix hyperlinks (324 primary, 72 control, 26 supplement, 4 aliases, 384 comparisons) are properly documented, and publication acceptance remains explicitly pending.

## Conclusion
The editable LaTeX document `shot_pattern_analysis.tex` now contains a complete, source-verified phase-two numerical expansion with full contextual rigor, ready for subsequent publication workflow integration.