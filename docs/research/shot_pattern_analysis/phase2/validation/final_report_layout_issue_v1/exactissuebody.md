## Overview

During independent visual inspection of the 8-page rendered partial report preview (`docs/research/shot_pattern_analysis/phase2/validation/final_report_composition_layout_v1/REVIEW.md`, `review.json`), publication-blocking PDF layout and readability failures were confirmed.

> **Note on Scope**: The 8-page preview inspected is deliberately partial (focused on diagnostic numerical sections) and is not the complete final report or final publication acceptance.

### Relationship to Issue #12128

This issue is related to, but distinct from, #12128 ("Resolve Six Definite Final-Report Composition Gaps in Phase-Two Shot Pattern Analysis"). Issue #12128 addresses upstream composition gaps, data-assembly logic, and missing selections in the synthesis scripts (`build_final_report.py`, `final_report_inputs.py`). In contrast, this issue addresses concrete layout failures, column-width bottlenecks, and token fragmentation discovered during physical PDF rendering and visual page inspection. It is not a duplicate.

### Implementation Ownership

Astra owns the fix implementation for this issue.

---

## Confirmed Layout Failures

### 1. Resolved Selection Table Numeric Token and Header Splitting (Pages 2–5)
- **Defect**: The resolved results table contains 9 columns, which causes aggressive line breaks inside numeric tokens, signs, scientific exponents, and column headers.
- **Concrete Example**: On Page 2, the Driver SG value `+7.462e-05` splits after `e-`, and interval digits span multiple lines. Table headers also break awkwardly inside single words.
- **Root Cause**: The ReportLab renderer in `build_pdf.py` allocates approximately 22% of page width to the first column and divides the remaining 78% equally across 8 columns. Nine columns provide insufficient width for high-precision scientific notation and confidence intervals at readable type sizes.
- **Severity**: Publication blocking.

### 2. Paired Effect Interactions Table Identifier Fragmentation (Pages 5–8)
- **Defect**: In the paired effects table, repeated full ordered source identifiers (e.g., condition descriptors and contrast labels) break across multiple lines, causing rows to become excessively tall and difficult to scan visually.
- **Impact**: While numerical estimates remain readable, layout efficiency is severely degraded by repeating lengthy identifiers across table rows.
- **Severity**: Readability defect.

### 3. Confirmed Acceptable Components
- Page 1 summary and correlation tables.
- Body prose, statistical qualifications, and model/population disclaimers.
- Page numbering, headers, repeated table headers, and absence of page-edge clipping or footer overlaps.

---

## Acceptance Criteria

### 1. Complete Data and Precision Preservation
- [ ] Retain all 24 selected cells (including the recovered zero-lean PW curve-2 cells) and all 72 patterns without omission.
- [ ] Retain all exact 12 ordered contrasts across SD1 and SD2, for both `shaft_only` and `combined2` profiles.
- [ ] Preserve full numerical precision and confidence intervals for strokes gained (SG) and dispersion metrics (no truncating, rounding away precision, or dropping interval bounds).
- [ ] Maintain exact protocol identities, population distinctions (individual vs. shared scoring counts), and full provenance mapping.

### 2. Structural Layout Restructuring and Readable Formatting
- [ ] Restructure the 9-column resolved results table into narrower protocol/club-labelled blocks and/or separate scoring and population tables to guarantee adequate column width.
- [ ] Introduce a readable condition-key mapping adjacent to the paired effects table (or allocate adequate source column width) so that lengthy identifiers do not fragment or dominate vertical row heights.
- [ ] Enforce the existing readable font floor: do not resolve column width constraints by shrinking typography below readable publication standards.
- [ ] Guarantee zero within-token line breaks across all rendered tables (all signs, decimal values, scientific notation exponents `e-XX`, and header words must remain intact on a single line).

### 3. Verification and Quality Gates
- [ ] Implement a failing-first regression test that validates numeric token integrity and table layout structure against the rendered output before applying fixes.
- [ ] Re-render and visually inspect all revised preview pages (1–8) and the complete final report once the pipeline is completed to verify layout fidelity and absence of visual regressions.
