## Overview

During independent root visual inspection of full production saved-data plots from the first 12 completed `curve_strength` comparisons under `docs/research/shot_pattern_analysis/phase2/supplement_report_v1/figures/comparisons`, a confirmed visual defect was identified in paired density and overhead figures. When comparison conditions include full covariance profiles with multiple signed cross-correlations and standard deviations, the resulting condition captions become extremely long (>300 characters).

Because `src/tools/shot_pattern_analysis/matched_supplement_figures.py` renders these strings as single unwrapped lines:

1. In academic figures (`_density.png` and `_overhead.png`), Matplotlib's `bbox_inches="tight"` expands the canvas bounding box horizontally to encompass the unwrapped text, producing unexplained giant image widths (e.g. 6449×2262 and 6450×1791 pixels compared to typical shaft-only dimensions of 4321×2457 and 4502×1791).
2. In full-size social overhead cards (`_overhead_social.png`), which render onto a fixed 1920×1080 canvas without `bbox_inches="tight"`, both condition lines are severely clipped at the right margin, truncating vital scientific context including correlation coefficients, shaft lean, curve scale, shaft twist standard deviation, and source protocol identifiers.

By contrast, paired density social figures (`write_pair_social_density`) wrap header text and keep captions within nominal bounds.

> **Important Scope and Operational Disclosures**:
>
> - **No Numerical Error Alleged**: This issue concerns graphical presentation, text wrapping, and figure bounds only. All underlying simulation endpoints, trajectory samples, 2D histogram bins, HDR contours, coordinate systems, and statistical metrics remain strictly valid and unchanged.
> - **No Monte Carlo Rerun**: Any resolution must operate as an explicit downstream child presentation lineage that consumes existing parent data and preserves all computed statistics without rerunning simulation pipelines.
> - **Inspection Evidence**: Root inspected all 12 contact sheets and the full-size social card in `docs/research/shot_pattern_analysis/phase2/validation/supplement_v5_curve_graphics_first12_v1/`.

---

## Relationship to Existing Issues and Active Workflows

### Disambiguation From Issue #12131 (Separate Row Gutter Defect)

This issue is strictly separate from issue [#12131](https://github.com/D-sorganization/UpstreamDrift/issues/12131) ("Resolve Cramped Inter-Row Vertical Spacing in Academic Paired-Density Figures").

- Issue #12131 addresses internal subplot grid spacing and inadequate vertical clearance between row 0 horizontal axis labels and row 1 subplot titles (`hspace`).
- This issue addresses the lack of text wrapping in the figure header lines containing `_condition_caption`, causing tight-bbox lateral expansion in academic density/overhead figures and right-edge text clipping in fixed-canvas social overhead figures.
- **Do not duplicate, merge, or close issue #12131.**

### Disambiguation From Issue #12129

This issue is separate from report-table layout issue [#12129](https://github.com/D-sorganization/UpstreamDrift/issues/12129) ("Resolve Partial Report PDF Layout Fragmentation and Numeric Token Splitting in Phase-Two Shot Pattern Analysis"), which targets table column widths and token splitting inside the partial PDF report.

### Linkage to Issue #11976

This issue is linked to the delivery matrix and supplemental figure publication tracked under [#11976](https://github.com/D-sorganization/UpstreamDrift/issues/11976) ("Publish Reviewed Matched-Delivery Matrix and Traceable Scientific Reference").

---

## Confirmed Root Visual Finding and Concrete Evidence

### Inspected Production Example

- **Location**: `docs/research/shot_pattern_analysis/phase2/supplement_report_v1/figures/comparisons/`
- **Filename Stem**: `curve_strength_pitching_wedge_lean0_curve1_rotation1_cov_facepath_negative_vs_pitching_wedge_lean0_curve2_rotation1_cov_facepath_negative_edfeeda1b4`
- **Affected Condition Lines**:
  - `Curve ×1: Pitching Wedge · Covariance Profile (Path SD 0.5°, Speed SD 1 mph, Strike SD 0.25 in on Each Axis, Shaft Twist–Path Correlation r = +0.6, Shaft Twist–Speed Correlation r = +0.2, Path–Speed Correlation r = +0.12, Path–Horizontal Strike Correlation r = +0.24, Speed–Vertical Strike Correlation r = +0.244949) · Lean 0° · Curve ×1 · Shaft Twist SD 1° · original_primary/1`
  - `Curve ×2: Pitching Wedge · Covariance Profile (Path SD 0.5°, Speed SD 1 mph, Strike SD 0.25 in on Each Axis, Shaft Twist–Path Correlation r = +0.6, Shaft Twist–Speed Correlation r = +0.2, Path–Speed Correlation r = +0.12, Path–Horizontal Strike Correlation r = +0.24, Speed–Vertical Strike Correlation r = +0.244949) · Lean 0° · Curve ×2 · Shaft Twist SD 1° · pw-primary-face-bounds/1`

### Quantitative Image Dimension Disparity

Across the first 12 completed `curve_strength` comparisons (cataloged in `docs/research/shot_pattern_analysis/phase2/validation/supplement_v5_curve_graphics_first12_v1/inventory.json`):

- **Typical Shaft-Only Comparison (Pair 5)**:
  - `_density.png`: 4321 × 2457 px
  - `_overhead.png`: 4502 × 1791 px
- **Long Covariance Comparisons (Pairs 3, 4, 12)**:
  - Pair 3 (`cov_facepath_negative`): `_density.png` is **6449 × 2262 px**; `_overhead.png` is **6450 × 1791 px**
  - Pair 4 (`cov_facepath_positive`): `_density.png` is **6379 × 2285 px**; `_overhead.png` is **6418 × 1791 px**
  - Pair 12 (`rotation2 cov_facepath_negative`): `_density.png` is **6436 × 2397 px**; `_overhead.png` is **6450 × 1793 px**
- **Fixed-Canvas Social Overhead Card (`_overhead_social.png`)**:
  - Rendered at fixed 1920 × 1080 px (`figsize=(16, 9)`, `dpi=120`).
  - Text starting at `x = 0.06` (pixel 115) extends over 2500 pixels wide, clipping off >700 pixels past the right boundary (1920 px).
  - Truncated text loses: `Shaft Twist–Speed Correlation r = +0.2, Path–Speed Correlation r = +0.12, Path–Horizontal Strike Correlation r = +0.24, Speed–Vertical Strike Correlation r = +0.244949) · Lean 0° · Curve ×1 · Shaft Twist SD 1° · original_primary/1`.

### Reference Contrast: Density Social Figures

In `src/tools/shot_pattern_analysis/matched_pair_social.py` and `academic_plots.py` (`write_academic_dispersion_figure`), social captions and scenarios are wrapped via `textwrap.fill` and positioned using rendered text bounding boxes (`get_window_extent`), ensuring complete containment without clipping or canvas blowout.

---

## Root Cause Analysis

In `src/tools/shot_pattern_analysis/matched_supplement_figures.py`:

1. **Paired Density Figure Header (`_write_pair_density_figure`)**:
   ```python
   figure.text(
       0.06,
       0.895,
       f"{first_label}: {_condition_caption(first)} · {first['source_protocol']}",
       fontsize=7.7,
   )
   figure.text(
       0.06,
       0.875,
       f"{second_label}: {_condition_caption(second)} · {second['source_protocol']}",
       fontsize=7.7,
   )
   ```
2. **Paired Overhead Figure Header (`_create_pair_overhead_figure`)**:
   ```python
   figure.text(
       0.06,
       0.895,
       f"{first_label}: {_condition_caption(first)} · {first['source_protocol']}",
       fontsize=8.5 * font_scale,
   )
   figure.text(
       0.06,
       0.87,
       f"{second_label}: {_condition_caption(second)} · {second['source_protocol']}",
       fontsize=8.5 * font_scale,
   )
   ```
   Both functions draw `_condition_caption` as unwrapped single-line strings at fixed vertical coordinates. When covariance profiles introduce multiple correlation terms and standard deviations, the text exceeds the figure canvas bounds.

---

## Acceptance Criteria

### 1. Failing-First Rendered-Text Containment and Separation Tests

- [ ] Implement failing-first automated tests verifying text containment within nominal canvas boundaries for both paired density and overhead figures using actual long signed covariance captions (including all correlation pairs and standard deviations).
- [ ] Verify that header lines maintain clear vertical row separation without vertical glyph collision or line overlap.

### 2. Header Text Wrapping and Placement Within Nominal Bounds

- [ ] Wrap long condition captions cleanly (similar to `textwrap.fill` or dynamic bounding-box layout in `write_academic_dispersion_figure`).
- [ ] Place header text without clipping at the right margin on fixed 1920×1080 social overhead cards (`_overhead_social.png`).
- [ ] Eliminate unexplained giant canvas widths in academic exports (`_density.png` and `_overhead.png`), keeping rendered dimensions in line with typical figures (~4300–4500 px width) under `bbox_inches="tight"`.

### 3. Strict Scientific and Typographic Preservation

- [ ] Preserve all scientific caption content, including every correlation coefficient, standard deviation, club name, lean angle, curve scale, and source protocol token.
- [ ] Preserve existing font families, weights, and hierarchical font sizing.
- [ ] Preserve all underlying scientific data, 2D histogram bins, HDR contours (50%, 80%, 95%), scatter dot coordinates and counts, target crosshairs, and flight trajectories identically.
- [ ] Zero numerical changes and strictly NO Monte Carlo simulation reruns.

### 4. Comprehensive Multi-Profile Asset Review

- [ ] Perform actual visual and automated bounding-box review across all affected comparison figures and noise profiles (covariance profiles, combined profiles, shaft-only, speed, strike, and path variations) in the supplement.

### 5. Explicit Child Publication Lineage and Ledger Integrity

- [ ] Structure the update as an explicit downstream child publication lineage that binds:
  1. Parent source references and input data hashes.
  2. Exact byte integrity of all scientific coordinate and numerical data.
  3. Producer source hashes for corrected presentation scripts.
  4. Fully verified figure asset ledgers.
