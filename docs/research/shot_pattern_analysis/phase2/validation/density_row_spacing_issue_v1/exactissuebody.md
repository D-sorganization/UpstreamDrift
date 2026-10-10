## Overview

During independent visual inspection of rendered academic supplemental figures (`docs/research/shot_pattern_analysis/phase2/validation/supplement_v5_across_lean_graphics_review_v1/pw_academic_row_boundary_crop.png`), cramped vertical spacing between subplot rows was confirmed in academic paired-density figures. Specifically, the upper row's horizontal axis labels sit almost flush against the lower row's panel titles.

> **Operational and Scope Notes**:
> - **Scope Under Active Review**: Root and Astra are reviewing the live supplemental graphics separately. The full extent of affected figures is still under evaluation; a single confirmed example suffices to file and track the issue.
> - **Independent Live Pipeline**: The figure generation pipeline remains live in another checkout (`FINAL_LAYOUT`). Do not touch or stop the running pipeline.
> - **No Scientific Changes**: This issue concerns presentation-only grid geometry. All underlying data, numerical arrays, and statistical parameters remain strictly unchanged.

---

## Relationship to Existing Issues and Active Workflows

### Disambiguation From Issue #12129

This issue is separate from report-table layout issue [#12129](https://github.com/D-sorganization/UpstreamDrift/issues/12129) ("Resolve Partial Report PDF Layout Fragmentation and Numeric Token Splitting in Phase-Two Shot Pattern Analysis"). Issue #12129 addresses column-width limits and token wrapping inside tabular summaries of the partial PDF report. This issue addresses subplot grid spacing in standalone academic paired-density figures generated for the supplement.

### Linkage to Issue #11976

This issue is linked to the broader phase-two graphics study and delivery matrix tracked under [#11976](https://github.com/D-sorganization/UpstreamDrift/issues/11976) ("Publish Reviewed Matched-Delivery Matrix and Traceable Scientific Reference").

---

## Confirmed Visual Finding

### Inspected Artifact and Concrete Example

- **Inspected Artifact**: `docs/research/shot_pattern_analysis/phase2/validation/supplement_v5_across_lean_graphics_review_v1/pw_academic_row_boundary_crop.png`
- **Confirmed Condition**: Pitching Wedge (PW) across-lean comparison (lean 0° vs. lean 10°, `lean0→10`), curve 1, shaft rotation SD 1 (`shaft SD1`), shaft-only error model (`shaft_only`).
- **File Prefix / Stem**: Ending in `9a7f95d848_density` (full asset: `across_lean_pitching_wedge_lean0_curve1_rotation1_shaft_only_vs_pitching_wedge_lean10_curve1_rotation1_shaft_only_9a7f95d848_density.pdf` / `.png`).
- **Visual Defect**: In the 2×3 multi-panel figure, the first-row horizontal axis label ("Downrange (m)") is located almost against the lower-row panel title ("Straight · Lean 10°").

### Visual Assessment and Glyph Clarification

- Visual inspection confirms that while vertical clearance is severely cramped, text remains decipherable.
- **Important Qualification**: Do **not** claim proven glyph collision. Although larger PDF font-metric bounding boxes overlap in automated layout checks (e.g., negative box clearance gaps recorded in `font_bbox_scope.json` around `-9.77 pt`), font bounding box overlap is an artifact of tall font metric bounds and is not equivalent to actual character glyph collision.

### Root Cause and Grid Parameterization

- The layout is generated in `src/tools/shot_pattern_analysis/matched_supplement_figures.py` within the `_write_pair_density_figure` function.
- The figure defines a 2×3 subplot grid using:
  ```python
  figure.add_gridspec(
      2, 3, left=0.06, right=0.91, top=0.76, bottom=0.13, hspace=0.24, wspace=0.24
  )
  ```
- Subplots enforce equal metric scaling via `axis.set_aspect("equal", adjustable="box")`. Because metric dispersion ranges differ by club (e.g., downrange vs. lateral scale for pitching wedge versus driver), the physical subplot bounding box geometry combined with `hspace=0.24` leaves inadequate vertical clearance between row 0's x-axis label (`axis.set_xlabel("Downrange (m)")`) and row 1's title (`axis.set_title(...)`).

---

## Acceptance Criteria

### 1. Render-Aware Clear Inter-Row Separation

- [ ] Implement a render-aware regression test verifying deliberate, clear vertical separation between the upper-row x-axis labels ("Downrange (m)") and lower-row subplot titles across rendered academic paired-density figures.
- [ ] Preserve retained typography standards (font sizes and weights) without resolving spacing via typography degradation or font shrinking.
- [ ] Maintain equal metric scaling (`set_aspect("equal")`) across all subplots.
- [ ] Qualify minimal presentation-only corrections against saved actual figure examples across all clubs (Driver, 7-Iron, Pitching Wedge) and their respective metric aspect ratios.

### 2. Strict Scientific and Data Preservation

- [ ] Preserve all 2D histogram bins, bin counts, and bin edges identically.
- [ ] Preserve all highest density region (HDR) contour calculations, probability mass thresholds (50%, 80%, 95%), and contour lines.
- [ ] Preserve all scatter overlay sample indices, point coordinates, target crosshairs, and reference lines.
- [ ] Preserve all underlying population subsets, shot counts, and numerical files (`.json`, `.npy`, `.csv`) with zero scientific changes.

### 3. Comprehensive Asset Review

- [ ] Review all affected academic paired-density exports across the full supplement to ensure uniform vertical clearance across all comparison kinds and conditions.

### 4. Provenance, Ledger, and Fail-Closed Guard Integrity

- [ ] Preserve the original v5 source-bound final receipt.
- [ ] Ensure any revised publication explicitly binds:
  1. Original parent source references and checksums.
  2. Exact numeric bytes for all scientific data.
  3. Reused, untouched asset files.
  4. Corrected-asset producer source hashes.
  5. The complete, updated final ledger.
- [ ] Prohibit any silent hash editing or bypassing of existing fail-closed source verification guards.
