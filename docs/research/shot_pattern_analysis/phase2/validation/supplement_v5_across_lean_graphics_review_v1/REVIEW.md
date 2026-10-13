# Across-Lean Graphics Component Review

The 56-pair component is complete and its 896 assets are byte-stable: 336 PNGs, 280 PDFs and 280 SVGs. All 56 six-image contact sheets were inspected, followed by eight larger representative views spanning Driver, Seven Iron and Pitching Wedge, narrow shaft-only and combined-noise distributions, SD 1/2, curvature 1/2, original controls, supplemented controls and the explicit PW alias. Exact inspected paths and hashes are in `review.json`; the full inventory is in `inventory.json`.

## Scientific Labels and Populations

The ordered protocols, physical targets and common-six population counts agree with the 56 numerical comparison records. Density, HDR and marginal captions distinguish all common-supported endpoints from 600 displayed dots. Social cards disclose shared scales within each figure and potentially different scales across physical-pattern figures. Overhead graphics correctly identify saved paths as unpaired by shot index, and disclose lateral display magnification. Model-conditional/carry-only caveats remain visible. No new statistics or physical simulations were run.

## Remaining Academic Layout Gate

The academic six-panel density layout has insufficient row spacing when equal-aspect axes fill their grid slots. The upper Downrange label is visually cramped against the next row title. Full-size PW and Seven Iron examples confirm this spacing defect; this review does **not** claim proven glyph collision. PDF font-metric screening flags 30 of 56 academic density figures: 12 PW shaft-only, 12 Seven Iron shaft-only and six Seven Iron combined-noise figures. Font boxes overlap by up to 9.76949 pt, but those boxes exceed visible glyph ink. The exact files and screening values are in `font_bbox_scope.json`.

The minimal repair belongs in `_write_pair_density_figure` in `matched_supplement_figures.py`: increase or measure the row gutter currently set by `hspace=.24`. A failing regression should measure upper-row decorated x labels against lower-row titles in all three columns, then require positive readable clearance while preserving fonts, limits, density arrays, HDR levels, common masks and counts. No other renderer change is justified by this review.

## Publication Revision Plan

Allow the current producer to finish. Verify its complete receipt with the frozen producer and preserve it unchanged. A separately qualified child publication may rerender only academic paired-density PNG/PDF/SVG files from the same verified endpoint populations, reuse all other assets by exact hash, and bind parent receipt, numerical inputs, old/new renderer hashes and per-output origin in an explicit child manifest. Do not silently update the parent receipt, relax the existing current-source loader guard, or recompute bootstrap statistics for this layout repair. Review every regenerated academic density image before accepting the child publication.

Social density and overhead components pass this review. The entire component remains pending the academic spacing repair and eventual final pipeline-manifest reconciliation. This is not full-study publication acceptance.
