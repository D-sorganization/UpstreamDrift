# Final Layout Independent Initial Review

Status: **Changes Required**. The actual all-club render is still running; this is a source review and partial visual inspection, not final visual acceptance.

## Confirmed Publication Defect

The actual Driver Draw two-panel social figure visibly places its last downrange xlabel against the footer. A rendered-bounds regression confirms overlap: xlabel y-range `[68.7837, 84.0615]` versus footer `[24.3000, 74.7337]`, a 5.95-pixel intersection on the 1080-pixel-wide fixture. The independent test returns one failure and two passes. Existing tests compare captions with axes rectangles and miss decorated axis labels outside those rectangles.

After the source-bound diagnostic render finishes, reserve bottom spacing based on the rendered xlabel/tick decorations, then rerun the red regression and actual two-panel saved-data renders. Do not remove population or model-conditional disclosures to make room.

## Typography and Layout Findings

Whole multiline text-block height and rotated-label height are insufficient readability metrics: a 4-point multiline caption or long rotated label passes the previous 11-pixel-height assertion. The independent regression reproduces both cases and uses per-line font em-size instead, including all text artists, legends and ticks. Actual candidate fonts pass this stronger metric with a minimum of 12.5 em-pixels at a 1080-pixel canvas width. This is a coverage defect, not evidence of tiny candidate fonts.

The first Driver three-panel social rendering is substantially clearer than the previous landscape card: readable HDR, marginals, metadata and footer, with separated main/marginal ticks. Its tall scrolling layout is reasonable for this technical post. A 4:5 format was a proposed style preference rather than an explicit user constraint; compression is not required solely to enforce it.

Source inspection found layout, artist sizing, legend construction and locator changes. Density/HDR calculations, endpoint data selection, shared limits and marginal normalization remain unchanged in the inspected diff. Complete source/runtime equivalence and all-club visual qualification remain separate gates.

## Evidence

- `final_layout_independent_initial_review.json`: source and two inspected PNG hashes, findings and test status.
- `FINAL_LAYOUT_INDEPENDENT_FOOTER_RED.txt`: exact failed regression output before repair.
- `final_layout_independent_font_metrics.json`: all visible text metrics for two/three-panel fixtures.
- Permanent regression: `tests/tools/shot_pattern_analysis/test_final_layout_review_contracts.py`. It was moved from the review directory after the red run; the historical log truthfully retains the old test path. Only the reviewer's own orphan was removed.

No renderer or numerical source was edited by the reviewer. No simulation was run.
