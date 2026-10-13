# V6 Vector-Format Acceptance Plan

## Scope and Evidence Basis

This plan supplements MAIN `phase2/validation/v6_output_review_harness_v1/FINAL_REVIEW_PLAN.md` and current harness SHA `1db7be642e097377afeb44874687e41639a29c32c0440f13b3055eb3457d0bf6`. No live v6 asset was opened in preparing it. All production review starts only after the pinned successful terminal boundary and strict public loader verification described there.

Expected final coverage is6,456 assets:2,408 PNG,2,024 PDF and2,024 SVG. There are2,024 three-format figure families and384 additional pair-overhead-social PNG-only figures. Pair families: academic density, academic overhead and three physical-pattern social densities. Cell families: academic/social density and academic/social overhead. The actual final ledger remains authoritative; unique ordered protocol/cell/target identities, kinds and paths must reconcile exactly.

Read-only source inspection confirms `academic_plots._save_formats` exports the same Figure object to PNG/PDF/SVG using `bbox_inches="tight"`; `matched_supplement_figures._save_figure` likewise exports the same Figure, PNG at300dpi and vectors without that raster DPI override. This supports a shared-content export contract; it does NOT prove backend-identical text metrics, clipping, font embedding or vector usability. Fixed-canvas pair social overhead has no vector counterpart. No artist, source or output should be regenerated during review.

## Existing Tools Available

- Poppler `pdftoppm` and `pdfinfo` are available through the Codex bundled override binaries; system `pdftotext` is available.
- Validation Python has `pypdf` and Pillow. Bundled Python `/home/dieterolson/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3` also has `pdfplumber`.
- Bundled Node `/home/dieterolson/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node` has `sharp` (SVG rendering through its existing library stack) and `playwright`. No need to install an alternative PDF engine, browser or SVG library. `cairosvg`, PyMuPDF and `rsvg-convert` are absent in the inspected runtimes.
- Existing validation scripts `supplement_v3_graphical_review/root_pw_primary_cells/executed_label_check.py.txt` and original v4 club label checks demonstrate saved-summary/PDF-text reconciliation. Reuse their extraction conventions, correcting wrapping normalization against actual schema, rather than inventing a new scientific parser.

Pin executable/library versions and qualification-script SHA in the new receipt. Tool presence is established; no current vector was rendered yet. A small known-good/known-bad fixture qualification is required before using any new operational bounds/comparison check as evidence.

## Pass 1: Every File and Every Scientific Identity

After the terminal gate, use the existing strict loader and harness to bind the final ledger, seven expected numerical hashes and producer sources. For every asset record:

1. Resolve a contained canonical path; reject missing/duplicate/unbound paths and SHA mismatch. Record figure family, ordered source protocols/cells, target, population type/count, format and bytes.
2. For each PDF, require successful parsing, one expected page, finite positive MediaBox/CropBox, no encryption/password requirement, usable content/resources and successful render. Record warnings rather than hiding them. Check missing/substituted font resources with `pdffonts` if available; Type3/path glyphs are not inherently defective.
3. For each SVG, parse XML safely, require SVG root and finite positive viewport/viewBox, reject broken local references, and inventory embedded images, clipping references and external dependencies. Normal RDF namespace/metadata URLs are not external assets. Matplotlib's glyph-path `<use>` references and data-URI raster image layers are valid. A network-dependent stylesheet/font/image is a portability finding.
4. Extract PDF text and SVG text/comments; normalize whitespace/hyphen wrapping only, without deleting numeric signs or protocol punctuation. Match required protocol order, displayed target rounded by the actual caption rule, physical pattern, condition labels, counts and units to numerical records and ledger metadata. Different kinds legitimately carry different population labels: common-six density is not raw trajectory support. Use exact family-specific expectations. Do not require SG count on a figure that depicts only endpoints.
5. SVG comments alone are not visible-text proof. They provide an identity cross-check; rendered glyphs must remain present and legible. PDF text extraction also cannot establish layout. Record these as content checks, not visual approval.

All4,048 vectors receive the above checks; none is covered merely by inspecting a sibling PNG. Link each result to the exact SHA rather than a mutable filename.

## Pass 2: Render Every Non-Reused Vector Once

Use old visual acceptance only when the vector bytes AND exact scientific identity/kind match an accepted historical artifact. Metadata-normalized PDF equality is not byte equality and must not be labeled as such. Rejected long-caption/gutter graphics never supply visual acceptance.

For all other vectors, perform one deterministic read-only rasterization, streamed in the same36 batches as PNG review. At most two raster workers; no producer/statistical subprocess. Examples (paths supplied from the bound ledger, not glob guessing):

```bash
pdfinfo "$PDF_PATH"
pdftotext -bbox-layout "$PDF_PATH" "$OWN_BATCH_DIR/pdf_text_boxes.xhtml"
pdftoppm -f 1 -singlefile -r 144 -png "$PDF_PATH" "$OWN_BATCH_DIR/pdf_render"
```

For SVG, use bundled `sharp` to render at the corresponding PDF raster dimensions (preserving aspect ratio and white background), with source pixel limits enabled. Keep renderer metadata/version. Do not force unequal aspect ratios to fit. PDF and SVG render output creation is review-only rasterization, not alteration/re-export of the deliverable.

For efficiency, retain compressed legible contact sheets, render hashes, dimensions, extracted bounds and only full-size images needed to inspect findings/representatives; discard only this review's own temporary raster buffers once their checks and evidence are saved. Never delete producer logs/assets. Check disk budget before starting; do not keep4,048 uncompressed full-size pages unnecessarily.

144dpi is an inspection rendering choice, not an acceptance threshold. Full-resolution or higher-resolution crops are required whenever small labels cannot be read. The physical PDF/SVG dimensions, text size and plotted limits remain authoritative.

## Pass 3: Bounds, Clipping and Cross-Format Comparisons

Use Poppler bbox output or `pdfplumber` to check all extracted text rectangles against the actual page box. Record outside-page text and tight boundary gaps. Text rectangles may overestimate painted glyphs (descenders/ascenders); an intersection is a flag, not automatic proof of glyph collision. Intentional data marks clipped to axes are not caption defects. The prior four-point neighbor-label and six-point decorated-footer requirements apply only where already qualified; do not invent a global threshold for every word box.

PDF glyph coordinates and SVG viewport dimensions can differ subtly from Agg PNG tight-bbox metrics. Measure physical aspect/dimensions and report the difference, rather than assuming exact equal size. Raster comparison should align physical page content without arbitrary crop/stretch. Never trim off a caption to improve a comparison score.

Cross-format image differences are diagnostic:

- Exact decoded-pixel equality at the same rendering settings is strong evidence of identical visible content and can reuse an already-reviewed raster's visual decision, with both file identities retained.
- Nonzero differences are expected from antialiasing, font hinting, raster density layers and stroke tessellation. Do NOT invent an SSIM/MAE cutoff to call them equivalent. Use difference images/edge overlays and text/geometry checks to localize changes; significant-looking changes require direct inspection.
- Reuse the qualified shared-Figure export path as contract evidence, with every vector's independent content/parse/render checks. It cannot waive a missing label or renderer warning.
- If a PDF/SVG backend shows any genuine clipping, glyph loss, displaced axes, missing HDR/image layer or changed scientific label, inspect the entire affected exporter/kind subset, not just a second convenient sample. Unexplained differences remain unresolved.

Avoid building a custom SVG path/transform/font engine. Use the existing renderer to resolve glyph paths and clip transforms. If clipping cannot be assessed reliably from its rendered output, inspect that file directly at full size instead of asserting an XML-only proof.

## Human Review Coverage and Honest Acceptance

Every one of2,408 PNGs receives either new explicit visual inspection or exact-byte, identity-matched prior acceptance. For vectors, all4,048 receive independent hash/content/parse/render/bounds checks. Newly rendered vector contact sheets support broad visual screening. Open PDF/SVG raster pages side-by-side with the sibling PNG for a predeclared representative set covering:

- Each distinct exporter and all nine vector family types (five pair, four cell).
- All clubs and contrast kinds; ordinary/strong curves; both SDs/profiles; narrow ranges and widest ranges; long covariance captions; minimum-support cells/pairs.
- Driver speed-bound controls, PW fixed-target aliases and cross-protocol comparisons.
- Every warning, out-of-page text flag, aspect anomaly, content mismatch and unexplained render difference.

Prefer reviewing vectors for the same representative identities already selected for full-size PNG inspection, saving navigation and giving direct format comparisons. Contact generation or successful parse is never recorded as human inspection. A per-file coverage table distinguishes:

`hash_verified`, `content_checked`, `parsed`, `rendered`, `bounds_checked`, `prior_visual_sha_reused`, `human_contact_viewed`, `human_full_size_viewed`, `finding`, and `disposition`.

For an artifact not directly viewed by a human, the final statement must say **automated vector-format checks plus qualified shared-export contract and representative visual review**, not “all PDFs/SVGs visually reviewed.” If user-facing acceptance requires literal individual visual review of every vector, inspect all vector contacts; do not redefine the requirement or pretend one PNG covered them. This plan retains the choice transparently rather than hiding it in a pass flag.

## Evidence Closure

Aggregate full6,456 path/hash coverage with exact2,024 PDF,2,024 SVG and2,408 PNG counts and all410 scientific records. Keep structural/content/render success separate from visual decisions and unresolved findings. Rehash sources, terminal receipt, ledger and outputs at the end; retain seven-number-file equality. Any mutation invalidates the affected snapshot. Final prose should report actual automated and human coverage counts separately.

This qualifies a graphical publication component only. It does not compile the user's editable LaTeX, accept the final assembled report PDF, validate the model empirically, or establish PR/transport completion. No additional scientific calculation, Monte Carlo run or bootstrap is necessary for this review.

## Qualified Parsing Helper

The canonical `scripts.research.shot_pattern_analysis.vector_inspection` module now provides hash-bound PDF/SVG parsing/resource diagnostics through `inspect_pdf`, `inspect_svg` and `inspect_vector`. See [Qualification](../vector_inspection_qualification_v1/README.md). Its 37 regression tests and completed original sample checks pass. Use it only after this plan’s terminal/strict-loader boundary; rendering, scientific label reconciliation, bounds and visual decisions remain separate. Do not add it retroactively to producer source maps or treat its `visual_approval: false` as acceptance.
