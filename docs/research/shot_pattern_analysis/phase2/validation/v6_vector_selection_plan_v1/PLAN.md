# V6 Representative Vector Selection Plan

## Status and Boundary

Preparation only: no v6 graphic was opened, rendered, or accepted. Execute selection only after the pinned successful v6 terminal receipt and frozen PRESENTATION strict `load_verified_supplement_artifacts` pass. Require the seven numerical hashes to equal `validation/protocol_name_recovery_v6/expected_numeric_hashes.json`. Freeze the selected identity/path/hash table **before** opening any representative image. This sample supplements all-file parsing, rasterization, content/bounds checks and the full PNG review; it does not replace them.

Use the existing qualified loader, ordered-pair reconstruction adapter and public `matched_figure_identity.pair_comparison_identity`/`pair_comparison_slug`. Do not infer protocol from cell name or reconstruct old filenames. Read actual final ledger paths and match them to public identities. Resolve ties by the complete ordered identity tuple, ascending; for cells use `(source_protocol, cell)`. No statistical or simulation rerun is needed.

## Actual Schema

`REPORT_SUPPLEMENT_COMPARISONS.json['comparisons']` contains 384 records with `comparison_kind`, `sources[0:2]` (`protocol_id`, `cell`, `alias_cell`, `execution_kind`), `source_protocols`, `target_x_m`, `common_supported_n`, `total_n_per_population`, `status`, and per-pattern first/second conditions. Verify the redundant protocol fields agree.

`REPORT_SUPPLEMENT.json['unique_cells']` contains 26 source refs with `cell`, `source_protocol`, `config`, and `summary`. Per-pattern summary is **`ref['summary']['summary'][pattern]`**, not `ref['summary'][pattern]`. Pair source refs are reconstructed through the existing adapter, including verified aliases; these provide configs and saved summaries. Use `config.club.club_id`, `shaft_lean_deg`, `curve_scale`, `noise.standard_deviations`, and actual `noise.correlation` rather than interpreting a profile nickname as covariance.

The final figure ledger supplies `figure_kind`, presentation, physical pattern, source kind, protocols, condition IDs, target, conditioning/counts where applicable, relative path and SHA. Density common-six count is not raw-trajectory count or course-SG count. Preserve those distinctions in the selection table.

## Nine Vector Families

| Record | Family | Required Formats |
|---|---|---|
| Pair | Academic Density (`density_pdf/svg`) | PDF and SVG |
| Pair | Academic Overhead (`overhead_pdf/svg`) | PDF and SVG |
| Pair | Straight Social Density (`density_straight_social_pdf/svg`) | PDF and SVG |
| Pair | Draw Social Density (`density_draw_social_pdf/svg`) | PDF and SVG |
| Pair | Fade Social Density (`density_fade_social_pdf/svg`) | PDF and SVG |
| Cell | Academic Density (`density_pdf/svg`) | PDF and SVG |
| Cell | Academic Overhead (`overhead_pdf/svg`) | PDF and SVG |
| Cell | Social Density (`density_social_pdf/svg`) | PDF and SVG |
| Cell | Social Overhead (`overhead_social_pdf/svg`) | PDF and SVG |

There is **no pair social-overhead vector export**. Its PNG remains covered by the full PNG review, not a fabricated tenth vector family. Supplemental unique cells contain Driver and PW only; seven-iron coverage must come from pairs. Absence of seven-iron unique supplemental cells is an inventory fact, not a silent omission.

## Deterministic Selection Slots

Select all five vector families for each selected pair and all four for each selected cell. Deduplicate identities across slots, preserving every selection reason.

| Slots | Selection Rule | Purpose |
|---|---|---|
| P01–P09 | For each club (Driver, Seven Iron, PW) × each contrast (`across_lean`, `curve_strength`, `shaft_rotation_sd`), choose the eligible pair with smallest `common_supported_n`, ties by ordered identity. | All clubs, all contrast kinds, and low-support captions. |
| P10 | Driver `across_lean`, one source `driver-control-speed-bounds/1`, the other original control; prefer actual shaft SD2 and combined2, then identity. | Explicit cross-protocol speed-bound fixed target. |
| P11 | PW `across_lean` with a verified `provenance_alias` source and `pw-control-face-bounds/1` counterpart; prefer actual shaft SD2 and combined2. | Reused zero-lean primary as fixed-target alias; no independent-run claim. |
| P12 | Pair with the greatest combined count of nonzero off-diagonal covariance entries in the two configs; ties by length of canonical JSON serialization of the two `config.noise` objects (`sort_keys=True`, compact separators), then identity. | Longest real covariance caption, not profile-name inference. |
| P13 | Seven-iron pair containing lean10 shaft-only, prefer curve2 and shaft SD2; then smallest saved downrange SD across its six source-pattern summaries, then identity. | Previously demonstrated narrow-range tick stress. |
| P14 | Driver pair with combined2 and shaft SD2; choose greatest saved downrange SD across the six summaries, then identity. | Broad-dispersion range stress. |
| P15 | Globally smallest positive saved downrange SD across all six source-pattern summaries, then identity. | Independent small-scale proxy outside the named prior case. |
| C01–C03 | One unique cell per supplemental protocol (`pw-primary-face-bounds/1`, `pw-control-face-bounds/1`, `driver-control-speed-bounds/1`), minimizing the minimum of three saved `supported_n`, ties by identity. | All unique execution protocols and minimum-support cell captions. |
| C04 | PW primary cell maximizing number of nonzero off-diagonal correlations; prefer actual SD2, then longest canonical `config.noise` JSON, then identity. | Long social-cell context with strong curve and covariance. |

Eligibility: materialized completed source refs, expected vector family actually available in final ledger, finite target, and at least two common supported rows for pair density. Do not substitute a scoring denominator for endpoint eligibility. Within preference clauses apply the listed criterion in order, not undocumented judgment.

The source-summary SD ranking is explicitly a **range-stress proxy**, not a claim to select the global minimum/maximum exported axis span. Numeric summaries do not contain the final common-six plotted limits. On the selected exported vectors, record the actual visible axis endpoints and check narrow/broad coverage. If the selected examples fail to exhibit distinct narrow/broad scales, add the lexicographically first qualifying case identified in the complete PNG/metadata review; record the trigger and additional identity before opening its PDF/SVG. Do not silently call summary SD the exported range.

Expected initial size is at most **15 pairs × 10 vectors + 4 cells × 8 vectors = 182 vectors** (91 PDF/SVG families), reduced only by exact identity deduplication. The nine club/contrast base slots are disjoint, and three protocol-cell slots are disjoint, so the initial lower bound is **114 vectors**. Publish actual selected counts, not the upper bound as performed work. The sample includes both ordinary/strong curves through curve-strength comparisons and both SDs through shaft-SD comparisons; record actual condition values in the coverage table.

## Missing Coverage and Expansion

An empty slot, missing expected vector, unavailable density, failed alias proof, or absent summary field is an explicit selection failure/pending gate. Record its predicate and reason; do not silently drop the slot or pick another club. A legitimate structural absence (seven-iron unique supplemental cells, pair overhead-social vectors) is documented above. If no positive-SD narrow candidate exists, retain that fact and request a scope decision rather than changing the rule after viewing.

Every parser/resource warning, outside-page label flag, aspect anomaly, missing scientific string, or unexplained cross-format difference adds that exact file and its sibling PDF/SVG/PNG to direct review. A genuine backend defect expands to the complete affected exporter/family subset. These additions are not capped at 182.

## Per-File Record and Decisions

Before views, save slot reasons, full ordered identity, source protocols/cells/alias status, exact target/basis, actual lean/curve/SD/noise/covariance, endpoint/common-six/raw-flight population fields, final-ledger relative path and SHA, sibling PNG identity, and final receipt/source hashes. Keep source numeric-file hashes and the selection-plan SHA.

Use canonical `inspect_vector` and `render_vector` with explicit qualified native tool paths, one native conversion at a time. Record physical viewport, actual raster dimensions, render/diagnostic hashes and warnings. SVG's proportional viewport fit is not equal-native-DPI or pixel equivalence. Inspect each selected PDF and SVG raster, with the corresponding PNG, at readable size; log actual file-specific decisions for axes, HDR layers, marginal labels, legends, row gutters, context and population captions, clipping and footers. Mark automated checks separately from human contact/full-size views. Successful native exit or SVG comments do not prove visible glyphs.

Final source/input/output rehash and strict-loader closure remain mandatory. This plan cannot accept the complete publication, assembled report PDF, editable LaTeX compilation, or model empiricism.
