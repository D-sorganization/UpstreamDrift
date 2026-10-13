# Partial Report Composition Layout Review

**Changes required.** All eight rendered pages were inspected. This is a partial diagnostic, not the complete report, PDF acceptance, or an alternative LaTeX document.

Pages 2–5 contain a concrete readability failure: the nine-column resolved-results table splits scientific notation, decimal values and headings inside tokens. Page 2 splits `+7.462e-05` after `e-`; the interval digits are similarly fragmented. This must be corrected without reducing numerical precision or the readable font floor. Prefer narrower club/protocol-labelled blocks and/or separate scoring and population tables while retaining exact identity and population mappings.

Pages 5–8 retain readable effect estimates, but the repeated full source identifiers make rows tall and hard to scan. A nearby exact ordered-condition key with shorter table references would preserve provenance more readably. No data omission or page-edge clipping was observed. Page 1 summary/correlation tables and body prose are readable; page numbers, headers and repeated table headings are present.

The renderer currently allocates 22% of width to the first column and divides the rest equally. Nine columns leave insufficient width for precise intervals. Merely avoiding clipping is not an adequate regression: each numeric token must remain intact at the retained font size.

The initial diagnostic wrapper skipped H2 between H1 and the supplied H3 and failed the PDF outline contract. Adding H2 in the preview wrapper fixed that harness error; its failed log is preserved. Neither the source Markdown fragments nor the PDF builder was modified. Production-relative links in this relocated preview are not accepted link checks.

`review.json` binds the four relevant producer source hashes, Markdown, PDF, eight page PNGs and render logs. Reproduction used the existing `python -m scripts.research.shot_pattern_analysis.build_pdf` and `pdftoppm -scale-to 1600 -png` commands. No full report builder, simulation or statistical rerun was invoked.
