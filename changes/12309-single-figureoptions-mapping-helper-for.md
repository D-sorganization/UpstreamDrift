---
issue: 12309
summary: "Single FigureOptions mapping helper for all three final-figure writer calls; overhead writer accepts frozen presentation/dpi keywords with pixel-identical output"
---

`build_final_figures._figure_options(presentation, dpi)` is now the only place
the builder maps its presentation/DPI controls onto `FigureOptions`; the two
detail calls and the pair-density call all use it (bins and HDR masses keep
writer defaults). `write_overhead_trajectory_figure` gains the same frozen
keyword surface as the density writer through `_resolve_figure_options`, so a
legacy `presentation=`/`dpi=` call no longer raises `TypeError`. New tests in
`tests/research/test_final_figure_options_mapping.py` show decoded-RGBA and
text-artist identity between the keyword and options surfaces for both writers
and both presentations, and pin the exact options passed at all three call
sites. The overhead header was extracted to keep the entry point within its
100-line budget; four current-main renders are RGBA-identical before and after.
Frozen artifacts are not regenerated; the manifest source-hash provenance is
computed live from source bytes, unchanged in mechanism.
