# Footer Repair Independent Acceptance

Status: **Accepted for the Reviewed Social-Pair Layout Component**.

The repair measures the actual decorated axis bounds and reserves six points above the footer. Independent replay of the regression and academic plotting suite passed all 19 tests. The saved-data rerender exited 0 and generated 27 source-bound assets plus nine 1080-pixel previews for Driver, Seven Iron and Pitching Wedge, comparing shaft-rotation SD 1° and 2° at lean 0°, curve ×1 and Combined 2 delivery noise. No simulations were run.

All nine previews were visually inspected. Footer, axis labels, main/marginal ticks and legend are separated; metadata, HDR and marginals are legible. Measured footer clearance is six points in every figure, and the minimum font em-size is 12.5 pixels at a 1080-pixel canvas width across all text artists. No geometric issue was detected. Common-six populations are 10,000, 9,996 and 9,999 respectively; the disclosed 600-dot overlay does not reduce the density/HDR population.

All 27 artifact hashes and nine preview hashes passed. Review-source and original-input hashes stayed unchanged; only `academic_plots.py` differs from the accepted execution snapshot, and that difference remains a reporting revision. The exact render receipt, tests and nine previews are preserved in `footer_repair_review_v2/`; `footer_repair_independent_acceptance.json` binds this review. The earlier 84-asset diagnostic with the footer defect remains historical evidence.

PDF/SVG bytes were verified but not independently rendered. This does not approve supplemental Monte Carlo execution or the final study narrative. Future final publication assets require their own source-bound check. The Straight subtitle wraps “Curve ×1” across two lines without losing content; that is a minor style observation, not a blocker.

## Reproduction

From this isolated layout worktree, choose a new output directory:

```bash
OPENBLAS_NUM_THREADS=1 QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy /tmp/shot-pattern-validation-venv-20261009/bin/python -m docs.research.shot_pattern_analysis.phase2.review_followups.render_footer_repair_review /home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift /tmp/footer-repair-new-review
```

The executed render harness is preserved byte-for-byte in the evidence bundle. Its later cleanup added only narrow type-checker comments for dynamic instrumentation and a known two-condition tuple; `harness_typing_only_transition.json` records equal Python ASTs and both hashes. Scoped Mypy and Ruff then passed. The original render receipt still identifies the exact executed bytes.
