---
issue: 12309
summary: "Migrate final figure builder calls to FigureOptions and verify real writer integration and intended-output parity"
dl_state: "in_progress"
next_step: "Complete real writer regression tests and include the fix in the shot-pattern integration PR"
branch: "codex/shot-pattern-refinement-integration"
paths: "scripts/research/shot_pattern_analysis/build_final_figures.py"
---

The pre-fix builder raises TypeError on legacy presentation/dpi keywords. Parity evidence must identify the keyword-to-options adapter as intended behavior, not claim the broken original rendered successfully. Frozen study artifacts remain unchanged.
