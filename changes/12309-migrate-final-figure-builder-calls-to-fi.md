---
issue: 12309
summary: "Migrate final figure builder calls to FigureOptions and verify real writer integration and intended-output parity"
dl_state: "in_review"
next_step: "Include validated fix in the shot-pattern integration PR; merge remains pending"
branch: "codex/shot-pattern-refinement-integration"
paths: "scripts/research/shot_pattern_analysis/build_final_figures.py"
---

The pre-fix builder raises TypeError on legacy presentation/dpi keywords. Parity evidence must identify the keyword-to-options adapter as intended behavior, not claim the broken original rendered successfully. Frozen study artifacts remain unchanged.

Validation: 22 focused tests pass, including all three native writer interfaces and four byte-identical ledger fixtures. An explicitly adapted pre-refactor baseline matches density rows, captions, and RGBA pixels.
