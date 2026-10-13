---
issue: 12309
summary: "Migrate final figure builder calls to FigureOptions and verify real writer integration and intended-output parity"
dl_state: "in_review"
next_step: "Include validated fix in the shot-pattern integration PR; merge remains pending"
branch: "codex/shot-pattern-safe-followup"
paths: "scripts/research/shot_pattern_analysis/build_final_figures.py"
---

The pre-fix builder raises TypeError on legacy presentation/dpi keywords. Parity evidence must identify the keyword-to-options adapter as intended behavior, not claim the broken original rendered successfully. Frozen study artifacts remain unchanged.

Validation: 22 focused tests pass, including all three native writer interfaces and four byte-identical ledger fixtures. An explicitly adapted pre-refactor baseline matches density rows, captions, and RGBA pixels.

The historical pilot diagnostic renderer had the same obsolete writer interface. Its bins=48 setting now uses FigureOptions and passes a native writer regression test; archived receipt parity is separately tested with explicit fixture adapters.

The historical footer-review renderer now passes a typed PairSocialContext to the current pair writer. Synthetic checks preserve the six point populations, context, and geometry diagnostics; native-writer coverage verifies real PNG/PDF/SVG assets. The original source-transition guard and frozen artifacts remain unchanged.

The new native integration render measured74.18 seconds and exceeded the default60-second watchdog. It is marked slow with a240-second test-only watchdog; measurements are recorded on #12309 (comment6114197016). Existing numerical tolerances and the1ms performance budget are unchanged.

The successor also restores the original typed public keyword surfaces for frozen
reproduction callers (all four academic figure controls and pair context fields).
Adapters reuse the modern options/context implementations and reject mixed input.
Four new red-to-green compatibility tests, scoped mypy, Ruff and format checks
pass; the final canonical gate remains pending. No frozen script or study output
changed. Exact evidence: `safe_followup_v1/writer_api_compatibility.txt`.
