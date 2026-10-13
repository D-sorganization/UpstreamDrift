# Preliminary Repository Gate Result

The diagnostic pre-PR invocation ended with exit 1. Its test phase reported 24 failed, 3,822 passed, 29 skipped, 26 deselected, one xpassed and seven errors. Consistency passed. The study-owned renderer failure is tracked in #12169; the remaining 23 failures and seven errors have unchanged baseline source surfaces, not a fresh baseline execution. Static helper failures were repaired and independently checked afterward (#12160). Owned title and inventory findings are under review; unrelated shared-log policy failures remain recorded. Final fixed-state gates are still required.

Exact output is retained in `original_evidence.zip`; the readable copy normalizes whitespace only. The run targeted `32877b6ecc` but helper repairs occurred while tests ran, so this is diagnostic evidence, not a fixed-tree acceptance receipt.
