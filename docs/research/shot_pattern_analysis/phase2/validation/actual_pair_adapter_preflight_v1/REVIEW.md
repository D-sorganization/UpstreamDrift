# Actual Ordered-Pair Adapter Preflight

The read-only check completed with exit 0. All 384 existing numerical comparison records reconstructed successfully: 56 across-lean, 164 curve-strength and 164 shaft-twist SD comparisons. They reference 328 distinct executed source cells and include 16 occurrences of the existing verified control aliases. There were no schema, source-order, target, configuration, count or alias-proof failures.

For every referenced cell, the check verified the comparison's plan and manifest hashes, all files listed in that manifest, and the saved summary/configuration and CSV identity. The production adapter then validated its source inputs and reconstructed the ordered renderer references, including the requested fixed-target configuration for aliases. The two numerical input files and every referenced manifest remained unchanged through the check. Exact identities, hashes and all 384 outcomes are in `review.json`; `check.py` reproduces this preflight without rendering or computing statistics.

This is provisional component evidence only. The live parent has not yet published its final receipt and figure ledger. The check therefore constructs a provisional receipt-shaped source snapshot from the hash-verified saved files and projects only the ledger identity/count fields from the existing numerical comparison records. It does not substitute for the unchanged full parent verifier, qualify completed-parent provenance, or establish selected-renderer metadata parity. Those checks remain required after the parent publication and candidate renderer are complete.

Review receipt SHA-256: `515bb797dd275cbb5993dbe1b22d05965c7d14c8d3d2341a0c450dd51e93dcca`.
