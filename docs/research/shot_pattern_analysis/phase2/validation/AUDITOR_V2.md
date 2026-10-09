# Independent Auditor Version Two

`independent_final_audit.py` now reports `independent-saved-data/2` and the
explicit data stage. Original 324/72-cell checks and all numerical comparison
tolerances remain unchanged. Historical receipts retain their original source
hashes; in particular the 16-cell partial audit at commit `4b98b97b47` used
script SHA `e7452f6dedb9e52191d31ebc405b99f58b359720c5c7f065f7a68cb8fe7dd86e`.
None of those receipts was regenerated or reinterpreted as full acceptance.

The new helper compares supplements against the committed 26-entry inventory,
its protocol-plan hashes, and the hash-bound original primary configuration.
Only the three named 18/4/4 protocols are accepted. Every full configuration is
checked after exact declared bounds/target/stage overrides. Duplicate JSON keys,
missing/extra/duplicate cells, and changed noise/seed/physics are rejected.
Pilot requires exactly 32 attempts per pattern; final requires 10,000. The CLI
requires `--supplement-stage pilot` or `--supplement-stage final` explicitly.

`--alias-references` optionally checks the separately emitted four-reference
`matched-study-control-aliases/1` document. References bind source plan,
manifest, saved CSV bytes, nominal raw target, and the equivalent explicit-target
PW0 request. They cannot contain an independent summary, receipt, or execution
claim. Alias review reports zero new executions. A per-protocol successful audit
does not by itself establish completion of all three protocols and aliases.

The row audit continues to verify each primitive sample against the seeded
noise plus its own nominal path/speed and club lean/attack offsets. Comparing
raw delivered path or speed between rematched patterns would be incorrect.
The separate report comparison must use common supported indices and centered
primitive errors across source populations.

Twenty-five focused tests passed, including all three protocols at both stages and
negative alias-integrity cases. Mypy (Python 3.12), Ruff, and diff checks passed.
A zero-cell base-plan CLI smoke check accepted the unchanged 324-cell plan but
correctly reported incomplete/false acceptance; it did not read or score a new
population. No Monte Carlo or large data audit was run for this software change.

Final supplemental audits now require `--supplement-acceptance PATH`. The
independent acceptance check requires all five review gates, the exact accepted
execution snapshot, and all three protocol inventories. Every hashed 32-shot
pilot plan must match its protocol and final configuration after the declared
population change. Changed source, missing protocol, changed pilot input, or a
failed review gate is rejected. This is independently checked by the auditor;
it does not rely on a claim that the production launcher performed the check.
Final 26-cell data and four-reference audits remain pending future accepted runs.
