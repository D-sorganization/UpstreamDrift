# F09c Native Replay Execution

Added a fail-closed F09 execution boundary that binds explicit F01 inventory
rows to Tools T01 native replay bundles, calls the reviewed MuJoCo or Drake
adapter, and validates the complete native output against exact state, input,
policy, model, and time-grid identities. Reports preserve unsupported or
unavailable required rows across all six engines. Execution receipts remain
unqualified and omit local model paths.
