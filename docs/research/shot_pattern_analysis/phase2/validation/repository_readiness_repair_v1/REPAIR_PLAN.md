# Validation Script Typing Repair Without Rewriting Evidence

The pre-PR log reports 33 Mypy diagnostics in ten runnable validation scripts. Keep their `.py` paths and keep them under normal checking. Do not exclude, suppress, or rename runnable files to evade the gate.

Before edits, preserve the exact ten executed source files in one deterministic source archive, with a manifest of original path/SHA and the existing commit32877b6ecc. Bind the archive and old source hashes in a NEW transition receipt. Existing review JSON/logs stay byte-for-byte unchanged and continue to describe the historic source, not the repaired current file. The archive is an evidence copy, not a substitute for repairing the checked `.py` file. Git history also preserves the original bytes.

Make only these targeted changes:

- Four Decimal exponent sites (audit_interactions.py, final_check.py, audit_candidate.py and root/verify.py): parse once, reject nonfinite Decimal input, narrow `.as_tuple().exponent` to int with an explicit domain check before exponentiation. The existing inputs are finite displayed numbers; NaN/Infinity should raise a clear ValueError. Add finite/nonfinite focused tests; no broad cast or ignore.
- audit_candidate.py: rename the reused `key` variable for the protocol/cell tuple. Preserve extraction and lookup semantics.
- root/verify.py: annotate empty effects/seen containers with actual element types.
- capture_bounds_gui.py: bind scrollbar once and reject None before maximum/setValue. Do not change GUI settings or launch native GUI; test offscreen only if needed.
- reproduce_saved_cell.py: annotate the heterogeneous capture dictionary accurately (or a small TypedDict), preserving exact figure arguments and values.
- test_review_batch.py: check spec and loader before module construction; use monkeypatch.setattr with explicit raising policy for fake module attributes, or a typed fake namespace matching the imported module contract. Avoid arbitrary dynamic attribute assignments that Mypy treats as real module APIs.
- review_batch.py: explicitly reject a None numeric-reference result after the optional loader (CLI requires a path); annotate the empty pending list. Preserve the terminal gate and all failure behavior. Rerun its existing22 negative/positive tests and record the NEW harness SHA; prior qualification remains historical.
- driver_curve_bounds_reporting_review_v1/preflight.py: use separate names for record dictionaries and path strings rather than conflicting loop-variable reuse.
- selected_renderer_representative_v1/qualify.py: exclude class objects before `asdict`; replace direct Figure.savefig method assignment with scoped `unittest.mock.patch.object` (same wrapper behavior and guaranteed restoration). Tests should verify interception and restoration without rerendering28assets merely for typing.

Run scoped Mypy and Ruff/format on all ten repaired files, then relevant fast tests. Preserve exact red diagnostics and green logs. For annotation/variable-name-only changes, normalized AST comparison can document behavior preservation; for explicit new guards/scoped mocking, use negative/positive tests instead of claiming AST equality. Do not overwrite old acceptance JSON by rerunning an historical script whose output paths are fixed. If replay is needed, run against temporary copies/new output paths and label it a new validation, with unchanged input hashes.

The article audit scripts include paths to preserved rejected candidates and output receipts. They are historical reproducible procedures, not permission to rewrite current article/evidence. Their type checks can run without executing their top-level output writes. Preserve that distinction in the transition note. The live producer's frozen source and data are outside this repair scope.
