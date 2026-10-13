# Renderer Protocol Precondition Review

The production change remains the four-line early guard from the preserved AGY candidate. It applies after the existing incomplete-status skip and before identity extraction, endpoint loading or rendering. It requires actual resolved-reference `source_protocol` values; generic serialized `protocol_id` fallback remains supported by the separate identity API.

Independent tests strengthen the candidate's side-effect contract across first/second reference, missing/None value and canonical/selective entry point. They trap endpoint/writer operations, Figure construction, Path.open and directory creation, and require the output root to remain absent. Eight cases failed against original HEAD source; the candidate bytes were restored in a finally block. The final focused suite passed 48 tests with six warnings. Ruff, format and Mypy passed; source/test sizes are 1,072/315 lines.

The actual-data parity harness uses historical verified four-pair fixtures, preserved simulation inputs and separate worker processes. It reads no live production graphics. Acceptance requires exact PNG bytes and all figure artist arrays/limits/text metadata, density recipes and non-hash ledger fields to match frozen PRESENTATION402. PDF/SVG byte equality is not required because export metadata can differ. The parity receipt records the final outcome separately; this note alone does not assert completion or whole-publication acceptance.

Historical v6 source/receipt reproduction must continue using its exact frozen producer checkout. This invalid-input repair does not authorize refreshing historical producer hashes or weakening strict loaders. No commit or producer modification was made.
