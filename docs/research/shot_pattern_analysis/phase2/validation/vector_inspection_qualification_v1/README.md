# Vector Inspection Qualification

The operational helper is `scripts/research/shot_pattern_analysis/vector_inspection.py`; public functions are `inspect_pdf`, `inspect_svg` and `inspect_vector`, each taking a path and its expected SHA-256. Both reads are hash-bound. Use it only after the existing completed-producer and strict publication gates qualify the artifact inventory. The helper neither loads that inventory nor grants visual approval.

Astra independently demonstrated seven failures, corrected them, and passed 37 tests plus lint, formatting and type checks. Root verified the source/evidence hashes, relocated the byte-identical helper, adapted test imports with optional-dependency skipping, then independently passed 37 tests and static checks. The original Gemini candidate and unsupported initial TDD claims remain preserved separately; no initial worker red chronology is asserted.

The completed primary 7-iron PDF and SVG both pass with unchanged file hashes and seven embedded images each. PDF text extraction produces 1,235 characters with no warnings. The SVG uses glyph paths and 89 comments rather than text elements; empty extracted text is expected for that sample, not proof of missing labels. Exact sample hashes and diagnostics are in the receipts.

## Reproduction

Install the existing `shot-pattern-report` extra and use the qualified runtime recorded in the diagnostics. Run:

```bash
python -m pytest -o addopts='' -q tests/research/test_shot_pattern_vector_inspection.py
python -m ruff check scripts/research/shot_pattern_analysis/vector_inspection.py tests/research/test_shot_pattern_vector_inspection.py
python -m ruff format --check scripts/research/shot_pattern_analysis/vector_inspection.py tests/research/test_shot_pattern_vector_inspection.py
python -m mypy --python-version 3.12 --follow-imports=silent scripts/research/shot_pattern_analysis/vector_inspection.py tests/research/test_shot_pattern_vector_inspection.py
```

## Qualification Limits

This is bounded PDF/SVG parsing, resource-reference and extracted-content diagnostics. It is not a general SVG renderer or proof that text is visible. A standard inert Matplotlib SVG DOCTYPE is allowed without external fetching; entities and unsafe declarations are rejected. Viewport cases outside the qualified explicit-dimension/viewBox contract are unsupported by this helper, not universally malformed SVG. PDF warnings are captured through lazy extraction and the logger handler is restored even on failure.

No live v6 asset was read, and no producer source or historical receipt was changed. All-file vector rasterization, bounds/content reconciliation, explicit visual review and final publication acceptance remain pending. Preserve the original source map; this review helper is a separate consumer and must not be inserted retroactively into producer provenance.
