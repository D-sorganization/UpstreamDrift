# Historical Original-Protocol Compatibility

## Finding

The restored, accepted 72-cell control-pilot plan failed report validation after configurable matching bounds were added. A recursive comparison of all 72 configurations found exactly two differences per cell: the historical plan omits `match_face_bounds_deg` (current default `null`, meaning the legacy reference-centered face window) and `match_speed_factors` (current default `[0.75, 1.25]`). Cell identities and every other field matched. `null` does not mean an unbounded face search.

The same direct comparison also affected original final primary/control protocol validation. It was a report compatibility defect, not changed flight data or a reason to widen the original search bounds.

## Narrow Remedy and Evidence

The report layer now supplies only those two known defaults on a copied configuration for protocol comparison. Explicit values and all other fields remain unchanged. The complete factorial configuration comparison still rejects expanded face/speed bounds, altered seeds and changed tolerances. Raw saved configurations, accepted plan bytes, audit hashes and immutable acceptance documents are never rewritten.

Three regression failures (historical primary, historical control, and hash-bound historical pilot) were observed before the fix. Nine focused tests then passed, including the restored real accepted control-pilot test. The actual MAIN primary 324-cell and control 72-cell final plan configurations also passed the corrected validator without changing their bytes. This replay validates protocol compatibility only; it does not claim completed control data or replace data audits.

Both report protocol consumers use the same narrow compatibility helper. Raw plan-versus-cell equality in `load_validated_cell` remains strict because both records belong to the same execution; no normalization is needed or permitted there. Supplemental explicit-bounds validation and pair-comparison semantics remain unchanged. The report module is outside the execution source snapshot, and no physics or presentation-renderer source was edited.

Scoped Ruff and Mypy (Python 3.12, silent import following, fresh cache) passed. Reproduce the focused checks with `OPENBLAS_NUM_THREADS=1 /tmp/shot-pattern-validation-venv-20261009/bin/python -m pytest -q tests/tools/shot_pattern_analysis/test_original_protocol_compatibility.py tests/tools/shot_pattern_analysis/test_matched_supplement_pipeline.py -k "original_protocol or historical or common_targets"`. Restore the accepted pilot plan through the documented archive process before the real-pilot check.
