# Across-Lean Interaction Independent Qualification

Status: **Accepted Component Candidate**, bound to the four BOUNDS source hashes in `independent_qualification.json`. MAIN production files and existing output remain unchanged.

## Verified Estimand and Population

The producer now applies its existing sorted common-six scoring vectors and paired bootstrap to across-lean comparisons. The contrast is `(curved − Straight) at B − (curved − Straight) at A`, evaluated rowwise on the same six-population supported-index intersection. Separate confidence-interval endpoints are never subtracted.

The actual selected 10,000-shot Driver 0°→5° control has 9,940 common indices; its Draw relative SG change is +0.005980075434717972 and Fade change is −0.0013974497727458593. The corresponding PW control has 9,999 common indices, Draw −0.0003513244825749287 and Fade +0.0006868011750515057. Independent index assembly and rowwise contrasts agree within 1e−12. All prior per-pattern metric objects are exactly unchanged. The real emitted results render through the report selector and precision-aware table, saved as `ACTUAL_CONTROL_TABLE.md`.

## Report Boundary Repair

Two failing regressions reproduced successful selection of a geometrically available pair with missing or failed relative-scoring fields. The narrow selector repair now records those cases as explicitly unavailable and preserves the scorer’s actual reason. Forty-five focused tests pass, with Ruff, formatting, scoped Mypy and diff checks clean. All four changed files remain below the 1,200-line cap.

## Provenance and Remaining Gate

Public source validation, verified alias resolution and the accepted execution-transition receipt were exercised. Three harness construction failures are preserved: an unverified alias, an incorrect inventory path and an unnecessary transition receipt for identical PW snapshots. They were corrected in the harness without weakening production guards. The executable helper was preserved byte-for-byte as `.py.txt`; its SHA matches the actual-data receipt.

This accepts the component, not the pending full pipeline or final scientific report. After the live job ends, integrate the reviewed source and regenerate all 56 across-lean effects from saved data, retaining the prior receipts. Driver speed-bound and nominal-delivery rematching remain part of the contrast; these results do not isolate a causal contribution of lean or distance and do not establish empirical accuracy.
