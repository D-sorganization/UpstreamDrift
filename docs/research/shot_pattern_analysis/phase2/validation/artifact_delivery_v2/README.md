# Phase-Two Artifact Delivery Evidence v2

This directory records local bundle construction, offline restoration, exact release staging, and the cell PNG hash bridge. It does not claim a GitHub release or remote download.

- Supplemental source is STUDY `docs/research/shot_pattern_analysis/phase2/supplement_report_v2_layout_repair/`; the frozen baseline STUDY checkout remains untouched. The archive API used is DELIVERY `scripts/research/shot_pattern_analysis/archive_data.py`.
- `local_fresh_restoration_receipt_v2.json` records verified primary (690), control (548), and supplement (6,465) file inventories and exact member hashes after offline extraction. It is not remote restoration evidence.
- `release_staging_inventory.json` binds all 19 unique actual basenames to size/SHA256 in `/tmp/shot-pattern-delivery-resume/release-assets-v2/`. The direct report files are byte-identical to accepted REPORT.md/PDF. No upload occurred.
- The six direct social PNGs are selected from lean0, curve1, rotation2, combined2 (shaft-twist SD 2 deg) for driver, seven-iron, and pitching wedge. Their hashes match the accepted primary archive manifest. This selection is explicitly not face SD 2 deg.
- `cell_png_hash_bridge.json` proves all 104 current child-v2 cell PNGs byte-match the previously accepted 26-cell visual review hash inventory (review receipt SHA 19c398c21416cd96f6ab796359d6e38542cbc441418b70549f8da586e44a71f6; inventory SHA 9c9083eb06e58f231bf641e901d210aa00b71f354f7c7fbe1ce66b58ca29867b). The v2 final PNG coverage receipt accounts for the other 2,304 accepted comparison PNGs. Together these cover 2,408 PNGs.
- The PDF text-box painted-ink and repaired native-vector review receipts remain in `../supplement_v6_vector_review_v2/` and `../supplement_v6_final_graphics_review_v2/`; see the artifact plan for limitations.

The acceptance and delivery commands are in `../review_followups/ARTIFACT_DELIVERY_PLAN.md`. Release creation, upload, and fresh remote download/restore remain unperformed.

The exact DELIVERY pre-PR attempt is bound by `pre_pr_receipt.json`. It ran against base `fa7ee92f5cddd753099d7418fc974b142b7ebb54` and tested commit `dac6618c58834b671acc7cf7e1137f7a548cee73`; it exited 1. Lint/format, diff mypy, consistency, policy/fragments, and workflow lint passed. Affected tests had 34 failures and 7 setup errors; all 37 nodes were in unchanged test files outside the scoped diff, which is path-scope evidence only, not proof of runtime baseline failures. Semgrep still flags the intentional `xml.parsers.expat` import in `vector_inspection.py:12`; Fleet Policy has the recorded DEVELOPMENT_LOG debt. Remote publication is blocked pending disposition of these gates; no bypass was used.
