# Control Final Archive Restoration Review

Status: **Passed for Packaging and Restoration**. The independent executable audit exited 0 after root confirmed both the complete control scientific audit and archive builder exited 0.

Three published ZIP parts restored 427 files into a fresh temporary directory. Every restored and original file's SHA-256 and byte size matched the archive manifest with exact inventory equality. The full restored `load_matrix_report` result equaled the original: 72 records, 52 completed, 20 unavailable, none unrun. All cell manifests/statuses, the plan, and the execution snapshot match the completed independent control audit covering 1,560,000 rows.

Original files, bundles, API/helper sources, and the prior audit were unchanged across the review. The temporary restoration directory was removed. Existing primary and pilot restoration scripts/receipts were not changed. The new helper reuses the primary helper's hashing, inventory and contract-check primitives plus the qualified archive/report APIs.

## Evidence

- Receipt: `control_final_archive_restoration.json`, SHA-256 `cc6c433c62e8c309671639f0af723d90dd440cc19da51f13c4d2160d5de4b169`.
- Archive manifest SHA-256: `6c6e89ae6fccb47d19551d59bc0e2060b3faae5ad423a1d402a747f9d0ca5ba6`.
- Completed control scientific audit SHA-256: `41ea81786d110aac0e30c0cb3393534255e9aa1b7e99e59c426ad5a092d3aef9`.
- Process output: `CONTROL_FINAL_ARCHIVE_RESTORATION_LOG.txt`.
- Ruff, formatting, scoped Mypy (`--python-version 3.12 --follow-imports=silent`) and module CLI help passed.

## Reproduction and Limits

From the study repository root, supply a new receipt path:

```bash
OPENBLAS_NUM_THREADS=1 QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy /tmp/shot-pattern-validation-venv-20261009/bin/python -m docs.research.shot_pattern_analysis.phase2.validation.restore_final_control_archives /tmp/control-final-restoration-new.json
```

The published control archives, original control dataset and completed audit must be present. Existing receipt paths are rejected. This establishes packaging/restoration only; it does not rerun scientific calculations or qualify supplemental data, empirical accuracy or final publication figures. The earlier queued audit wrapper's unexplained exit 143 is not used as evidence; this review binds the successful direct audit explicitly.
