# Primary Final Archive Restoration Review

Status: **Passed for Packaging and Restoration**. The independent executable audit exited 0 after the archive builder's exit 0 was confirmed. This is not a new scientific or presentation qualification.

The 15 published ZIP parts restored 2,097 files into a fresh temporary directory. Every restored and original file's SHA-256 and byte size matched the archive manifest, with exact inventory equality. The complete restored `load_matrix_report` result equaled the original report: 324 planned records, 270 completed, 54 unavailable, and none unrun. The plan, execution snapshot, all cell manifest hashes and statuses match the prior independent primary-final scientific audit covering 8,100,000 rows.

All original files, archive parts/manifest, review source files, and the prior scientific audit were unchanged across the review. The temporary restoration directory was removed successfully. Existing pilot restoration scripts and receipts were not changed. No simulations or scoring were rerun.

## Evidence

- Receipt: `primary_final_archive_restoration.json`, SHA-256 `37c1deda30d11fb7f6762080b76cdf62cae6cbb948229464ec5e100b76e8339a`.
- Archive manifest SHA-256: `62af2c77fcbf2b0f965fe0eec28fb3f49f04d27c072ea2f0baaf430251354dcc`.
- Prior scientific audit SHA-256: `a9bf9e87eb3c72c21e002e9010a10d48f85cb1c71a2e0f250169485a54982e7d`.
- Full process output: `PRIMARY_FINAL_ARCHIVE_RESTORATION_LOG.txt`.
- Ruff check, Ruff formatting, scoped Mypy (`--python-version 3.12 --follow-imports=silent`), and module CLI help passed. This helper exercises the already qualified archive and report APIs; it does not implement a new archive format or numerical algorithm.

## Reproduction

From the study repository root, use a **new** receipt path (existing receipts are rejected):

```bash
OPENBLAS_NUM_THREADS=1 QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy /tmp/shot-pattern-validation-venv-20261009/bin/python -m docs.research.shot_pattern_analysis.phase2.validation.restore_final_primary_archives /tmp/primary-final-restoration-new.json
```

The published `phase2/data_archives/final_primary_v1`, original `phase2/final_v1`, and hash-bound prior audit must be present. The receipt records the actual helper/API source hashes used. This proves byte preservation and saved-report loading for the primary final dataset only; controls, supplements, empirical validation, and final figures remain separate acceptance scopes.
