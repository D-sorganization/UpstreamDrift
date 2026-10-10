# Supplemental Pilot Validation

All three source-matched pilots completed with exit code 0: 18 pitching-wedge within-lean cells, four pitching-wedge common-target cells, and four driver speed-bound sensitivity cells. Each cell attempts 32 shots per physical pattern. All 26 cells completed, and the full independent audits passed all 2,496 saved shot rows. Four pitching-wedge control references were verified as reuse aliases, not additional executions.

The predeclared refinement passed all 333 selected deliveries (1,998 evaluations). Maximum primary-to-fine endpoint change was 0.009001249870284642 m; maximum refined nominal residual was 0.0011381967888155486 m. Both are below the unchanged 0.05 m criterion. This finite grid does not certify every Gaussian-tail delivery or empirical accuracy.

The exact inputs and source hashes are in `supplement_refinement_inputs_v1.json` and `supplement_refinement_v1.json`. Complete per-protocol audits and the alias envelope are adjacent. All 188 archived pilot files passed fresh restoration with exact byte/hash/inventory and full per-protocol report equality. The temporary restoration was removed. The executed restoration command is preserved in `supplement_pilot_restore_executed.py.txt` and bound by its receipt.

Independent pilot-figure review and separate final-run acceptance remain required before the 10,000-shot supplemental trials. These pilot receipts never replace original unavailable cells or establish final publication acceptance.

## Reproduction

Use the qualified environment recorded in [Turnover](../TURNOVER.md). Exact pilot CLI commands and exit times are in `SUPPLEMENT_PILOT_RUNS.txt`; complete audit output is in `SUPPLEMENT_PILOT_AUDITS.txt`. Reproduce into new output paths. Refinement command:

```bash
python -m docs.research.shot_pattern_analysis.phase2.validation.supplemental_refinement --inputs docs/research/shot_pattern_analysis/phase2/validation/supplement_refinement_inputs_v1.json --output NEW_RECEIPT.json
```
