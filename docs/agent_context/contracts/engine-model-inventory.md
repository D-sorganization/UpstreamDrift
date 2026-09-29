# Engine Model Inventory (MS-102)

## Responsibilities

`EngineModelInventory` loads `src/config/engine_model_inventory.json`, which
indexes runnable engine/model packages derived from `models.yaml`,
`engine_capability_matrix.json`, and `ENGINE_TIERS`. It does not invent a
second engine catalog. `qualify_package` runs fail-closed smoke steps and
writes content-addressed receipts.

## Evidence

- Implementation: `src/engines/model_inventory.py`
- Ledger: `src/config/engine_model_inventory.json`
- Tests: `tests/unit/engines/test_model_inventory.py`
- Structural receipts: `docs/development/matched_swing_program/evidence/ms102/`

## Failures

Missing assets and hash mismatches fail with remediation. Native PASS requires
an installed engine SDK on a supported host (consume MS-103 preflight). MyoSuite
flagship packages remain repair until MS-51 (#10344). Simscape requires MATLAB
R2025b.
