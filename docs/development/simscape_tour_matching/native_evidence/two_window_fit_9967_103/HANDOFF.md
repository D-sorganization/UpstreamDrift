# MS-61 Topology + Full-Marker Terminal Qualification (#10348)

Software contracts for Simscape R2025b topology classification and
full-marker terminal disclosure. **Native G1 is blocked** — this
directory does not invent a physical pass.

## R2025b Runtime / License

- Release: **R2025b** (`25.2.0.3177638 (R2025b) Update 5`)
- Host: **DeskComputer** (from run-102 manifest)
- License cap assumed: **false** (no inferred 1000-block ceiling)

## Derived From Run-102

- Full-marker terminal: **40.303 mm** (G1 ceiling 35 mm)
- Head-cluster terminal: **72.258 mm**
- Body-excluding-head (diagnostic only): **33.673 mm**
- Topology profile: `reduced_27_no_neck` (no independent neck DOFs)
- Pinocchio↔Simscape max Euclidean: **0.061 mm**

## Receipts

- `topology_report.json`
- `terminal_breakdown.json`
- `native_gate.json` (status=`blocked`)
- `runtime_license_receipt.json`
- `parity_receipt.json`

## Next Native Action (DeskComputer / R2025b Only)

```powershell
PYTHONPATH=. python scripts/matlab/materialize_ms61_topology_receipts.py
powershell scripts/matlab/run_simscape_candidate.ps1 -Run two_window_fit_9967_103 -Fit
```

If neck/model development is required, track under MS-104 (#10378).
MS-102 (#10376) inventory is a soft dependency and does not block
these software contracts.
