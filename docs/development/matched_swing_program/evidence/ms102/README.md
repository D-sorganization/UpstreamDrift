# MS-102 Engine and Model Inventory Evidence

Structural smoke receipts for every inventoried package. Generated with:

```bash
python -c "from pathlib import Path; import json; from src.engines.model_inventory import EngineModelInventory, qualify_package; ..."
```

Native load/step/view receipts require the engine SDK on a supported host
(MS-103 preflight). MyoSuite flagship packages are `ready` for structural smoke
after MS-51 (#10344) golfer scenes; G1 dynamics and 15 mm parity remain open.
Simscape native dynamics use MATLAB R2025b via MS-60 run management.

`inventory_summary.json` is the machine-readable rollup. Per-package
`*_structural_receipt.json` files record resolve/hash outcomes without claiming
native qualification.
