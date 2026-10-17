# Vertical Force-Plate Contract Reference

Start with the [maintained LaTeX reference](force_plate_contract.tex) for
OF-01 (#13065) under epic #13061. It is an editable standalone research
source (see `AGENTS.md`, Modeling Reference Documentation). The engineering
design manual source remains `manuals/upstreamdrift`.

| Item                                                   | Status                                  |
| ------------------------------------------------------ | --------------------------------------- |
| Trial validity rules (threshold, net CoP, noise floor) | Done, unit-tested (`test_contracts.py`) |
| Yaw-only proper placement, absolute tolerance only     | Done, unit-tested                       |
| Observed view: NaN shear and moments, observability    | Done, unit-tested                       |
| Plate hardware measurement                             | **Deferred** (OF-HW #13082, DV-13061)   |
