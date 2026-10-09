# MyoSuite Kinematic Retarget Map Reference

Start with the [maintained LaTeX reference](myosuite_retarget_map.tex). It is an
editable standalone research source (see `AGENTS.md`, Modeling Reference
Documentation) for #11729. The engineering design manual source remains
`manuals/upstreamdrift`, and the decision record is sections 16 and 17 of
[`DESIGN_DECISIONS.md`](../../development/full_body_models/DESIGN_DECISIONS.md).

| Item                                                      | Status                      |
| --------------------------------------------------------- | --------------------------- |
| Source and target frames, one-to-one map, exact inverse   | Documented, unit-tested     |
| `NeckInputY -> neck_flexion` sign (-1)                    | Derived by FK (section 16)  |
| Hip, scapula and neck lateral-bending sources             | Omitted, with reasons       |
| Pelvis orientation and shoulder girdle in MyoSuite replay | Not represented (follow-up) |

## Build and Reproduce

```bash
cd docs/research/myosuite_retarget_map && tectonic myosuite_retarget_map.tex
python3 -m pytest tests/unit/engines/myosuite/test_retarget.py -q
```

The built PDF is not committed.
