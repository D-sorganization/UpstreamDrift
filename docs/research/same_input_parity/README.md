# Same-Input Cross-Engine Dynamics Parity Reference

Start with the [maintained LaTeX reference](same_input_parity.tex). It is an
editable standalone research source (see `AGENTS.md`, Modeling Reference
Documentation) for epic #11605, child P-9 (#11614); the engineering design
manual source remains `manuals/upstreamdrift`.

| Item | Status |
| ---- | ------ |
| Model definition, KKT formulation, closure projection | Documented |
| Pointwise parity (L1) for MuJoCo, Drake, Pinocchio | Measured (PR #11615) |
| Trajectory (L2) and closed-loop (L3) parity, input bundle | Planned (#11607-#11610) |
| OpenSim, MyoSuite, Simscape | Planned (#11611, #11612) and deferred (#11613) |

## Build and Reproduce

```bash
cd docs/research/same_input_parity && tectonic same_input_parity.tex
MUJOCO_GL=egl PYTHONPATH=.:src python3 -m scripts.same_input_pointwise_parity
```

The generated PDF is not committed.
