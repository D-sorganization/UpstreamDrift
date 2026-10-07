# Same-Input Cross-Engine Dynamics Parity Reference

Start with the [maintained LaTeX reference](same_input_parity.tex). It is an
editable standalone research source (see `AGENTS.md`, Modeling Reference
Documentation) for epic #11605, child P-9 (#11614); the engineering design
manual source remains `manuals/upstreamdrift`.

| Item | Status |
| ---- | ------ |
| Model definition, KKT formulation, closure projection | Documented |
| Pointwise parity (L1) for MuJoCo, Drake, Pinocchio | Measured (PR #11615) |
| Input bundle `same-input-bundle/v1`, integration policy | Documented |
| Full-swing L2 (50 ms restarts) and L3 (closed loop) for Drake, Pinocchio, OpenSim, MyoSuite | Measured, 2026-10-06 (#11614); worst 1.1e-9 rad |
| Single full-horizon open loop | Growth about 40 /s, agreement horizon 0.5-0.6 s (documented, not a pass/fail bound) |
| Simscape | Pending on the MATLAB R2025b host (#11613) |

## Build and Reproduce

```bash
cd docs/research/same_input_parity && tectonic same_input_parity.tex
MUJOCO_GL=egl PYTHONPATH=.:src python3 -m scripts.same_input_pointwise_parity
```

The generated PDF is not committed.
