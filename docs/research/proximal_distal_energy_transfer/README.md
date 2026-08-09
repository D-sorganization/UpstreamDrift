# Proximal-to-Distal Energy Transfer in the Golf Swing

Open research materials for a study of proximal-to-distal (P→D) energy
transfer in the golf swing. The study combines a literature synthesis
with reproducible two-link simulations, counterfactual acceleration
decomposition, segment-level energy accounting, actuator-bound checks,
impact-definition sensitivity, and model-parameter sensitivity.

The public-facing article is available on
[affinedrift.com](https://affinedrift.com/articles/proximal-distal-energy-transfer.html).
Ongoing validation and extension work is tracked in
[#8426](https://github.com/D-sorganization/UpstreamDrift/issues/8426).

## Layout

| Path                                                                         | What it is                                                                           |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| [`proximal_distal_energy_transfer.qmd`](proximal_distal_energy_transfer.qmd) | Master Quarto document (front matter + chapter includes)                             |
| [`chapters/`](chapters/)                                                     | Chapter source files (`_ch01`–`_ch09`, `_appendices`)                                |
| [`references.bib`](references.bib)                                           | Verified bibliography (61 entries; every entry checked against a publication record) |
| [`figures/`](figures/)                                                       | Figures generated from the recorded analyses (PDF and SVG)                           |
| [`data/`](data/)                                                             | Recorded experiment outputs with provenance (JSON + NPZ)                             |
| [`proximal_distal_energy_transfer.tex`](proximal_distal_energy_transfer.tex) | LaTeX generated from the Quarto source (`keep-tex: true`)                            |
| [`proximal_distal_energy_transfer.pdf`](proximal_distal_energy_transfer.pdf) | Rendered PDF (40 pp.)                                                                |

## Reproducing everything

```bash
# primary analyses (E1 sweep, E2 counterfactuals, E4 interface powers)
python3 -m scripts.research.proximal_distal_energy.run_experiments
# robustness analyses
python3 -m scripts.research.proximal_distal_energy.e1b_bounded_torque
python3 -m scripts.research.proximal_distal_energy.e1c_impact_sensitivity
python3 -m scripts.research.proximal_distal_energy.e1d_parameter_sensitivity
# figures
python3 -m scripts.research.proximal_distal_energy.make_figures
# document
cd docs/research/proximal_distal_energy_transfer
quarto render proximal_distal_energy_transfer.qmd --to pdf
```

Requires Quarto + a LaTeX distribution (TeX Live with `lmodern`), and
Python 3.11+ with `numpy`, `matplotlib`, `pydantic`, `simpleeval`,
`pandas`. Experiments are deterministic (fixed-step RK4); provenance
(git SHA, parameters) is stamped into `data/*.json`.

## Evidence boundaries

- Every reference is verified; two candidate sources that could not be
  confirmed were rejected rather than cited.
- Every number in the Results chapter is produced by the committed
  scripts in `scripts/research/proximal_distal_energy/` from the
  recorded data in `data/`.
- Model-derived findings are labeled separately from empirical findings.
- The simulation demonstrates a mechanism within a planar 2-DOF,
  fixed-hub, rigid-shaft model; it does not establish a universal
  coaching prescription or population-level effect.
- Generalization and human-data work is tracked in
  [#8426](https://github.com/D-sorganization/UpstreamDrift/issues/8426).

Edit the `.qmd`/chapter files, never the generated `.tex`.
