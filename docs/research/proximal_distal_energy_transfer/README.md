# Proximal-to-Distal Energy Transfer in the Golf Swing

Open research materials for a study of proximal-to-distal (P→D) energy
transfer in the golf swing. The study combines a literature synthesis
with reproducible two-link simulations, counterfactual acceleration
decomposition, exact interaction-force and force-power accounting, a
matched-state torque-killswitch, actuator-bound checks, impact-definition
sensitivity, model-parameter sensitivity, and a 96-case cut-time/horizon/step
counterfactual ensemble with gravity and damping ablations, and a direct
two-hand wrench audit of the archived WSCG BASE/ZTCF/DELTA model tables. A
matched rigid/flexible three-coordinate study separately accounts for control,
momentum, gravity, joint damping, shaft elasticity, and shaft damping, with
closed energy bookkeeping and a 120-case robustness grid. A common-observable
model ladder then tests wrench transport, prescribed mobile-hub reactions,
two-hand constraint rank, and proper 3-D frame invariance without representing
the still-open full-body cross-engine comparison as complete.

The public-facing article is available on
[affinedrift.com](https://affinedrift.com/articles/proximal-distal-energy-transfer.html).
Ongoing validation and extension work is tracked in
[#8426](https://github.com/D-sorganization/UpstreamDrift/issues/8426) and the
[interaction-force mechanisms epic](https://github.com/D-sorganization/UpstreamDrift/issues/8443).

## Layout

| Path                                                                         | What it is                                                                        |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| [`proximal_distal_energy_transfer.qmd`](proximal_distal_energy_transfer.qmd) | Master Quarto document (front matter + chapter includes)                          |
| [`chapters/`](chapters/)                                                     | Chapter source files (`_ch01`–`_ch09`, `_appendices`)                             |
| [`references.bib`](references.bib)                                           | Linked bibliography plus a clearly labeled project-originated presentation source |
| [`figures/`](figures/)                                                       | Figures generated from the recorded analyses (PDF and SVG)                        |
| [`data/`](data/)                                                             | Recorded experiment outputs with provenance (JSON + NPZ)                          |
| [`proximal_distal_energy_transfer.tex`](proximal_distal_energy_transfer.tex) | LaTeX generated from the Quarto source (`keep-tex: true`)                         |
| [`sources/wscg_2024/`](sources/wscg_2024/)                                   | Hash-registered WSCG presentation sources and interpretation boundaries           |
| [`proximal_distal_energy_transfer.pdf`](proximal_distal_energy_transfer.pdf) | Rendered scientific PDF                                                           |

## Reproducing Everything

```bash
# primary analyses (E1 sweep, E2 counterfactuals, E4 interface powers)
python3 -m scripts.research.proximal_distal_energy.run_experiments
# registered source extraction and exact interaction-force study
python3 -m scripts.research.proximal_distal_energy.extract_wscg_charts
python3 -m scripts.research.proximal_distal_energy.run_interaction_force_study
python3 -m scripts.research.proximal_distal_energy.run_counterfactual_ensemble
python3 -m scripts.research.proximal_distal_energy.run_two_hand_wscg_analysis
python3 -m scripts.research.proximal_distal_energy.run_shaft_contribution_study
python3 -m scripts.research.proximal_distal_energy.run_mechanism_ladder_study
# robustness analyses
python3 -m scripts.research.proximal_distal_energy.e1b_bounded_torque
python3 -m scripts.research.proximal_distal_energy.e1c_impact_sensitivity
python3 -m scripts.research.proximal_distal_energy.e1d_parameter_sensitivity
# figures
python3 -m scripts.research.proximal_distal_energy.make_figures
python3 -m scripts.research.proximal_distal_energy.make_interaction_force_figures
python3 -m scripts.research.proximal_distal_energy.make_counterfactual_figures
python3 -m scripts.research.proximal_distal_energy.make_two_hand_wscg_figures
python3 -m scripts.research.proximal_distal_energy.make_shaft_contribution_figures
python3 -m scripts.research.proximal_distal_energy.make_mechanism_ladder_figures
# document
cd docs/research/proximal_distal_energy_transfer
quarto render proximal_distal_energy_transfer.qmd --to pdf
```

Requires Quarto + a LaTeX distribution (TeX Live with `lmodern`), and
Python 3.11+ with `numpy`, `matplotlib`, `pydantic`, `simpleeval`,
`pandas`. Experiments are deterministic (fixed-step RK4); provenance
(git SHA, parameters) is stamped into `data/*.json`.

## Evidence Boundaries

- Peer-reviewed claims use linked publication records. The author's WSCG
  presentation is separately registered as project-originated hypothesis
  evidence and is not represented as independent validation.
- Every number in the Results chapter is produced by the committed
  scripts in `scripts/research/proximal_distal_energy/` from the
  recorded data in `data/`.
- Model-derived findings are labeled separately from empirical findings.
- The archived two-hand tables can optionally be re-exported with MATLAB by
  running `export_two_hand_wscg_tables`; committed CSV caches allow the audit
  and figures to run without MATLAB or Simscape.
- The simulation demonstrates mechanisms within a planar 2-DOF, fixed-hub
  model and a separate three-coordinate point-mass shaft-flex surrogate;
  neither establishes a universal coaching prescription, equipment effect, or
  population-level result.
- The higher-order ladder executes a three-coordinate interface audit,
  prescribed mobile-hub inverse dynamics, planar closed-loop constraint
  geometry, and 3-D frame transformations. Full-body cross-engine dynamics are
  explicitly recorded as unexecuted.
- Generalization and human-data work is tracked in
  [#8426](https://github.com/D-sorganization/UpstreamDrift/issues/8426).

Edit the `.qmd`/chapter files, never the generated `.tex`.
