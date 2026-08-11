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
two-hand constraint rank, and proper 3-D frame invariance. A reduced full-body
common-state tier adds genuine nonplanar motion and compares MuJoCo inverse
dynamics with an independent Lagrange--Christoffel formulation built from the
same hashed model. It preserves the geometry sign intervention while keeping
passive and forward-contact claims explicitly open. A three-tier
hand-path attribution study now separates stitched pointwise ZTCF drift,
same-state control, and separately defined ZVCF reactions for force vectors,
impulse, power, work, every modeled joint, and four neutral time windows. A
bounded first-order residual-couple preview study records the late two-hand
preactivation hypothesis without promoting it to a physiological finding.
A forward constrained two-arm study now evolves the floating club and both arm
chains under four independent grip constraints. It records exact same-state
zero-command branches, separated force-generated and direct-wrist moments,
contact modes and power, a zero-moment-arm negative control, and timestep and
projection sensitivity. The result establishes finite passive-couple
persistence in the declared planar model only.
A coupled moving-base and flexible-club extension now evolves a finite-mass
translating base, both constrained arms, both grip reactions, and a two-segment
club in one forward solve. It records base and shaft energy, a same-state
zero-command branch, a coincident-grip geometric control, parameter
sensitivity, and timestep convergence without prescribing base motion or
shaft flex.
A coupled uncertainty, identifiability, and delayed-control phase now varies
12 inputs simultaneously in that same forward model. It publishes deterministic
Latin-hypercube/PRCC screening, rank and null-space audits, and separate
training/held-out comparisons of eight preselected programs across five
objectives. The result exposes non-identifiability and strategy tradeoffs; it
does not claim a population distribution, physiological actuator, universal
optimum, or coaching prescription.

The public-facing article is available on
[affinedrift.com](https://affinedrift.com/articles/proximal-distal-energy-transfer.html).
Ongoing validation and extension work is tracked in
[#8426](https://github.com/D-sorganization/UpstreamDrift/issues/8426) and the
[interaction-force mechanisms epic](https://github.com/D-sorganization/UpstreamDrift/issues/8443).
The hand-path attribution, two-hand redundancy, and preactivation validation
program is tracked in
[#8458](https://github.com/D-sorganization/UpstreamDrift/issues/8458).

## Layout

| Path                                                                                   | What it is                                                                        |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| [`proximal_distal_energy_transfer.qmd`](proximal_distal_energy_transfer.qmd)           | Master Quarto document (front matter + chapter includes)                          |
| [`chapters/`](chapters/)                                                               | Chapter source files (`_ch01`–`_ch09`, `_appendices`)                             |
| [`HAND_PATH_ATTRIBUTION_CONTRACT.md`](HAND_PATH_ATTRIBUTION_CONTRACT.md)               | Canonical source, terminology, and estimand contract for hand-path attribution    |
| [`EVIDENCE_SCHEMA_V2.md`](EVIDENCE_SCHEMA_V2.md)                                       | Falsifiable prediction and named spatial-interface evidence contract              |
| [`MODEL_COMPLETION_FALSIFICATION_MATRIX.md`](MODEL_COMPLETION_FALSIFICATION_MATRIX.md) | Claim, alternative-explanation, model-discrepancy, and falsifier register         |
| [`EXPERIMENTAL_FALSIFICATION_PROTOCOL.md`](EXPERIMENTAL_FALSIFICATION_PROTOCOL.md)     | Frozen human-data acquisition, split, analysis, and inference-boundary protocol   |
| [`REVIEWER_WORKBENCH.md`](REVIEWER_WORKBENCH.md)                                       | Claim-first figure, evidence, and download index by model tier                    |
| [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md)                                             | Artifact and recurring-field definitions with interpretation boundaries           |
| [`release_manifest.json`](release_manifest.json)                                       | Hash-pinned presets, claim status, artifacts, and open release gates              |
| [`references.bib`](references.bib)                                                     | Linked bibliography plus a clearly labeled project-originated presentation source |
| [`figures/`](figures/)                                                                 | Figures generated from the recorded analyses (PDF and SVG)                        |
| [`data/`](data/)                                                                       | Recorded experiment outputs with provenance (JSON + NPZ)                          |
| [`proximal_distal_energy_transfer.tex`](proximal_distal_energy_transfer.tex)           | LaTeX generated from the Quarto source (`keep-tex: true`)                         |
| [`sources/wscg_2024/`](sources/wscg_2024/)                                             | Hash-registered WSCG presentation sources and interpretation boundaries           |
| [`proximal_distal_energy_transfer.pdf`](proximal_distal_energy_transfer.pdf)           | Rendered scientific PDF                                                           |

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
python3 -m scripts.research.proximal_distal_energy.run_hand_path_attribution_study
python3 -m scripts.research.proximal_distal_energy.two_hand_preactivation_hypothesis
python3 -m scripts.research.proximal_distal_energy.run_forward_two_arm_study
python3 -m scripts.research.proximal_distal_energy.run_moving_base_flexible_study
python3 -m scripts.research.proximal_distal_energy.run_spatial_full_body_study
python3 -m scripts.research.proximal_distal_energy.run_uncertainty_control_study
python3 -m scripts.research.proximal_distal_energy.run_experimental_protocol_dry_run
python3 -m scripts.research.proximal_distal_energy.qualify_open_release validate
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
python3 -m scripts.research.proximal_distal_energy.make_forward_two_arm_figures
python3 -m scripts.research.proximal_distal_energy.make_moving_base_flexible_figures
python3 -m scripts.research.proximal_distal_energy.make_spatial_full_body_figures
python3 -m scripts.research.proximal_distal_energy.make_uncertainty_control_figures
# document
cd docs/research/proximal_distal_energy_transfer
quarto render proximal_distal_energy_transfer.qmd --to pdf
cd ../../..
python3 -m scripts.research.proximal_distal_energy.optimize_article_pdf
```

Requires Quarto + a LaTeX distribution (TeX Live with `lmodern`), and
Python 3.11+ with `numpy`, `matplotlib`, `pydantic`, `simpleeval`,
`pandas`, and `pymupdf`. The final command performs lossless PDF object/stream
compaction and fails if the page, URI-link, or outline contract changes.
Experiments are deterministic. The open-chain studies use fixed-step RK4; the
forward constrained two-hand study uses velocity Verlet with mass-metric
position and velocity projection. Parameters and numerical contracts are
recorded in `data/*.json`.

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
  prescribed mobile-hub inverse dynamics, planar closed-loop geometry, proper
  3-D frame transformations, and reduced full-body nonplanar common-state
  inverse dynamics in two independent formulations. The spatial tier prescribes
  hand loads; forward closed contact and passive load generation remain
  explicitly untested.
- The hand-path evidence uses two forward-simulated open-chain reference cases
  and one prescribed, constraint-consistent two-arm local sweep. Its normalized
  time quartiles are bookkeeping windows, not anatomical swing phases. ZVCF
  forces are same-configuration diagnostics and are never added to the exact
  total = drift + control closure.
- The residual-couple preview test holds archived BASE and pointwise ZTCF
  traces fixed. It tests a signal-delay hypothesis, not muscle activation,
  metabolic effort, clubhead-speed improvement, or a human timing prescription.
- The forward two-hand study uses a fixed shoulder base, rigid club, planar
  point constraints, and one declared command/parameter set. Its zero-command
  branch demonstrates mechanical persistence, not muscle passivity, human use,
  or an optimal speed strategy.
- The coupled moving-base/flexible-club study replaces prescribed hub motion
  with a finite-mass translating base and adds one lumped torsional club mode.
  Its planar point contacts, linear springs and dampers, and declared
  mechanism-study parameters are not a calibrated body, distributed shaft, or
  equipment comparison.
- The reduced full-body spatial study uses 20 generalized coordinates and
  spherical inertia elements shared by MuJoCo and an independent analytical
  implementation. Its same-state agreement is an implementation-transport
  result, not anatomical validation or a forward-contact simulation.
- The uncertainty/control study uses declared engineering envelopes and small
  deterministic model ensembles. PRCC is a screening statistic; the effort
  and face/path quantities are proxies; and held-out model performance is not
  participant validation.
- Generalization and human-data work is tracked in
  [#8426](https://github.com/D-sorganization/UpstreamDrift/issues/8426).

Edit the `.qmd`/chapter files, never the generated `.tex`.
