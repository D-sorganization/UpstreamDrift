# Proximal-to-Distal Energy Transfer in the Golf Swing

The latest adversarial extension separates transmission-pathway identity,
nominal speed, lower-tail performance, dispersion, contact loading, task-null
variability, and model-conditional perturbation rejection. Start with
[`ADVERSARIAL_TRANSMISSION_REVIEW.md`](ADVERSARIAL_TRANSMISSION_REVIEW.md) and
the chapter “Transmission Pathways, Robust Speed, and Task Stability.” These
results do not establish human self-stabilization or a universal strategy.

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
passive and forward-contact claims open at that tier. A subsequent reduced
spatial experiment independently advances native MuJoCo and Pinocchio forward
dynamics with finite-mass hand carriages, paired compliant interfaces, an
exact same-state grounded-driver killswitch, long-axis rotation, swing-plane
evolution, and ground-pathway bookkeeping. It supports reduced forward-contact
transport while keeping anatomical, muscle, equipment, human, and coaching
claims explicitly open. A separate
subject-scaled geometry audit now tests six deterministic de Leva design
profiles and three grip spans. It finds 0.171--0.616 m anatomical
hand-to-grip error despite a full-rank local constraint Jacobian, thereby
rejecting the prescribed states. A bounded follow-up closes all 234 registered
profile/span/phase configurations while preserving the club pose, achieved
rank, broad engineering-limit margins, and coarse collision clearance. This
clears a reduced-tree necessary condition and advances the open gate to
subject-specific anatomy and calibrated forward contact; it does not establish
forces, passivity, timing benefit, slack benefit, or human strategy. A three-tier
hand-path attribution study separates stitched pointwise ZTCF drift,
same-state control, and separately defined ZVCF reactions for force vectors,
impulse, power, work, every modeled joint, and four neutral time windows. A
bounded first-order residual-couple preview study records the late two-hand
preactivation hypothesis without promoting it to a physiological finding.
A constrained-contact extension now separates configuration, velocity,
control, and other-external contributions to a uniquely determined reaction;
the fixed-support double-pendulum benchmark adds pointwise GRF-analogue ZTCF
and ZVCF traces, closure tests, prediction metrics, vector figures, and an
explicit human force-plate falsification protocol. It does not infer bilateral
foot forces from a resultant wrench.
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
A coupled forward modal-shaft extension replaces the single lumped flex
coordinate with multiple bending modes while retaining the moving base, two
hand contacts, and same-state controls. It is a synthetic mechanism test, not
an equipment-calibrated shaft result.
A distributed-shaft structural study now reuses the shared Euler--Bernoulli
finite-element model, adds declared head inertia, identifies a synthetic
two-parameter modal case, and compares one-mode and six-mode responses under
slow and short-duration loads. It exposes higher-mode discrepancy while
remaining explicitly separate from equipment calibration and from the coupled
two-hand solve.
A matched-task allocation study then compares the proximal joint-torque and
direct wrist-moment extremes that can produce the same declared club moment.
A separate phenomenological transmission channel tests persistent-direction
and wrist-to-arm role-reversal programs with and without preload, across a
registered dead-zone and time-constant grid. These are actuator-allocation and
transmission hypotheses; they do not identify scapular muscles, tissue slack,
or a preferred human technique.
A coupled uncertainty, identifiability, and delayed-control phase now varies
12 inputs simultaneously in that same forward model. It publishes deterministic
Latin-hypercube/PRCC screening, rank and null-space audits, and separate
training/held-out comparisons of eight preselected programs across five
objectives. The result exposes non-identifiability and strategy tradeoffs; it
does not claim a population distribution, physiological actuator, universal
optimum, or coaching prescription.
An advanced bridge now makes force-first wrench, linear-first twist,
reference-point transport, Jacobian virtual work, and power conventions
executable. It adds a reduced Hill-type agonist--antagonist redundancy surface
and continuous activation/series-force comparison of persistent-direction and
complete-role-reversal preparation histories. A question-to-engine ladder then
connects MuJoCo, Pinocchio, Drake, OpenSim, and MyoSuite to common observables
without promoting optional backend capability to human validation.

The public-facing technical article is available on
[affinedrift.com](https://affinedrift.com/articles/proximal-distal-energy-transfer.html).
For a visual, book-like route through the same mechanics, use the companion
[_How a Golf Swing Carries Energy_](https://affinedrift.com/articles/proximal-distal-a-journey-through-the-swing.html),
including its downloadable
[PDF edition](https://affinedrift.com/articles/proximal-distal-a-journey-through-the-swing.pdf).
The companion labels model results, human evidence, and hypotheses separately
and returns readers to this directory for the complete evidence and limitations.
Readers can also run the PyQt6 or React/Tauri
[interactive dynamics workbench](COMPANION_WORKBENCH.md). Its guided experiments,
hover help, glossary, hypotheses, falsifiers, and limitations share one
canonical catalog maintained by the open-source Tools provider.
Ongoing validation and extension work is tracked in
[#8426](https://github.com/D-sorganization/UpstreamDrift/issues/8426) and the
[interaction-force mechanisms epic](https://github.com/D-sorganization/UpstreamDrift/issues/8443).
The paper-wide claim audit, biomechanics and nonlinear-control expansion, and
comprehensive open modeling program are tracked in
[#8557](https://github.com/D-sorganization/UpstreamDrift/issues/8557), with the
durable execution contract in
[`COMPREHENSIVE_RESEARCH_PROGRAM.md`](COMPREHENSIVE_RESEARCH_PROGRAM.md).
The MT-E07 measurement program now includes both the exact bilateral-wrench
rank audit and a trajectory-level synthetic point-force qualification under
noise, normalized cross-talk, calibration residual, and contact-center
migration. It demonstrates that net-wrench closure does not establish force
allocation and retains physical-device and human validation as open gates.
MT-E08 now executes the prescribed-state rejection and reduced-tree
closed-contact screen, then retains independently integrated calibrated
articulated contact and participant-held-out falsification as open gates.
The first external-evidence audit slice narrows the introduction and empirical
evidence synthesis, records reciprocal candidate dispositions, and separates
sample-specific associations from causal mechanisms. In particular, it corrects
the composition of the 2026 foot-ground regression block and preserves the
source's explicit cross-sectional causal limitation.
The second slice corrects the kinetic-energy literature: distal amplification
of peak kinetic-energy magnitude in Anderson (2006) and Kenny (2008) did not
coincide with a simple proximal-to-distal ordering of peak times. It also uses
content-stable candidate identifiers so unrelated line insertions no longer
invalidate completed reviews, and it records explicit falsifiers and scope
limits for torso-pelvis, X-factor, timing, and casting claims.
The inventory parser also treats labeled Quarto display-math closers correctly
and canonicalizes line endings in its source digest. This repair exposed 308
previously omitted narrative paragraphs, raising the fail-closed audit queue
from 567 to 875 candidates without invalidating the 18 completed reviews.
The third evidence slice corrects the active-wrist-torque comparison reported
for Sprigings and Neal (approximately 9%, not 4%), separates it from the 1.6%
forced-delay comparison in Sprigings and MacKenzie, and records White's
different driven-model result rather than claiming uniform agreement. A
source-level contrast matrix now distinguishes release delay, active wrist
torque, passive transfer, prescribed inward pull, hub-path optimization, and
negative-to-positive wrist-torque programs.
The fourth evidence slice separates four non-equivalent optimization studies.
It records each model's state space, objective, speed or carry result, actuator
timing, omitted degrees of freedom, and validation boundary. The paper no
longer labels their outputs collectively as expert-like or treats late distal
activation as a repeated estimate of one universal strategy. The expanded
paper had 882 candidates at that merge; 32 were reviewed and 34 atomic claims
were registered.
The fifth evidence slice corrects the foundational mechanics. It replaces the
universal assertion that every torque accelerates every joint with the exact
nonzero inverse-inertia coupling condition, bounds the affine drift--control
split to its declared unconstrained model class, and source-matches the Putnam,
Feltner--Dapena, Herring--Chapman, and Marshall--Elliott findings. A new
contrast matrix preserves Putnam's negative findings, distinguishes the 1986
pitch-load analysis from the 1989 angular-acceleration decomposition, and
labels the golf extension of racquet-task long-axis rotation as a hypothesis.
The expanded paper now has 889 candidates; 44 are reviewed, 43 atomic claims
are registered, and 845 remain unadjudicated.
The release-level external-source qualification consolidates 78 exact links
into 56 underlying works and records a bounded assessment for all 54 atomic
claims that use external support. It removes three unresolvable DOI-like
identifiers, replaces a broken DOI destination with its stable PubMed record,
and removes two redundant author-hosted mirrors with TLS failures. The
committed availability snapshot contains 36 directly resolving links and 42
publisher or index pages that restrict automated access; it contains no broken,
transient, omitted, or unchecked link. Multiple mirrors never count as
independent replication, and all prospective human gates remain open.
The hand-path attribution, two-hand redundancy, and preactivation validation
program is tracked in
[#8458](https://github.com/D-sorganization/UpstreamDrift/issues/8458).
The arm--wrist allocation, role-reversal, and preload program is tracked in
[#8497](https://github.com/D-sorganization/UpstreamDrift/issues/8497).
The advanced frame, biological, visual, and cross-engine expansion is tracked
in [#8505](https://github.com/D-sorganization/UpstreamDrift/issues/8505).
Ground-reaction drift attribution and human validation requirements are tracked
in [#8493](https://github.com/D-sorganization/UpstreamDrift/issues/8493).
The independent review adjudication and its numerical remediation are tracked
in [#8499](https://github.com/D-sorganization/UpstreamDrift/issues/8499), with
the finding-by-finding record in
[`ADVERSARIAL_REVIEW_ADJUDICATION.md`](ADVERSARIAL_REVIEW_ADJUDICATION.md).

## Layout

| Path                                                                                                 | What it is                                                                        |
| ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| [`proximal_distal_energy_transfer.qmd`](proximal_distal_energy_transfer.qmd)                         | Master Quarto document (front matter + chapter includes)                          |
| [`chapters/`](chapters/)                                                                             | Chapter source files (`_ch01`–`_ch09`, `_appendices`)                             |
| [`HAND_PATH_ATTRIBUTION_CONTRACT.md`](HAND_PATH_ATTRIBUTION_CONTRACT.md)                             | Canonical source, terminology, and estimand contract for hand-path attribution    |
| [`TERMINOLOGY_AND_CONVENTIONS.md`](TERMINOLOGY_AND_CONVENTIONS.md)                                   | Normative scientific vocabulary, frame, wrench, power, and evidence-status rules  |
| [`ADVANCED_EXPANSION_REVIEW.md`](ADVANCED_EXPANSION_REVIEW.md)                                       | Completed review, implemented expansion, and falsifiable next-model roadmap       |
| [`EVIDENCE_SCHEMA_V2.md`](EVIDENCE_SCHEMA_V2.md)                                                     | Falsifiable prediction and named spatial-interface evidence contract              |
| [`CLAIM_AUDIT_SCHEMA.md`](CLAIM_AUDIT_SCHEMA.md)                                                     | Atomic claim, candidate-inventory, source, alternative, and adjudication contract |
| [`data/claim_evidence_manifest.json`](data/claim_evidence_manifest.json)                             | Claim-complete local hashes and external-support URL inventory                    |
| [`data/external_source_review.json`](data/external_source_review.json)                               | Work-deduplicated source, correction, claim-fit, and link-availability review     |
| [`COMPREHENSIVE_RESEARCH_PROGRAM.md`](COMPREHENSIVE_RESEARCH_PROGRAM.md)                             | Biomechanics, nonlinear-control, model-ladder, data, and validation roadmap       |
| [`MOMENTUM_TRANSFER_QUESTION_PROGRAM.md`](MOMENTUM_TRANSFER_QUESTION_PROGRAM.md)                     | Drift, geometry, timing, robustness, proximal-velocity, and typed-slack questions |
| [`data/momentum_transfer_experiment_registry.json`](data/momentum_transfer_experiment_registry.json) | Frozen interventions, controls, outcomes, uncertainty, falsifiers, and data needs |
| [`MODEL_COMPLETION_FALSIFICATION_MATRIX.md`](MODEL_COMPLETION_FALSIFICATION_MATRIX.md)               | Claim, alternative-explanation, model-discrepancy, and falsifier register         |
| [`EXPERIMENTAL_FALSIFICATION_PROTOCOL.md`](EXPERIMENTAL_FALSIFICATION_PROTOCOL.md)                   | Frozen human-data acquisition, split, analysis, and inference-boundary protocol   |
| [`REVIEWER_WORKBENCH.md`](REVIEWER_WORKBENCH.md)                                                     | Claim-first figure, evidence, and download index by model tier                    |
| [`COMPANION_WORKBENCH.md`](COMPANION_WORKBENCH.md)                                                   | Interactive PyQt6 and React/Tauri model guide, experiments, and evidence boundary |
| [`ADVERSARIAL_REVIEW_ADJUDICATION.md`](ADVERSARIAL_REVIEW_ADJUDICATION.md)                           | Verified disposition and remediation record for the independent technical review  |
| [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md)                                                           | Artifact and recurring-field definitions with interpretation boundaries           |
| [`release_manifest.json`](release_manifest.json)                                                     | Hash-pinned presets, claim status, artifacts, and open release gates              |
| [`references.bib`](references.bib)                                                                   | Linked bibliography plus a clearly labeled project-originated presentation source |
| [`figures/`](figures/)                                                                               | Figures generated from the recorded analyses (PDF and SVG)                        |
| [`data/`](data/)                                                                                     | Recorded experiment outputs with provenance (JSON + NPZ)                          |
| [`proximal_distal_energy_transfer.tex`](proximal_distal_energy_transfer.tex)                         | LaTeX generated from the Quarto source (`keep-tex: true`)                         |
| [`sources/wscg_2024/`](sources/wscg_2024/)                                                           | Hash-registered WSCG presentation sources and interpretation boundaries           |
| [`proximal_distal_energy_transfer.pdf`](proximal_distal_energy_transfer.pdf)                         | Rendered scientific PDF                                                           |

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
python3 -m scripts.research.proximal_distal_energy.run_grf_drift_study
python3 -m scripts.research.proximal_distal_energy.two_hand_preactivation_hypothesis
python3 -m scripts.research.proximal_distal_energy.run_forward_two_arm_study
python3 -m scripts.research.proximal_distal_energy.run_moving_base_flexible_study
python3 -m scripts.research.proximal_distal_energy.run_moving_base_modal_shaft_study
python3 -m scripts.research.proximal_distal_energy.run_shaft_beam_reference
python3 -m scripts.research.proximal_distal_energy.run_torque_allocation_preload_study
python3 -m scripts.research.proximal_distal_energy.run_spatial_full_body_study
python3 -m scripts.research.proximal_distal_energy.run_spatial_forward_contact_study
python3 -m scripts.research.proximal_distal_energy.run_uncertainty_control_study
python3 -m scripts.research.proximal_distal_energy.run_timing_viability_study
python3 -m scripts.research.proximal_distal_energy.run_typed_slack_dynamic_study
python3 -m scripts.research.proximal_distal_energy.run_experimental_protocol_dry_run
python3 -m scripts.research.proximal_distal_energy.run_advanced_biological_bridge
python3 -m scripts.research.proximal_distal_energy.claim_audit inventory
python3 -m scripts.research.proximal_distal_energy.claim_audit validate
python3 -m scripts.research.proximal_distal_energy.claim_evidence_integrity validate
python3 -m scripts.research.proximal_distal_energy.external_source_review validate
python3 -m scripts.research.proximal_distal_energy.momentum_question_readiness validate
python3 -m scripts.research.proximal_distal_energy.qualify_open_release validate
# robustness analyses
python3 -m scripts.research.proximal_distal_energy.e1b_bounded_torque
python3 -m scripts.research.proximal_distal_energy.e1c_impact_sensitivity
python3 -m scripts.research.proximal_distal_energy.e1d_parameter_sensitivity
python3 -m scripts.research.proximal_distal_energy.e1e_smooth_command_sensitivity
# figures
python3 -m scripts.research.proximal_distal_energy.make_figures
python3 -m scripts.research.proximal_distal_energy.make_interaction_force_figures
python3 -m scripts.research.proximal_distal_energy.make_counterfactual_figures
python3 -m scripts.research.proximal_distal_energy.make_advanced_biological_bridge_figures
python3 -m scripts.research.proximal_distal_energy.make_two_hand_wscg_figures
python3 -m scripts.research.proximal_distal_energy.make_shaft_contribution_figures
python3 -m scripts.research.proximal_distal_energy.make_mechanism_ladder_figures
python3 -m scripts.research.proximal_distal_energy.make_forward_two_arm_figures
python3 -m scripts.research.proximal_distal_energy.make_moving_base_flexible_figures
python3 -m scripts.research.proximal_distal_energy.make_moving_base_modal_shaft_figures
python3 -m scripts.research.proximal_distal_energy.make_shaft_beam_reference_figures
python3 -m scripts.research.proximal_distal_energy.make_torque_allocation_preload_figures
python3 -m scripts.research.proximal_distal_energy.make_spatial_full_body_figures
python3 -m scripts.research.proximal_distal_energy.make_spatial_forward_contact_figures
python3 -m scripts.research.proximal_distal_energy.make_uncertainty_control_figures
python3 -m scripts.research.proximal_distal_energy.run_bilateral_wrench_identifiability_study
python3 -m scripts.research.proximal_distal_energy.run_bilateral_wrench_sensor_qualification
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
recorded in `data/*.json`. Re-executing the two-engine spatial forward study
also requires the declared `mujoco` and `pinocchio` extras. The native
Pinocchio `pin` wheel used for this archive executes in Linux/WSL; the adapters
reject unrelated packages that expose the same import name without the native
model and ABA APIs.

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
  hand loads and therefore cannot establish passive load origin by itself.
- The reduced spatial forward-contact tier independently executes native
  MuJoCo and Pinocchio trajectories with two translational hand carriages and a
  rigid club. Its compliant-contact killswitch supports mechanism persistence
  only for that declared model; it is not articulated anatomy, measured grip
  tissue, a distributed shaft, muscle coordination, or human evidence.
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
- The distributed-shaft comparison identifies only a declared synthetic modal
  truth. Its finite-element convergence and work--energy closure do not
  constitute equipment calibration, measured shaft validation, or proof that
  the result survives coupling into the constrained two-hand solve.
- The coupled modal-shaft study closes that synthetic coupling gap only for the
  declared planar mechanism case. It does not provide measured equipment
  calibration, torsion, nonlinear shaft behavior, or a subject-specific grip.
- The allocation study solves an exact same-state club-moment task over declared
  generalized actuator subspaces. Calling the proximal subspace an arm or
  scapular strategy is shorthand, not muscle identification. Its transmission
  channel gives a falsifiable operational meaning to lost continuity; it is not
  evidence that anatomical tissue was literally slack.
- The ground-reaction extension demonstrates constrained-reaction algebra and a
  fixed-support planar benchmark. Human drift fractions, bilateral foot-force
  allocation, COP, and free moment remain unvalidated without synchronized
  force-plate, whole-body, and club data.
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
