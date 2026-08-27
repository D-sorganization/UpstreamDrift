# Event Topology, Delay, and Perturbation Robustness

## Scope

This study asks whether a delivery-event conclusion survives a global search,
causal command delay, synthetic state/command/event-surface perturbations, and
coordinate-authority countermodels. It extends the bounded local result in
`BOUNDED_EVENT_REACHABILITY.md`; it does not replace that result or convert
topology preservation into target feasibility.

All cases use the analytical planar double pendulum, the declared geometric
guard $\theta_1+\theta_2=0$, exact RK4 replay, explicit crossing direction,
and a common global horizon. Every crossing is retained. Absent, unique,
multiple, grazing, initial-on-guard, and numerical-failure outcomes are typed
rather than selecting whichever event is nearest the nominal crossing.

## Registered Designs

### Phase A: Small Synthetic Perturbations

Phase A uses eleven physical command delays from 0 to 200 ms, a 0.60 s common
horizon, seed 9125, and dimensionless perturbation fractions 0, 0.001, 0.005,
and 0.01 relative to declared state, command, and guard scales. Each nonzero
cell contains 192 antithetic replicates, analyzed as 96 independent pairs.

### Phase B: Fixed Stress-to-Failure Extension

After Phase A found no topology loss through 1%, a separate public
preregistration fixed fractions 0.02, 0.05, 0.10, 0.20, and 0.50. All five
levels were executed. They are artificial model stresses, not percentages of
human variability.

### Phase C: Channel Masks and Numerical Controls

Phase C applies four generalized-coordinate masks to both the command and its
synthetic perturbation: both $(1,1)$, shoulder only $(1,0)$, wrist only
$(0,1)$, and zero authority $(0,0)$. State and guard draws remain matched.
The noisy matrix uses the 1% scenario, the same eleven delays, seed, horizon,
and 192 replicates. Deterministic controls use 1, 2, and 4 ms integration steps
at 0.60 s and 0.40, 0.60, and 0.80 s undelayed search horizons.

These masks remove authority in generalized torque coordinates. They are not
anatomical isolation experiments and cannot identify wrist, hand, arm, or
scapular action in a participant.

## Results

Phase A retains one positive transverse event in all 6,336 nonzero outcomes.
Each nonzero cell preserves 96 of 96 independent pairs, with a 95% Wilson
interval of [0.961524, 1.0]. Event times span 0.341962--0.541553 s. The common
0.60 s horizon is essential: an earlier apparent loss on a 0.40 s horizon was
horizon truncation, not a mechanics threshold.

Phase B first loses topology at the 2% stress and 200 ms delay: one outcome is
absent and 191 remain unique transverse. At the largest artificial stress
(50%) and 200 ms, 118 outcomes are absent, seven have multiple crossings, and
67 remain unique transverse. Only 2 of 96 pairs preserve the nominal topology,
with interval [0.00573197, 0.0728071]. This maps a synthetic failure region; it
does not estimate human tolerance.

In Phase C, both-channel and shoulder-only commands retain a unique positive
crossing in every nominal delay case and every 1% perturbation cell. The
wrist-only nominal crossing is unique through 40 ms and absent from 60 through
200 ms. Under perturbation, wrist-only crossing loss increases through 60 ms;
once the nominal case is absent, pairwise topology preservation rises again.
That rise means consistent absence, not successful delivery. Zero authority
is absent at every delay, and its retained command perturbation contains zero
nonzero entries.

![Channel Masks Expose Topology Loss and Horizon Truncation](figures/fig_event_topology_robustness.pdf)

All channel/delay topology identities agree across 1, 2, and 4 ms steps. The
largest cross-step event-time residual is $4.93\times10^{-9}$ s, the largest
event-state residual is $3.80\times10^{-6}$, the largest clubhead-speed
residual is $1.53\times10^{-6}$ m/s, and the largest transversality residual
is $2.17\times10^{-6}$ s$^{-1}$. The 0.60 and 0.80 s horizons agree for every
channel. The wrist-only crossing is absent at 0.40 s but unique at 0.579326 s
on both expanded horizons, explicitly identifying original-horizon truncation.

## Outcome Separation

| Quantity                                              | Phase C Status           | Interpretation Boundary                        |
| ----------------------------------------------------- | ------------------------ | ---------------------------------------------- |
| Global topology and crossing direction                | Retained per replay      | Event identity, not task success               |
| Event time, state, transversality, and clubhead speed | Retained separately      | Speed cannot rescue failed topology            |
| Bounded feasibility and event-tangent target error    | Source-linked from #9124 | Not inferred from topology preservation        |
| Amplitude/slew constraint status and objective        | Source-linked from #9124 | No channel or effort ranking                   |
| Work and power                                        | Unavailable              | No independently registered Phase C quadrature |
| Human variability, fatigue, or anatomy                | Unavailable              | Requires governed participant evidence         |
| Coaching recommendation                               | Unavailable              | No universal strategy is supported             |

The parent bounded study contains 32 feasible and six correctly infeasible
registered outcomes; all six infeasible outcomes are displaced zero-authority
targets. Those types remain independent of
the Phase C topology types. A faster event cannot change an absent, multiple,
reversed, grazing, numerical-failure, target-infeasible, or constraint-failed
classification.

## Falsification Status

- Global enumeration retains zero, unique, multiple, reversed, grazing,
  initial-on-guard, and numerical-failure controls.
- Antithetic pairs, fixed seed, common random numbers, raw counts, and Wilson
  intervals are retained. Probability-like fields fail closed below the
  registered pair-count or precision gate.
- The zero-authority mask cannot gain command torque through perturbation.
- Step refinement preserves topology identity over the registered matrix.
- Expanded horizons agree; the original wrist-only discrepancy is typed as
  truncation.
- Phase B completes its fixed ladder even after topology loss appears.

## Limitations

The model is synthetic, planar, open loop, and driven by declared command
programs. Perturbation scales are not calibrated motor noise. Channel masks do
not isolate biological structures. The delivery guard is geometric rather
than measured impact. Work/power, fatigue, injury, skill, participant
heterogeneity, bilateral grip-wrench validation, and coaching outcomes remain
unavailable. A robustness or preservation value is conditional on this model,
guard, command, horizon, delay policy, and perturbation design.

## Reproducible Validation

```powershell
python -m scripts.research.proximal_distal_energy.run_event_topology_robustness validate
python -m scripts.research.proximal_distal_energy.run_event_topology_stress_extension validate
python -m scripts.research.proximal_distal_energy.run_event_topology_channel_matrix validate
python -m pytest -n 0 -q tests/research/test_event_topology_robustness.py tests/research/test_event_topology_robustness_evidence.py tests/research/test_event_topology_stress_extension_evidence.py tests/research/test_event_topology_channel_controls.py tests/research/test_event_topology_channel_matrix_evidence.py
python scripts/ci/check_architecture_budget.py
python scripts/ci/check_file_size_budget.py
```

The portable summaries are
`data/event_topology_robustness.json`,
`data/event_topology_stress_extension.json`, and
`data/event_topology_channel_matrix.json`. Their paired NPZ files retain the
full-precision arrays.
