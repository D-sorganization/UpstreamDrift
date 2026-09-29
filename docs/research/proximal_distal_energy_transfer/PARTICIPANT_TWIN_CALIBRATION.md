# Participant-Calibrated Twins and Population Ensembles

## Purpose and Evidential Boundary

This workstream provides the machinery for participant-calibrated digital twins
and hierarchical population ensembles: an identity-safe cohort schema and
loader, a synthetic benchmark, a closed-form hierarchical inference workflow
with an explicit model-discrepancy term, an out-of-sample evaluation ledger, and
a fail-closed promotion gate set.

It does not provide a human result. The committed cohort is synthetic, the
committed observations are generated, and every governance gate that would admit
governed participant data is recorded as not applicable. Governed bilateral
grip-wrench and motion authority remains open under
[UpstreamDrift #8450](https://github.com/D-sorganization/UpstreamDrift/issues/8450)
and [#8556](https://github.com/D-sorganization/UpstreamDrift/issues/8556).
Nothing here calibrates anatomy, equipment, injury risk, or technique, and no
personalized recommendation is emitted.

## Identity-Safe Provenance

`scripts/research/proximal_distal_energy/participant_twin_provenance.py` freezes
the cohort contract in
[`data/participant_twin_cohort.json`](data/participant_twin_cohort.json). The
validator rejects a record rather than degrading it:

- every participant, session, club, and trajectory is an opaque namespaced
  pseudonym that may not encode a record index;
- an identity-bearing key, an e-mail address, a source filename, a filesystem
  path, or a calendar date anywhere in the record is a hard failure;
- participants, sessions, clubs, and trajectories are stored sorted, so record
  order carries no information beyond the pseudonyms themselves;
- a club pseudonym used by only one participant is rejected, because equipment
  that singles out a participant is an identifier;
- stratification is restricted to the coarse registered levels for
  anthropometry, skill, sex, age, handedness, injury history, impairment, club
  class, and task, each of which may be `withheld`;
- the governed-human data class is refused unless the ethics reference, consent
  and reuse basis, private data authority, calibration records, time
  synchronization records, and analysis release authorization are all satisfied.

The public facade emits counts only. It carries no pseudonym and suppresses any
stratum cell below three participants.

## Frozen Split and the Held-Out Outcome Barrier

The participant holdout and the per-trajectory calibration or evaluation role
are both ranked by a salted digest, never by list position or acquisition order,
and the committed split must reproduce from the salt during validation. The
record declares that it was frozen before outcome access and carries no outcome
field.

`HoldoutBarrier` is the only outcome accessor used by a calibration path. It
raises on any evaluation or unregistered trajectory and records an access
ledger, so "calibration never reads held-out outcomes" is an executed property
rather than an assurance.

## Hierarchical Inference Workflow

Observations are normalized fractional deviations from declared nominal feature
scales across six likelihood channels: kinematic, bilateral wrench, shaft,
impact, launch, and the optional activation channel. The model is

$$
y_{p,t}=X\,\theta_p+D\,b+\varepsilon,\qquad
\theta_p\sim\mathcal N(\mu,\Sigma_{\text{pop}}),\qquad
\mu\sim\mathcal N(0,\Sigma_0),
$$

with $b$ the explicit additive per-channel model-discrepancy term and
$\varepsilon$ the declared observation noise. The joint posterior over
$(\mu,\{\theta_p\},b)$ is conjugate and is solved in closed form, so every
reported number is reproducible without a sampler.

A held-out participant is calibrated on that participant's own
calibration-role trajectories only, conditioned on the population posterior
fitted without them.

## Identifiability Before Fitting

`screen_identifiability` runs on the design, the noise contract, and the prior,
never on data, and `fit_population` refuses a screen that does not match the
design it is asked to fit. The benchmark deliberately contains an exact alias
between `grip_stiffness_scale` and `grip_compliance_alias`, and the screen
reports the aliased direction, a structural rank below the parameter count, and
a practical-contraction verdict per parameter. Parameters that fail either test
are retained in the record and excluded from reported recovery; the prior keeps
the posterior proper but confers no evidence. Removing the optional activation
channel makes `activation_gain` structurally non-identifiable, which the screen
detects.

## Evaluation and Discrepancy Ablation

[`data/participant_twin_calibration.json`](data/participant_twin_calibration.json)
reports posterior contraction and out-of-sample prediction separately.
Trajectory-held-out and participant-held-out prediction are each compared with a
population baseline and an uncalibrated nominal baseline. The recorded ablation
shows that omitting the declared discrepancy term leaves the fit superficially
acceptable while the systematic channel error migrates into the twin parameters
and inflates their error against the generating values.

The transport audit reports, for every stratum level, the training count, the
evaluated count, the held-out error, and a transport status. Levels present only
among held-out participants are marked
`nontransportable_unrepresented_in_training` and retained; they are never
averaged into a transportable statement.

## Promotion Gates

The evaluation ledger records the following gates and fails closed on the first.

| Gate                                              | Meaning                                                     |
| ------------------------------------------------- | ----------------------------------------------------------- |
| `private_governed_authority_contract`             | governed participant authority exists and is satisfied      |
| `public_facade_contract`                          | the public record carries counts only                       |
| `calibration_never_reads_held_out_outcomes`       | the barrier ledger records zero held-out reads              |
| `identifiability_screened_before_fitting`         | the screen ran on the design and its verdict was retained   |
| `prior_predictive_check`                          | the prior can generate the calibration data                 |
| `posterior_predictive_check`                      | the fit reproduces the data it was calibrated on            |
| `explicit_discrepancy_term`                       | the discrepancy term is declared and estimated              |
| `contraction_and_prediction_reported_separately`  | neither statistic is reported as the other                  |
| `participant_held_out_beats_population_baseline`  | new-participant calibration adds out-of-sample value        |
| `trajectory_held_out_beats_uncalibrated_baseline` | within-participant calibration adds out-of-sample value     |
| `null_and_nontransportable_results_retained`      | nontransportable cells are kept, not dropped                |
| `no_personalized_recommendation_emitted`          | no statement is issued outside the declared evidence domain |

`private_governed_authority_contract` is
`blocked_no_governed_participant_data`, so the promotion decision is
`synthetic_benchmark_qualified` and human promotion stays blocked.

## Reproduce

```bash
python3 -m scripts.research.proximal_distal_energy.run_participant_twin_calibration write
python3 -m scripts.research.proximal_distal_energy.run_participant_twin_calibration validate
python3 -m pytest -n 0 tests/research/test_participant_twin_provenance.py
python3 -m pytest -n 0 tests/research/test_participant_twin_calibration.py
```
