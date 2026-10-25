# After One Solution: Workflow Mapping to the Null-Space API

This note maps the eleven sections of the AffineDrift article "After One
Solution: Exploring Dynamically Equivalent Force Allocations" (planned route
`articles/after-one-solution-null-space-exploration.qmd`, AffineDrift #5096,
epic #5093) to the `loop_nullspace` API. It belongs to NSE-UD-4a (#13464) and
builds on NSX (#12593) and on NSE-UD-1 (#13461), NSE-UD-2 (#13462) and
NSE-UD-3 (#13463), which are draft pull requests (#13475, #13470, #13467) at
the time of writing. The article owns the mathematics; this note owns only the
pointers into code and the limits of what the code certifies.

It is a research record, not a design-manual page. It makes no claim about any
golfer, and it makes no claim that a green test is scientific or publication
approval (see `publication_gate.md`).

## Runnable Example

[`after_one_solution_example.py`](after_one_solution_example.py) runs the
workflow on the generic planar fixture (`loop_nullspace.fixtures`), the circle
and cross solution sets, and a three-muscle segment fibre. It needs no
pinocchio. From the repository root, with the pinned Tools tree initialised
(`git submodule update --init vendor/ud-tools`):

```bash
PYTHONPATH=. python3 docs/research/loop_nullspace/after_one_solution_example.py
python3 -m pytest tests/unit/loop_nullspace/test_after_one_solution_example.py -n 0 -q
```

The smoke test `test_after_one_solution_example.py` runs the script and checks
each reported number against an independent statement of the property. The box
on the reduced coordinates in the example is declared (`|z_i| <= 1`), not
derived from any capacity data.

## Section Mapping

Every name in the right-hand column was read from the source of the package;
`test_every_named_api_exists_in_the_package` fails if one disappears.

| Section | Article section                                                      | API in `src/shared/python/loop_nullspace`                                                                                                                                                                                                                                                                                                                                                                                  | Verified by                                                                                          |
| ------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 1       | Setting: admitted trajectory, `A x = b`, exact versus fitted balance | `admit` <!-- api: admit --> with `AdmissionGates` <!-- api: AdmissionGates -->, `decompose` <!-- api: decompose -->, `decompose_trajectory` <!-- api: decompose_trajectory -->, `BalanceInconsistentError` <!-- api: BalanceInconsistentError -->                                                                                                                                                                          | `test_admission.py`, `test_decomposition.py`                                                         |
| 2       | Affine fibre and dimension accounting                                | `dimension_accounting` <!-- api: dimension_accounting -->, `Decomposition` <!-- api: Decomposition --> (`nullity`, `multiplier_gauge`, `attainable_torque_variation`, `x`)                                                                                                                                                                                                                                                 | `test_decomposition.py`; example section `fibre`                                                     |
| 3       | Metrics select representatives, not fibres                           | `AllocationMetric` <!-- api: AllocationMetric -->, `CharacteristicScales` <!-- api: CharacteristicScales -->, `min_metric_torque` <!-- api: min_metric_torque -->, `min_metric_equivalent` <!-- api: min_metric_equivalent -->                                                                                                                                                                                             | `test_decomposition.py`, `test_invariants.py`                                                        |
| 4       | Physical constraints, Phase-I, Chebyshev centre, conflict sets       | `FeasibleSet` <!-- api: FeasibleSet --> (`from_hypotheses`, `is_empty`, `conflict_set`), `chebyshev_margin` <!-- api: chebyshev_margin -->, `interior_point` <!-- api: interior_point -->                                                                                                                                                                                                                                  | `test_feasible_set.py`, `test_envelopes.py`                                                          |
| 5       | Optimising in reduced coordinates, selection criteria, priorities    | `allocate_trajectory` <!-- api: allocate_trajectory --> with `ObjectiveSpec` <!-- api: ObjectiveSpec -->, `pareto_front` <!-- api: pareto_front --> (epsilon-constraint trade-offs; no dedicated lexicographic routine exists)                                                                                                                                                                                             | `test_allocation.py`, `test_pareto.py`                                                               |
| 6       | Exploring: LP envelopes with certificates, hit-and-run               | `static_envelopes` <!-- api: static_envelopes -->, `x_functional_quantities` <!-- api: x_functional_quantities -->, `joint_torque_quantities` <!-- api: joint_torque_quantities -->, `hit_and_run` <!-- api: hit_and_run -->, `TrajectoryDomain` <!-- api: TrajectoryDomain -->, `run_chains` <!-- api: run_chains -->, `split_rhat` <!-- api: split_rhat -->, `effective_sample_size` <!-- api: effective_sample_size --> | `test_envelopes.py`, `test_hit_and_run.py`, `test_mixing_diagnostics.py`; example section `envelope` |
| 7       | What is identifiable: invariance, attribution intervals              | `classify_functional` <!-- api: classify_functional -->, `FunctionalStatus` <!-- api: FunctionalStatus -->, `functional_record` <!-- api: functional_record -->, `attribution_quantities` <!-- api: attribution_quantities --> (NSE-UD-1)                                                                                                                                                                                  | `test_functional_identifiability.py`; example sections `identifiability`, `attribution`              |
| 8       | Time coupling only shrinks envelopes; warm starts                    | `trajectory_envelope` <!-- api: trajectory_envelope --> (rate-coupled), `static_envelopes` with `warm_start`                                                                                                                                                                                                                                                                                                               | `test_probes.py` (coupled golden; no coupling equals static)                                         |
| 9       | Nonlinear: IFT chart, predictor-corrector, rank-drop counterexample  | `local_chart` <!-- api: local_chart -->, `predictor_corrector_step` <!-- api: predictor_corrector_step -->, `ManifoldSingularityError` <!-- api: ManifoldSingularityError -->, `ManifoldConvergenceError` <!-- api: ManifoldConvergenceError -->, `assess_nonlinearity` <!-- api: assess_nonlinearity --> (NSE-UD-2)                                                                                                       | `test_manifold_continuation.py`; example section `chart`                                             |
| 10      | How one solution buys the rest                                       | `ActivationFiber` <!-- api: ActivationFiber --> (`hull`, `hull_dimension`, `sample_result`, `level_subfibre_dimension`), `activation_fiber` <!-- api: activation_fiber -->, `ensemble_envelope` <!-- api: ensemble_envelope -->, `envelope_union_intersection` <!-- api: envelope_union_intersection --> (NSE-UD-3)                                                                                                        | `test_muscle_fiber_hull.py`; example section `muscle`                                                |
| 11      | Using it in UpstreamDrift                                            | This note, the example script and `README.md`                                                                                                                                                                                                                                                                                                                                                                              | `test_after_one_solution_example.py`                                                                 |

Names in the table that are attributes rather than top-level functions
(`from_hypotheses`, `hull`, `level_subfibre_dimension`, ...) are checked against
the members of their classes.

## What Each Mechanism Certifies

### Numerical Invariance Versus Exact Invariance

The exact statement is the theorem: `s . x` is constant over the affine fibre
`{x_p + N z}` if and only if `N^T s = 0`, that is `s` lies in `range(A^T)`.
`classify_functional` does not prove it. It computes the normalised annihilation
residual `rho = ||N^T s|| / ||s_x * s||` and labels the functional
`NUMERICALLY_INVARIANT` when `rho <= tol` at the declared tolerance, and
`FAMILY_DEPENDENT` otherwise. This is numerical invariance at a declared
tolerance, with the residual reported, and it is distinct from exact invariance.
`rho` lies in `[0, 1]`, is unchanged when `s` is rescaled, and is used for
classification only. `N` already carries `variable_scale`, so it is not scaled a
second time. `functional_record` turns the label into an `INVARIANT` or
`FAMILY_DEPENDENT` `QuantityRecord`; the latter needs at least one hypothesis.

### Absolute Bound on a Bounded Domain

When a bounded `FeasibleSet` and an enclosing radius `R` are supplied,
`classify_functional` also returns `|s . (x - x0)| <= ||N^T s|| R` and the full
width bound `2 ||N^T s|| R`, in the units of `s . x`. `R` must be an enclosing
radius of the reduced-coordinate domain about the chosen centre, certified by the
caller. It is not the Chebyshev inscribed radius, and the bound is not
`rho * R`. On an unbounded fibre any nonzero residual permits unbounded
variation, and a small `rho` then bounds nothing.

### Conic Caveat for Envelopes

`static_envelopes` certifies LP extrema over the declared polygonal
approximation of any friction cone. An outer approximation therefore gives
certified outer bounds on the exact cone set; an inner approximation gives
inner bounds only. There is no exact second-order-cone envelope claim anywhere in
this workflow. `attribution_quantities` is built on `x_functional_quantities`, so
its intervals inherit exactly this statement. Marginal extrema of different
quantities are not jointly attainable: each interval is separate.

### Local-Only Continuation

`local_chart` and `predictor_corrector_step` are valid only where the Jacobian
rank is locally constant. The tangent predictor leaves a residual of order `h^2`
and the minimum-norm Newton corrector converges locally and quadratically; no
higher order is claimed, and the correction is not the nearest-point projection.
A rank change raises `ManifoldSingularityError` (the cross `F = u1 u2` at the
origin in the example). Continuation does not certify coverage of a connected
component. The tangent basis is orthonormal in the declared scaling, not in raw
coordinates.

### Fibre Dimension From the True Affine Hull

`ActivationFiber.hull_dimension` is the dimension of the true affine hull, with
implicit equalities found by LP. It equals `m - rank G` when a strictly
box-feasible point exists (a sufficient condition, not a necessary one). In the example, `G = [40 40 0]` N m with `b = 80` N m
forces the first two activations to 1: the fibre is a segment of dimension 1
although `dim ker G = 2`. `sample_result` reports when the fibre is a single
point instead of repeating one point silently. Samples are coverage, not a
probability claim.

## Publication Gate Rows That Apply

Checked against `publication_gate.md`; nothing in the gate is changed by this
note.

- Non-claims and evidence axes: the note states no result about a golfer, uses
  `allocation_dependence` labels exactly as `evidence.py` defines them, and
  keeps the validation status `UNVALIDATED`.
- Design-manual governance: no governed calculation pathway is touched; the
  governance gate still reports release as blocked.
- Reproducibility: the exact commands are listed above; every sampled output is
  seeded (`SEED = 13464` in the example).
- Titles and headings use title case.

All other rows (external validation, cross-engine conformance, variability
sensitivity claims) concern results this note does not publish.
