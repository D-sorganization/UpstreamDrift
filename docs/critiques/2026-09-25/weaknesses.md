# Weakness Catalog — 2026-09-25

**Repository:** D-sorganization/UpstreamDrift
**Critic seat:** Bravo (fleet-critic)
**Review window:** commits since 2026-08-26 (NM-09 through NM-12, TB-08 through TB-10,
and the Bolt optimization sequence)

---

## Weakness 1: Synthetic Multi-Seed Evidence

### Summary of Concern

`build_checkpoint_matrix()` claims to provide three-seed training reproducibility
evidence for every neural model. In practice the "three seeds" are computed by
multiplying a single `seed_loss_base` scalar by fixed constants (0.98, 1.03,
1.0033, 0.021). All four output numbers — seed-11 loss, seed-22 loss, mean, and
standard deviation — are deterministic functions of the single base value. No
second or third training run is invoked; `converged=True` is always hard-coded.

### Location

- **File:** `src/shared/python/neural_motion/matrix/builder.py`
- **Lines:** 88–98 (`_make_evidence`)
- **Claim:** "3-seed evidence" per `MATRIX_SCHEMA = "neural-checkpoint-matrix/1.0.0"`

### Nature of the Issue

- [x] Empirical insufficiency
- [x] Unstated assumption
- [x] Terminological ambiguity

### Why This Is a Problem

A hostile reviewer inspecting the checkpoint matrix will see seed-11 loss,
seed-22 loss, and seed-33 loss as independent measurements confirming
convergence. The ratio of the first two losses is always 0.98; the ratio of the
first and third is always 1.03; the standard deviation is always 2.1% of the
mean. This is statistically impossible under genuine independent training with
different random seeds on non-trivial data. Any reviewer who computes the
coefficient of variation across the three seeds will immediately identify the
fabrication.

Because the checkpoint matrix feeds `build_reproduction_catalog()` (NM-12),
every published model card carries this synthetic evidence.

### Evidence / References

- True multi-seed variance on neural network training is strongly data- and
  architecture-dependent. Published benchmark reproducibility surveys (e.g.
  Bouthillier et al., 2019 "Unreproducible Research is Reproducible") report
  inter-seed loss variance of 2–25% depending on dataset size and optimizer
  — never deterministically 2.1%.
- The Python call path:
  `build_checkpoint_matrix()` → `_build_card_for_model()` →
  `_make_evidence(0.04 + 0.002 * (identity.dof % 5))`.

### Severity

- [x] High (core claim at risk)

### Suggested Remedies

- Remove `_make_evidence()` and replace the `three_seed_evidence` field with an
  explicit `Optional[ThreeSeedEvidence]` that is `None` until real runs are
  available.
- Add a `SEED_EVIDENCE_UNAVAILABLE` sentinel to `ModelCheckpointStatus` and
  block `QUALIFIED_NATIVE` promotion when evidence is absent.
- In CI, gate the published catalog on non-`None` evidence for any model with
  `status == QUALIFIED_NATIVE`.

---

## Weakness 2: Model-ID-Derived Provenance Hashes

### Summary of Concern

The checkpoint matrix builder constructs dataset, split, and weight-digest hash
strings by slicing a SHA-256 of the _model ID string_:

```python
mid_hash = hashlib.sha256(mid.encode("utf-8")).hexdigest()
dataset_hash = f"data_{mid_hash[:12]}"
split_hash   = f"split_{mid_hash[12:24]}"
weight_digest = f"weight_{mid_hash[24:36]}"
```

These strings look like cryptographic provenance but are injective
transformations of the model name. They do not bind to any data file, split
configuration, or checkpoint file. Replacing the actual parquet dataset with
random noise would not change a single hash. The `matrix_digest()` method
(line 67–78) compounds the problem: it hashes the matrix JSON, so the digest
changes when model names change but not when data changes.

### Location

- **File:** `src/shared/python/neural_motion/matrix/builder.py`
- **Lines:** 223–261 (`_build_card_for_model`)
- **Fields:** `dataset_hash`, `split_hash`, `weight_digest`

### Nature of the Issue

- [x] Logical gap
- [x] Unstated assumption
- [x] Empirical insufficiency

### Why This Is a Problem

A data-poisoning or accidental-corruption scenario (wrong file path, dataset
regenerated with a different seed) would be completely invisible to the hash
chain. The chain provides a false sense of data integrity. A reviewer verifying
model provenance by re-running the matrix and comparing hashes would get bit-for-bit
agreement regardless of what dataset was used — the hash is entirely determined
by the model ID string.

The NM-12 model card schema (`neural-model-reproduction-card/1.0.0`) positions
these hashes as reproduction evidence. Including them in a board packet without
noting that they do not hash actual files would be misleading.

### Evidence / References

- Compare `compute_package_digest()` in `tour_baselines/qualification.py`
  (lines 295–311), which correctly hashes actual trajectory arrays and manifest
  JSON. The checkpoint matrix builder applies a superficially similar pattern
  but never touches the data.

### Severity

- [x] High (core claim at risk)

### Suggested Remedies

- Accept `dataset_path: Path` and `checkpoint_path: Path` as inputs to
  `_build_card_for_model` and compute hashes from file contents.
- Where real checkpoints do not yet exist, set `dataset_hash = None` and
  document the gap rather than filling it with a model-name transform.
- Add a CI check that no `QUALIFIED_NATIVE` card carries null-or-string-derived
  hashes.

---

## Weakness 3: Self-Referential Speedup Baseline

### Summary of Concern

`run_model_comparative_benchmark()` accepts an optional `baseline_latency`. When
it is absent, the function synthesizes one by multiplying the _candidate's own_
latency samples by 2.5 (median), 2.0 (p95, min, max):

```python
effective_baseline_lat = baseline_latency or LatencySummary(
    median_s=latency_summary.median_s * 2.5,
    ...
)
```

Downstream, `compute_break_even()` and `evaluate_promotion_gate()` use this
manufactured baseline. Any model benchmarked without a real external baseline
will always receive `has_break_even=True` and a favorable promotion gate,
because the "classical" reference is defined as 2.5× the candidate.

### Location

- **File:** `src/shared/python/neural_motion/benchmark/runner.py`
- **Lines:** 63–82 (`run_model_comparative_benchmark`)

### Nature of the Issue

- [x] Logical gap
- [x] Empirical insufficiency

### Why This Is a Problem

The purpose of a benchmark is to compare two independently measured quantities.
Constructing one from the other eliminates the comparison entirely. A benchmark
card generated this way carries the vocabulary of rigor (median, p95, break-even
queries) while measuring nothing. If these cards reach the board as evidence of
speedup, the board cannot distinguish real acceleration from a tautology.

An adversarial reviewer who traces `ModelBenchmarkCard.promotion` back to the
speedup gate and then to the baseline construction will invalidate the entire
NM-10 benchmark chapter.

### Evidence / References

- The Surrogate Training Guide (`docs/motion_matching/SURROGATE_TRAINING_GUIDE.md`,
  lines 178–186) cites CPU batch latency of ~30 ms/batch (1 ms/sample) as a
  _target_, not a measured result. This is the intended real baseline; its
  absence in the benchmark runner means the runner never uses it.

### Severity

- [x] High (core claim at risk)

### Suggested Remedies

- Make `baseline_latency` a required parameter (remove the `| None` default) in
  `run_model_comparative_benchmark`, forcing callers to supply a real reference.
- Where no classical baseline exists, return a `ModelBenchmarkCard` with
  `promotion = PromotionDecision.INSUFFICIENT_EVIDENCE` rather than fabricating one.
- Measure and record the Simscape/ODE reference latency as a locked fixture.

---

## Weakness 4: Hardcoded Performance Economics in Model Cards

### Summary of Concern

`_build_performance_economics()` in the reproduction catalog builder returns
hardcoded latency and speedup numbers categorized entirely by `PromotionVerdict`
tier, not by any per-model measurement:

```python
if verdict == PromotionVerdict.PROMOTED:
    return {
        "neural_latency_ms": 1.25,
        "classical_latency_ms": 15.4,
        "speedup_factor": 12.32,
        ...
    }
```

Every PROMOTED model — regardless of architecture size, input dimension, or
hardware — receives the identical `speedup_factor: 12.32`.

### Location

- **File:** `src/shared/python/neural_motion/turnover/catalog.py`
- **Lines:** 114–147 (`_build_performance_economics`)

### Nature of the Issue

- [x] Overgeneralization
- [x] Empirical insufficiency

### Why This Is a Problem

Model cards are designed to communicate model-specific characteristics. Publishing
identical latency numbers for the driven double pendulum (2 DOF) and the
constrained upper body golfer (more DOF, larger network) understates the
architectural cost difference and prevents valid engineering tradeoffs.
If the cards are distributed, recipients may make production deployment decisions
on incorrect per-model benchmarks.

### Severity

- [x] Medium (argument tightening required)

### Suggested Remedies

- Thread actual `ModelBenchmarkCard` data (from NM-10) into the reproduction
  card, rather than looking up a tier-indexed constant.
- Where no benchmark card exists, emit `null` for latency fields rather than
  tier defaults.

---

## Weakness 5: Surrogate Refinement Sensitivity Is a Proxy Calculation

### Summary of Concern

`IndependentBaselineQualifier.evaluate_refinement_sensitivity()` is documented
as evaluating "sensitivity under refined integration timestep." The actual
implementation computes:

```python
coordinate_diff = float(abs(dt_nominal - dt_refined) * 0.1)
stable = coordinate_diff < 0.02
```

This is an analytic proxy: it declares a baseline stable if the timestep change
is less than 0.2 s (so that 10% of 0.2 = 0.02). No integration is re-run; no
trajectory is compared. The "coordinate difference" is ten percent of the
timestep-size difference.

### Location

- **File:** `src/shared/python/tour_baselines/qualification.py`
- **Lines:** 726–743 (`evaluate_refinement_sensitivity`)

### Nature of the Issue

- [x] Empirical insufficiency
- [x] Terminological ambiguity

### Why This Is a Problem

Numerical refinement sensitivity is the most common way to detect stiff dynamics
or insufficiently constrained systems that happen to pass broader metrics. The
planar pendulum model at 100 Hz has known stepsize-dependent artifacts near
impact. An analytic proxy that scales linearly with timestep difference cannot
detect these. A model that produces a very different trajectory at 500 Hz vs.
100 Hz would pass this check as long as the timestep numbers are close enough
in absolute value.

### Severity

- [x] Medium (argument tightening required)

### Suggested Remedies

- Re-integrate using the finer timestep and compute the actual RMS deviation
  of the reconstructed trajectory coordinates between nominal and refined runs.
- If re-integration is computationally expensive, sample 10 representative
  windows and bound the sensitivity; acknowledge the limited sampling in the
  result.
- Until a real integration is available, rename the field to
  `timestep_sensitivity_proxy` and document its limitations in the docstring.

---

## Weakness 6: Bolt Speedup Claims Lack Benchmark Fixtures

### Summary of Concern

The Bolt optimization journal (`.jules/bolt.md`) and the associated commit
messages assert speedup factors of 2×, 1.5×, 10–20×, and 2.4× for various
`np.linalg.norm` → `np.einsum` replacements. No reproducible profiling
artifact, benchmark file, or test fixture is cited or committed. The claims
exist only as prose assertions in commit bodies and the bolt journal.

### Location

- **File:** `.jules/bolt.md` (multiple entries)
- **Files:** various `src/` modules modified in commits #10838, #10939, #10938,
  #10880, #10876, #10844

### Nature of the Issue

- [x] Empirical insufficiency
- [x] Overgeneralization

### Why This Is a Problem

NumPy's `linalg.norm` implementation has improved substantially across versions
(1.23→1.26→2.x). A speedup measured on NumPy 1.24 may not hold on 1.26, and
some einsum patterns are slower than `linalg.norm` on NumPy 2.x with AVX-512
backends. Without pinned benchmark fixtures, the claimed speedups cannot be
verified by CI, and future contributors cannot tell whether a NumPy upgrade
invalidated an optimization. The 10–20× claim for axis=2 operations is an
outlier that warrants particular scrutiny.

More subtly, the Bolt pattern `math.sqrt(np.vdot(v, v))` for small 1-D arrays
silently returns incorrect results for complex-valued inputs (`np.vdot`
conjugates its first argument), and for integer dtypes `np.vdot` may accumulate
in the narrow dtype. The commit `#10838` added an integer-dtype guard after the
fact, which confirms that the pattern was deployed before its edge cases were
understood.

### Evidence / References

- NumPy 2.0 release notes: `np.linalg.norm` for contiguous arrays on modern
  hardware is significantly faster than in 1.x.
- Commit `#10838` body: "fix(physics): handle integer-dtype tangent forces in
  `check_slip_margin`" — edge case discovered after the Bolt optimization was
  merged.

### Severity

- [x] Low (clarification needed)

### Suggested Remedies

- Add a `tests/benchmarks/` directory with `pytest-benchmark` (or `timeit`)
  fixtures that reproduce at least the most aggressive claimed speedups (the
  10–20× motcap norm and the 12.32× neural inference speedup).
- Pin the NumPy version used for the original profiling in each bolt entry.
- Add a mypy/type-checking guard (or runtime assertion) to the
  `math.sqrt(np.vdot(...))` pattern to reject non-float inputs before the
  integer-dtype class of bugs can recur.
