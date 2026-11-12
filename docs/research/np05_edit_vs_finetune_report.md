# NP-05: Localized Edit vs Constrained Fine-Tune Benchmark Report

**Issue Reference:** #14480  
**Parent Epic:** #14475 (Exploratory research: Neural policy interventions)  
**Run Record Tracing:** #14436 (Run manifest v1), #14476 (Experiment contract spec v1)  
**Target Policy:** NP-02 compact policy model (`np02_compact_mlp`)  
**Evaluated Seeds:** 42, 43, 44, 45, 46 (N=5)

## Overview & Scope
This benchmark evaluates localized low-rank intervention (NP-03 `apply_low_rank_weight_edit`) against constrained parameter adaptation (NP-04 `LoRAAdapter`) on golf swing trajectory tracking. The comparison measures tracking RMSE, clubhead-speed error, torque constraint violations, held-out side-effect drift, and parameter edit locality.

## Benchmark Results

| Variant | Tracking RMSE (mean±std) | Speed Err (m/s) | Violations | Held-out Drift | Locality |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline` | 0.3047 ± 0.0000 | 0.798 | 3.0 | 0.0000 | 1.00 |
| `rank1_edit` | 0.1141 ± 0.0000 | 0.689 | 1.0 | 0.3145 | 1.00 |
| `lora_finetune` | 0.2704 ± 0.0202 | 0.785 | 2.8 | 0.0429 | 0.70 |

## Key Findings
1. **Locality & Isolation:** The rank-1 edit restricts weight modification strictly to targeted coordinates (`locality = 1.00`). In contrast, LoRA fine-tuning disperses adjustments across rows (`locality = 0.70`).
2. **Tracking & Constraint Performance:** The localized rank-1 edit achieves the largest reduction in tracking RMSE (0.1141 vs 0.3047 baseline) and cuts constraint violations from 3.0 to 1.0 under box projection.
3. **Held-out Stability:** LoRA fine-tuning exhibits lower held-out state drift (0.0429) due to distributed low-magnitude updates, while rank-1 intervention has higher held-out drift (0.3145) requiring validation of the training-state envelope.
4. **Seed Variance:** Rank-1 intervention with deterministic probe directions produces zero seed variance, whereas stochastic adapter rank initialization produces measurable output dispersion.

## Explicit Limitations & Guardrails
- These results represent local input-output mapping adjustments in a compact MLP architecture.
- They make **no physiological coordination or mechanical spring-damper claims**.
- Results do not generalize off-manifold; evaluations are strictly valid within the stated operating envelope.

## Go / No-Go Recommendation for NP-06
- **Status:** **GO** (conditional on Dieter's sign-off).
- **Rationale:** Localized low-rank edits demonstrate targeted error reduction and perfect coordinate isolation. Goal-conditioned policy studies (NP-06, #14482) can safely leverage localized edit formulations for coordinate-specific adjustments.
