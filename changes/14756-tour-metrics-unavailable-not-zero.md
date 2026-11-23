---
issue: 14756
summary: "Report unavailable (NaN) instead of a perfect 0.0 for empty-mask tour metrics, and refuse all-invalid captures."
---

P7 of the #14756 lineage/parity epic: compute_shared_metrics no longer converts empty valid-marker masks, absent club/pelvis labels or empty windows into 0.0 RMSE; all-invalid input raises ValueError. Thresholds are unchanged.
