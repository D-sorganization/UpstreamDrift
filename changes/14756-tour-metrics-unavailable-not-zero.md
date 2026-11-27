---
issue: 14756
summary: "Report unavailable instead of a perfect 0.0 for empty-mask tour metrics (strict-JSON null at persistence), and refuse all-invalid captures."
---

P7 of the #14756 lineage/parity epic: compute_shared_metrics no longer converts empty valid-marker masks, absent club/pelvis labels or empty windows into 0.0 RMSE; all-invalid input raises ValueError. Unavailable metrics are NaN on SharedMetrics and `None` (strict-JSON null) in `as_dict()`; CandidateReplayReceipt accepts `float | None` while still rejecting NaN/inf, and replay acceptance treats a null metric as missing evidence and fails closed. Thresholds are unchanged.
