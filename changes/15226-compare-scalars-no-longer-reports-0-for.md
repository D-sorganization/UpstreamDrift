---
issue: 15226
summary: "compare_scalars no longer reports 0% for opposite-sign or NaN values"
---

Evaluate the symmetric percentage after dividing both finite values by their
maximum magnitude. This is algebraically the same signed percentage,
`100 * (a - b) / mean(abs(a), abs(b))`, while avoiding overflow in sums or
differences and underflow from halving subnormal inputs. Both zeros still
return zero; NaN or infinite inputs produce an undefined percentage. The raw
difference retains floating-point subtraction semantics.

Reproduce the boundary checks with
`python3 -m pytest tests/unit/shared_python/test_compare_scalars_percent_diff.py tests/analysis/test_comparative_analysis.py`.
