---
issue: 15223
summary: "SwingComparator no longer scores 100 when the reference peak speed, club head speed or tempo ratio is zero."
---

`compare_peak_speeds` and `compare_tempo` turned a zero reference into a 0 % difference and a score of 100, so a student who moved against a reference that did not was reported as a perfect match. The percent difference is now undefined (NaN) with a score of 0 in that case; equal zeros still score 100.
