---
issue: 14206
summary: "Conflict Disarm runs its job inline (reading RM's conflict_disarm script via an App-token checkout) instead of calling an internal Repository_Management reusable workflow, which a public repo cannot do, so it no longer fails every run with 0 jobs"
---
