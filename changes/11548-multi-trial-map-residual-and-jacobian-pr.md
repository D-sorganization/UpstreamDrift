---
issue: 11548
summary: 'Multi-trial MAP: residual and Jacobian prior rows share one layout (locked priors excluded from both); posterior covariance is the Schur marginal over the shared parameters, reported as unavailable (`covariance_status="rank_deficient"`, NaN) when the Fisher matrix is singular; calculation documented in docs/conventions/canonical-v2.md §6.1'
---
