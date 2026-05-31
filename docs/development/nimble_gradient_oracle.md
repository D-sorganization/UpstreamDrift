# Nimble Gradient Oracle

Issue #6795 adds an offline-only validation surface for comparing canonical-core
candidate gradients against Stanford Nimble. Nimble is pinned as an explicit
extra and is never imported from runtime `src/` modules:

```bash
python -m pip install "upstream-drift[nimble-oracle]"
python -m pytest tests/unit/tools/test_nimble_gradient_oracle.py -m requires_nimble
```

The pinned package is `nimblephysics==0.10.52.2`. The extra also installs
PyTorch because Nimble exposes differentiable steps as tensor operations. This
stack is intentionally excluded from `all-engines`: wheel coverage is uneven,
and the oracle is for local or scheduled validation jobs only.

## Surface

Use `tools.offline_validation.nimble_gradient_oracle` for deterministic
request/response comparisons:

- `NimbleGradientOracleRequest` carries a model name, coordinate vector,
  candidate gradient, scalar `nimble_loss(nimble, torch_tensor)`, tolerance, and
  optional metadata.
- `compare_nimble_gradient(...)` returns `passed`, `failed`, or `skipped`.
  Missing Nimble produces `skipped` by default so normal CI remains green.
- `require_available=True` turns missing Nimble into
  `GradientOracleUnavailable` for dedicated oracle jobs.
- `TorchAutogradNimbleBackend` imports `nimblephysics` and `torch` lazily, then
  uses tensor backpropagation to compute the oracle gradient.

CC-18 residual functions can feed this oracle by passing the same canonical-v2
coordinates used by their JAX/Pinocchio gradient path as `coordinates`, the
candidate Jacobian row or reduced gradient as `candidate_gradient`, and a Nimble
toy-model loss that evaluates the corresponding scalar residual objective. Keep
toy models small, contact-free when possible, and tolerance-based; this oracle
guards sign, ordering, and scaling mistakes rather than acting as a production
engine.

## Runtime Boundary

The focused unit suite asserts no Python file under `src/` imports
`nimblephysics` or `nimble`. If production code needs differentiable physics,
use the canonical JAX/JaxSim or future MJX backend path documented in
ADR-0024/ADR-0025; do not route application code through Nimble.
