# Package Organization — `src/shared/python/`

> **Status**: Active · **ADR**: [ADR-0008 Shared Python Package Boundaries](../adr/0008-shared-python-package-boundaries.md)

This document answers the question: **"When does code belong in `src/shared/python/` vs. elsewhere?"**

## Decision Rule

Code belongs in `src/shared/python/` **only when it is consumed by more than one top-level product** (UpstreamDrift, Tools, Gasification_Model, or a future product).

Code that is used only by UpstreamDrift belongs in `src/` (non-shared root packages).

## Package Categories

### ✅ Legitimately Shared (consumed cross-product)

| Package            | Consumed By | Purpose                                  |
| ------------------ | ----------- | ---------------------------------------- |
| `engine_core/`     | UD + Tools  | Abstract engine contract, EngineRegistry |
| `motion_matching/` | UD + Tools  | Cross-engine parity spec types           |
| `contracts/`       | UD + Tools  | Design-by-contract decorators            |
| `logging_pkg/`     | UD + Tools  | Canonical `get_logger()`                 |
| `realtime/`        | UD + Tools  | WS pub/sub backend                       |
| `signal_toolkit/`  | UD + Tools  | DSP primitives                           |
| `validation_pkg/`  | UD + Tools  | Pydantic validators, schema helpers      |
| `spatial_algebra/` | UD + Tools  | `SE3`, `pose6dof`, `quaternion`          |

### ⚠️ Questionable (UpstreamDrift-specific, needs audit)

| Package            | Problem             | Action                             |
| ------------------ | ------------------- | ---------------------------------- |
| `ai/`              | UD-specific chat/AI | Move to `src/ai/` in a future PR   |
| `chat/`            | UD-specific UI      | Move to `src/chat/` in a future PR |
| `anthropometrics/` | UD-specific         | Move to `src/anthropometrics/`     |
| `biomechanics/`    | UD-specific         | Move to `src/biomechanics/`        |
| `body_part_viz/`   | UD-specific         | Move to `src/body_part_viz/`       |

### ❌ Deprecated (do not add new imports from these)

| Package                 | Reason                       | Replacement                         |
| ----------------------- | ---------------------------- | ----------------------------------- |
| `upstream_drift_tools/` | Superseded by `engine_core/` | Use `src.shared.python.engine_core` |

## Naming Rules

To reduce the number of `data_processing` vs `data_processor` style collisions, we enforce:

1. **Noun phrases, not verb phrases**: `biomechanics` not `biomech`; `plot_engine` not `plotting`.
2. **No suffix-only differentiation**: Don't create `foo` and `foo_pkg` for the same concept.
3. **One canonical name**: If a concept already has a package, add to it — don't create a sibling.

## Package Cohesion Checklist

Before adding a new package to `src/shared/python/`, confirm:

- [ ] Is it used by ≥ 2 products today?
- [ ] Is it described in an ADR?
- [ ] Does it have a `__init__.py` with `__all__`?
- [ ] Does it have a dedicated test directory under `tests/unit/`?
- [ ] Does it avoid importing from `src/api/` or product-specific `src/` packages?

## Known Issues (tracked)

- **Theme system fragmentation**: 4 independent theme implementations exist (issue #5908). Canonical is `src/shared/python/theme/`; others should become adapters.
- **Contracts split**: 3 coexisting contracts implementations (issue #5908). Canonical is `src/shared/python/core/contracts/`.
- **`upstream_drift_tools/` directory**: Deprecated but still ships (issue #5908). Queued for deletion in cleanup sprint.
