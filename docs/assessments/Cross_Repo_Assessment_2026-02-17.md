# Cross-Repo Quality Assessment (A-O + Pragmatic + DbC/DRY/TDD)

Date: 2026-02-17
Scope: AffineDrift, UpstreamDrift, Gasification_Model, Tools

## Method

- Overall score weighting:
  - 70%: A-O quality dimensions (A..O)
  - 30%: principles (DbC, DRY, TDD, Reversibility, Reusability, Orthogonality)
- Evidence sources:
  - repo sync/state checks
  - static code metrics (size, tests, exception patterns, wildcard imports, placeholders)
  - repository quality gates (pre-commit, pytest/mypy config, workflow presence)

## Weighted Scorecard

| Repo               | Overall | A-O Subscore | Principles Subscore |
| ------------------ | ------: | -----------: | ------------------: |
| AffineDrift        |    82.4 |           83 |                  81 |
| UpstreamDrift      |    62.9 |           62 |                  65 |
| Gasification_Model |    69.8 |           68 |                  74 |
| Tools              |    62.8 |           64 |                  60 |

## Principles Breakdown

### AffineDrift

- DbC: 84
- DRY: 75
- TDD: 70
- Reversibility: 80
- Reusability: 78
- Orthogonality: 76

### UpstreamDrift

- DbC: 78
- DRY: 58
- TDD: 60
- Reversibility: 52
- Reusability: 66
- Orthogonality: 54

### Gasification_Model

- DbC: 86
- DRY: 64
- TDD: 68
- Reversibility: 62
- Reusability: 72
- Orthogonality: 60

### Tools

- DbC: 70
- DRY: 55
- TDD: 50
- Reversibility: 58
- Reusability: 74
- Orthogonality: 57

## Top Risk Summary

### AffineDrift

- Moderate DRY/orthogonality pressure in wrist/tooling workflows.
- Placeholder/stub residue still present.
- Assessment/doc churn requires stronger governance.

### UpstreamDrift

- Significant monolith pressure (many oversized files).
- Orthogonality and boundary leakage between UI/engine/shared layers.
- Stub/placeholder residue and broad exceptions remain active risks.

### Gasification_Model

- Broad exception usage is the dominant runtime quality risk.
- Large tab/engine modules reduce modularity and reversibility.
- Local submodule hygiene requires policy and guardrails.

### Tools

- Test depth trails code volume in critical shared modules.
- Multiple oversized interfaces and implementation placeholders.
- Broad exception handling still obscures failure semantics.

## GitHub Issues Created from this Assessment

### AffineDrift

1. https://github.com/D-sorganization/AffineDrift/issues/1230
2. https://github.com/D-sorganization/AffineDrift/issues/1231
3. https://github.com/D-sorganization/AffineDrift/issues/1232
4. https://github.com/D-sorganization/AffineDrift/issues/1233

### UpstreamDrift

1. https://github.com/D-sorganization/UpstreamDrift/issues/1463
2. https://github.com/D-sorganization/UpstreamDrift/issues/1464
3. https://github.com/D-sorganization/UpstreamDrift/issues/1465
4. https://github.com/D-sorganization/UpstreamDrift/issues/1466
5. https://github.com/D-sorganization/UpstreamDrift/issues/1467
6. https://github.com/D-sorganization/UpstreamDrift/issues/1468

### Gasification_Model

1. https://github.com/D-sorganization/Gasification_Model/issues/1450
2. https://github.com/D-sorganization/Gasification_Model/issues/1451
3. https://github.com/D-sorganization/Gasification_Model/issues/1452
4. https://github.com/D-sorganization/Gasification_Model/issues/1453
5. https://github.com/D-sorganization/Gasification_Model/issues/1454

### Tools

1. https://github.com/D-sorganization/Tools/issues/827
2. https://github.com/D-sorganization/Tools/issues/828
3. https://github.com/D-sorganization/Tools/issues/829
4. https://github.com/D-sorganization/Tools/issues/830
5. https://github.com/D-sorganization/Tools/issues/831
6. https://github.com/D-sorganization/Tools/issues/832

## Execution Priority (Cross-Repo)

1. Decompose high-risk monoliths and restore layer boundaries.
2. Replace broad exceptions with typed/domain errors and add failure-mode tests.
3. Raise TDD depth in lowest-ratio repos and hotspot modules.
4. Consolidate duplicate algorithmic kernels for DRY.
5. Formalize reversibility controls (migration playbooks, submodule hygiene, rollback-safe changes).

## Notes

- All repos were synced to remote before assessment.
- Gasification_Model remained on its active feature branch and preserved local `vendor/ud-tools` state.
