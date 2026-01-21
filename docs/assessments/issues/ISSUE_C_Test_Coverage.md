---
title: Improve Test Coverage
labels: jules:assessment, needs-attention
---

# Issue: Low Score in Test Coverage

Current Grade: 4/10


## Summary
Test infrastructure is good, but actual code coverage is low (~15%).

### Strengths
- **Infrastructure**: `pytest`, `pytest-cov` configured.
- **Quality**: Existing tests are well-written.

### Weaknesses
- **Low Coverage**: ~15% is significantly below the typical 80% target.
- **Configuration**: `pyproject.toml` sets fail-under to 10%.

### Recommendations
- **Increase Coverage**: Systematically add unit tests for `shared/python`.
- **Raise Threshold**: Gradually increase `fail-under` target in `pyproject.toml`.
