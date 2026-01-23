# Test Coverage Recovery Plan

**Created**: 2026-01-22
**Related Issues**: #520, #533, #534

## Current State

- **Coverage**: ~10-15% (reduced from 60% target)
- **Test Files**: 262+ test files exist
- **Problem**: Many tests skip due to missing dependencies

## Staged Recovery Milestones

### Phase 1: Foundation (Current - Month 1)
**Target Coverage**: 10% (current baseline)

- [x] Configure pytest coverage in pyproject.toml
- [x] Add timeout enforcement (120s per test)
- [x] Fix Windows DLL initialization issues (conftest.py)
- [x] Enable Xvfb for GUI tests in CI
- [ ] Audit and document all skip reasons
- [ ] Ensure core MuJoCo tests pass consistently

### Phase 2: Core Coverage (Month 2)
**Target Coverage**: 25%

Focus areas:
- Shared utilities (`shared/python/`)
- Engine manager and registry
- Video pose pipeline
- Analysis service

Action items:
- [ ] Add tests for `shared/python/analysis/`
- [ ] Add tests for `shared/python/engine_manager.py`
- [ ] Reduce excessive mocking in existing tests
- [ ] Fix top 10 most-skipped test modules

### Phase 3: API & Services (Month 3)
**Target Coverage**: 40%

Focus areas:
- API endpoints (`api/`)
- Authentication flows
- Simulation service
- Analysis service

Action items:
- [ ] Add integration tests for API routes
- [ ] Test auth flows end-to-end
- [ ] Add functional tests for dashboard components (#535)

### Phase 4: Physics Engines (Month 4)
**Target Coverage**: 50%

Focus areas:
- MuJoCo engine tests
- Cross-engine validation
- Physics computation paths

Action items:
- [ ] Standardize coverage across engines (#534)
- [ ] Add cross-engine integration tests (#126)
- [ ] Test kinematics and dynamics calculations

### Phase 5: Target State (Month 5-6)
**Target Coverage**: 60%

Focus areas:
- AI/Workflow components
- Injury analysis module
- Optimization module
- Full integration tests

## Coverage Requirements by Module

| Module | Current | Phase 2 | Phase 4 | Target |
|--------|---------|---------|---------|--------|
| shared/ | 10% | 25% | 40% | 60% |
| api/ | 15% | 30% | 45% | 60% |
| engines/ | 10% | 20% | 50% | 60% |
| launchers/ | 5% | 15% | 30% | 50% |

## CI/CD Integration

1. **Current**: `--cov-fail-under=10` (emergency threshold)
2. **Phase 2**: Raise to 20%
3. **Phase 4**: Raise to 40%
4. **Target**: 60%

## Monitoring

- Coverage reports generated on every PR
- HTML reports uploaded as CI artifacts
- Codecov integration for trend tracking

## References

- [TEST_AUDIT_REPORT.md](../TEST_AUDIT_REPORT.md)
- pyproject.toml `[tool.pytest.ini_options]`
- pyproject.toml `[tool.coverage.*]`
