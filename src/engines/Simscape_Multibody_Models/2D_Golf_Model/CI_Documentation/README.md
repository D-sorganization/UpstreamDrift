# CI/CD Documentation

This directory contains supplementary CI/CD documentation for the 2D_Golf_Model repository and related D-sorganization projects.

## Document Structure

### Main Documents

- **[../UNIFIED_CI_APPROACH.md](../UNIFIED_CI_APPROACH.md)** - **START HERE**
  - Central CI/CD documentation for all 15 D-sorganization repositories
  - Unified standards for Python, MATLAB, JavaScript/TypeScript, and Shell scripts
  - Comprehensive workflow examples and best practices
  - Required reading for all CI/CD work

### Technology-Specific Guides

- **[MATLAB_COMPLIANCE.md](MATLAB_COMPLIANCE.md)** - MATLAB Quality Standards
  - Code quality checks with checkcode
  - Unit testing with MATLAB Unit Test framework
  - Documentation standards
  - Reproducibility requirements
  - Local development guide
  - CI integration instructions

## Quick Links

### For Developers

**Python Development:**
- See [UNIFIED_CI_APPROACH.md - Python CI Workflow](../UNIFIED_CI_APPROACH.md#python-ci-workflow)
- Tool versions: ruff==0.5.0, mypy==1.10.0, black==24.4.2

**MATLAB Development:**
- See [MATLAB_COMPLIANCE.md](MATLAB_COMPLIANCE.md) for complete guide
- Local checks: Run `matlab -batch "run_matlab_quality_checks"`
- CI checks: Bash-based checks run automatically (no MATLAB license needed)

**JavaScript/TypeScript Development:**
- See [UNIFIED_CI_APPROACH.md - JavaScript/TypeScript CI Workflow](../UNIFIED_CI_APPROACH.md#javascripttypescript-ci-workflow)

### For CI/CD Maintenance

**Updating CI Workflows:**
1. Review [UNIFIED_CI_APPROACH.md](../UNIFIED_CI_APPROACH.md) for standards
2. Check technology-specific sections
3. Ensure pinned versions are up to date
4. Test locally before committing

**Adding New Repositories:**
1. Follow patterns in UNIFIED_CI_APPROACH.md
2. Use appropriate technology-specific workflow template
3. Include replicant branch support if applicable
4. Add fail-fast strategy to matrix builds

## Key Principles

From UNIFIED_CI_APPROACH.md:

1. ✅ **Pinned Versions** - All tools explicitly versioned
2. ✅ **Comprehensive Detection** - Auto-detect source directories
3. ✅ **Proper Exit Codes** - Preserve failures with `|| exit 1`
4. ✅ **Conditional Uploads** - Check file existence before upload
5. ✅ **Security Checks** - Bandit, Safety, TruffleHog
6. ✅ **Documentation Checks** - Markdown linting, link validation
7. ✅ **Replicant Branch Support** - Include `claude/*_Replicants` patterns
8. ✅ **Quality Check Scripts** - Support multiple locations
9. ✅ **Fail-Fast Strategy** - `fail-fast: true` in matrix
10. ✅ **Cache Patterns** - Comprehensive dependency caching

## Repository-Specific Information

### 2D_Golf_Model

**Tech Stack:**
- MATLAB (primary) - 655 files in `matlab/`, 30 in `matlab_optimized/`
- Python (secondary) - Data processing and analysis

**CI Checks:**
- Python: Full pytest, ruff, mypy, black, coverage
- MATLAB: Bash-based checks (magic numbers, approximations, seeds)
- MATLAB (optional): Full checkcode and unit tests when license available

**Local Testing:**
```bash
# Python checks
python scripts/quality_check.py
pytest python/tests/

# MATLAB checks (if MATLAB installed)
matlab -batch "run_matlab_quality_checks"
matlab -batch "results = runtests('matlab/tests'); disp(results);"
```

## Contributing

When adding new CI/CD documentation:

1. **Update UNIFIED_CI_APPROACH.md** for cross-repository standards
2. **Add technology-specific docs** to this directory
3. **Include code examples** that can be copy-pasted
4. **Document both good and bad patterns** (✅ vs ❌)
5. **Update "Last Updated" dates**
6. **Test all examples** before committing

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [MATLAB Actions](https://github.com/matlab-actions)
- [Codecov Documentation](https://docs.codecov.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)

## Maintenance Schedule

- **Quarterly Review** - Check for tool version updates
- **After Major Changes** - Update documentation to reflect workflow changes
- **New Repository Additions** - Update repository lists
- **Tool Deprecations** - Update to new tools and patterns

---

**Last Updated:** 2025-11-29  
**Maintained By:** CI/CD Agent (@ci-cd-agent)
