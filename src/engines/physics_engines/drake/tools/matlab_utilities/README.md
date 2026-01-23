# MATLAB Quality Utilities Package

A comprehensive, reusable toolkit for MATLAB code quality checking, testing, and linting. This package can be easily integrated into any MATLAB project.

## Overview

This package provides automated quality control tools for MATLAB projects, including:
- **Code Analysis**: MATLAB mlint/checkcode integration
- **Static Analysis**: Python-based static analysis (no MATLAB license required)
- **Testing Framework**: Automated test execution and reporting
- **CI/CD Integration**: Pre-commit hooks and GitHub Actions support

## Package Structure

```
matlab_utilities/
├── README.md                       # This file
├── quality/                        # Quality checking tools
│   ├── run_quality_checks.m       # Unified quality check interface
│   ├── exportCodeIssues.m         # Code Analyzer export tool
│   └── matlab_quality_config.m    # Quality configuration
├── testing/                        # Testing utilities
│   └── run_matlab_tests.m         # Test suite runner
└── scripts/                        # Cross-platform scripts
    └── matlab_quality_check.py    # Python static analyzer (no MATLAB needed)
```

## Quick Start

### Installation

**Option 1: Copy to Your Project**
```bash
# Copy entire utilities package to your project
cp -r matlab_utilities /path/to/your/project/

# Add to MATLAB path
addpath('matlab_utilities/quality');
addpath('matlab_utilities/testing');
```

**Option 2: Git Submodule (Recommended)**
```bash
# Add as submodule (stays updated across projects)
git submodule add <repository-url> matlab_utilities
git submodule update --init --recursive

# In MATLAB startup.m or project setup:
addpath('matlab_utilities/quality');
addpath('matlab_utilities/testing');
```

**Option 3: Symbolic Link**
```bash
# Create symlink (for local development across multiple projects)
ln -s /path/to/matlab_utilities /path/to/your/project/matlab_utilities
```

### Basic Usage

#### 1. Run Quality Checks (MATLAB)

```matlab
% Check current directory
results = run_quality_checks();

% Check specific directory
results = run_quality_checks('../src');

% Save results to CSV
results = run_quality_checks('.', 'OutputFile', 'quality_report.csv');

% Strict mode (fail on any issues)
results = run_quality_checks('.', 'StrictMode', true);

% Customize exclusions
results = run_quality_checks('.', 'ExcludeDirs', {'.git', 'build', 'external'});
```

#### 2. Run Quality Checks (Python - No MATLAB Required)

```bash
# Basic check
python matlab_utilities/scripts/matlab_quality_check.py

# JSON output for automation
python matlab_utilities/scripts/matlab_quality_check.py --output-format json

# Strict mode
python matlab_utilities/scripts/matlab_quality_check.py --strict

# Custom project root
python matlab_utilities/scripts/matlab_quality_check.py --project-root /path/to/project
```

#### 3. Run Tests (MATLAB)

```matlab
% Run all tests
results = run_matlab_tests();

% Run tests and display detailed results
results = run_matlab_tests('Verbose', true);
```

## Integration with CI/CD

### Local Development: Pre-commit Hooks

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Pre-commit hook for MATLAB quality checks

echo "Running MATLAB quality checks..."

# Option 1: Use Python static analyzer (fast, no MATLAB needed)
python matlab_utilities/scripts/matlab_quality_check.py
if [ $? -ne 0 ]; then
    echo "Quality checks failed. Please fix issues before committing."
    exit 1
fi

# Option 2: Use MATLAB (if available)
if command -v matlab &> /dev/null; then
    matlab -batch "addpath('matlab_utilities/quality'); results = run_quality_checks('.', 'Verbose', false); exit(~results.passed);"
    if [ $? -ne 0 ]; then
        echo "MATLAB quality checks failed. Please fix issues before committing."
        exit 1
    fi
fi

echo "Quality checks passed!"
exit 0
```

### Local Development: Pre-push Hooks

Create `.git/hooks/pre-push` for more comprehensive checks before pushing:

```bash
#!/bin/bash
# Pre-push hook for comprehensive MATLAB quality checks

echo "Running comprehensive MATLAB quality checks..."

# Run MATLAB quality checks (if MATLAB available)
if command -v matlab &> /dev/null; then
    echo "Running MATLAB checkcode analysis..."
    matlab -batch "addpath('matlab_utilities/quality'); results = run_quality_checks('.', 'OutputFile', 'pre-push-quality.csv'); if ~results.passed; error('Quality checks failed'); end"

    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ MATLAB quality checks FAILED!"
        echo "Review the issues in pre-push-quality.csv"
        echo ""
        exit 1
    fi

    echo "✅ MATLAB quality checks PASSED!"
else
    echo "⚠️  MATLAB not found - skipping full quality checks"
    echo "Running Python static analysis instead..."
    python matlab_utilities/scripts/matlab_quality_check.py
fi

exit 0
```

Make hooks executable:
```bash
chmod +x .git/hooks/pre-commit
chmod +x .git/hooks/pre-push
```

### GitHub Actions CI/CD

Add to `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [ main ]

jobs:
  matlab-quality-check:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Run MATLAB Quality Check (Static Analysis)
        run: |
          python matlab_utilities/scripts/matlab_quality_check.py --output-format json

      - name: Upload Quality Report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: matlab-quality-report
          path: quality_report.json
```

### Pre-commit Framework Integration

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: matlab-quality-check
        name: MATLAB Quality Check
        entry: python matlab_utilities/scripts/matlab_quality_check.py
        language: system
        types: [text]
        pass_filenames: false
```

## Quality Checks Performed

### MATLAB Code Analyzer (mlint/checkcode)

When running with MATLAB, the following checks are performed:
- Syntax errors and warnings
- Unused variables
- Unreachable code
- Missing semicolons
- Undefined variables
- Performance issues
- Code complexity warnings

### Python Static Analyzer

The Python-based analyzer checks for:

**Required Structure**:
- Function docstrings
- Arguments validation blocks
- Proper error handling

**Banned Patterns**:
- TODO, FIXME, HACK, XXX placeholders
- Template placeholders (`<VAR>`, `{{var}}`)

**Code Quality Issues**:
- Magic numbers (undefined constants)
- Known physics constants without definition
- Global variables
- Use of `eval()`, `evalin()`, `assignin()`
- `load` without output variable
- `clear`, `clc`, `close all` in functions
- `exist()` usage (prefer validation or try/catch)
- `addpath` in functions

**Best Practices**:
- Proper encapsulation
- No workspace pollution
- Explicit path management

## Customization

### Excluding Directories

```matlab
% Exclude specific directories from analysis
results = run_quality_checks('.', ...
    'ExcludeDirs', {'.git', 'slprj', 'codegen', 'build', 'external', 'third_party'});
```

### Custom Output Formats

Supported output formats:
- `.csv` - Comma-separated values
- `.xlsx` - Excel spreadsheet
- `.json` - JSON format
- `.md` - Markdown table

```matlab
% Export to different formats
run_quality_checks('.', 'OutputFile', 'report.csv');
run_quality_checks('.', 'OutputFile', 'report.xlsx');
run_quality_checks('.', 'OutputFile', 'report.json');
run_quality_checks('.', 'OutputFile', 'report.md');
```

### Strict Mode

In strict mode, any quality issues cause the check to fail:

```matlab
results = run_quality_checks('.', 'StrictMode', true);
if ~results.passed
    error('Quality checks failed - fix all issues before proceeding');
end
```

## Using in Your Project

### Method 1: Template Integration

When starting a new MATLAB project:

1. Copy `matlab_utilities/` to your project root
2. Add to your project's `startup.m`:
   ```matlab
   % Add quality utilities to path
   addpath('matlab_utilities/quality');
   addpath('matlab_utilities/testing');
   ```
3. Set up pre-commit hooks (see above)
4. Add to your CI/CD pipeline (see above)

### Method 2: Shared Utilities

For organizations with multiple MATLAB projects:

1. Create a separate `matlab-utilities` repository
2. Add as git submodule to each project:
   ```bash
   git submodule add <repo-url> matlab_utilities
   ```
3. Update across all projects:
   ```bash
   git submodule update --remote --merge
   ```

### Method 3: Project-Specific Customization

Copy the utilities and customize for your project:

```matlab
% custom_quality_checks.m
function results = custom_quality_checks()
    % Add project-specific quality rules
    results = run_quality_checks('.', ...
        'ExcludeDirs', {'legacy', 'deprecated', 'third_party'}, ...
        'StrictMode', true, ...
        'OutputFile', 'quality_report.csv');

    % Add custom checks here
    % ...
end
```

## Requirements

### MATLAB Tools
- MATLAB R2019b or later (for `exportCodeIssues.m` and `run_quality_checks.m`)
- Code Analyzer (mlint/checkcode) - included with MATLAB

### Python Tools
- Python 3.7 or later (3.11+ recommended for CI/CD)
- No additional dependencies required for `matlab_quality_check.py`

## License Considerations

**Important**: The MATLAB-based tools (`*.m` files) require a valid MATLAB license to run.

For CI/CD environments:
- Use the **Python static analyzer** if you don't have a CI/CD-compatible MATLAB license
- The Python tools provide quality checking **without requiring MATLAB**
- For local development, use MATLAB tools with your standard development license

## Troubleshooting

### "MATLAB not found" in CI/CD

**Solution**: Use the Python static analyzer instead:
```bash
python matlab_utilities/scripts/matlab_quality_check.py
```

### Pre-commit hooks not running

**Solution**: Ensure hooks are executable:
```bash
chmod +x .git/hooks/pre-commit
chmod +x .git/hooks/pre-push
```

### Too many false positives

**Solution**: Customize the exclusions or use non-strict mode:
```matlab
results = run_quality_checks('.', 'StrictMode', false);
```

## Examples

See the parent project for complete examples of integration.

## Contributing

To improve these utilities:
1. Make changes to your local copy
2. Test thoroughly with your project
3. Submit improvements back to the main utilities repository

## Support

For issues or questions:
- Check the documentation in each tool's header comments
- Review the parent project's quality control documentation
- Consult MATLAB's checkcode documentation

---

**Version**: 1.0.0
**Last Updated**: 2025-11-17
**Compatibility**: MATLAB R2019b+, Python 3.7+
