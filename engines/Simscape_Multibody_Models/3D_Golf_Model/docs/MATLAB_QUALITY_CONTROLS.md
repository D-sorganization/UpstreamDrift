# MATLAB Quality Control System

This document describes the comprehensive MATLAB code quality control system implemented for the Golf Model project, following the requirements specified in `.cursorrules.md`.

## Overview

The MATLAB quality control system provides automated code quality checks, linting, testing, and integration with the project's overall quality control framework. It ensures that all MATLAB code follows the project's coding standards and best practices.

## Components

### 1. MATLAB Quality Configuration (`matlab/matlab_quality_config.m`)

**Purpose**: Core quality checking function that runs comprehensive analysis on MATLAB code.

**Features**:
- **mlint Integration**: Runs MATLAB's built-in linting tool on all `.m` files
- **Function Structure Validation**: Checks for required docstrings and arguments validation blocks
- **Comprehensive File Scanning**: Analyzes all 2,838+ MATLAB files in the project
- **Detailed Reporting**: Provides structured output with specific issue details

**Usage**:
```matlab
% Run quality checks
results = matlab_quality_config();

% Check results
if results.passed
    fprintf('All checks passed: %d files analyzed\n', results.total_files);
else
    fprintf('Issues found: %d\n', length(results.issues));
    for i = 1:length(results.issues)
        fprintf('  %d. %s\n', i, results.issues{i});
    end
end
```

### 2. Python MATLAB Quality Checker (`scripts/matlab_quality_check.py`)

**Purpose**: Command-line interface for running MATLAB quality checks without requiring MATLAB to be running.

**Features**:
- **Cross-Platform**: Works on any system with Python
- **Static Analysis**: Performs quality checks without running MATLAB
- **Multiple Output Formats**: JSON and human-readable text output
- **Integration Ready**: Designed to work with pre-commit hooks and CI/CD

**Usage**:
```bash
# Basic quality check
python scripts/matlab_quality_check.py

# JSON output for automation
python scripts/matlab_quality_check.py --output-format json

# Strict mode with detailed analysis
python scripts/matlab_quality_check.py --strict
```

**Static Analysis Features**:
- Function docstring validation
- Arguments validation block checking
- Banned pattern detection (TODO, FIXME, HACK, XXX)
- Magic number identification
- Template placeholder detection

### 3. Enhanced Test Runner (`matlab/run_matlab_tests.m`)

**Purpose**: Comprehensive test execution with quality integration.

**Features**:
- **Quality-First Testing**: Runs quality checks before executing tests
- **Multiple Test Frameworks**: Supports both custom test functions and MATLAB Unit Test Framework
- **Detailed Reporting**: Comprehensive test results with pass/fail statistics
- **Results Persistence**: Saves test results to JSON files for analysis
- **Error Handling**: Robust error handling and reporting

**Usage**:
```matlab
% Run complete test suite
results = run_matlab_tests();

% Check test status
if contains(results.summary, 'PASSED')
    fprintf('All tests passed!\n');
else
    fprintf('Tests failed: %d errors\n', length(results.errors));
end
```

### 4. Pre-commit Integration (`.pre-commit-config.yaml`)

**Purpose**: Automated quality checks before each commit.

**Features**:
- **MATLAB Quality Hook**: Runs MATLAB quality checks on every commit
- **Python Quality Hook**: Runs Python quality checks
- **Standard Tools**: Integrates Black, Ruff, MyPy for Python
- **File Exclusions**: Properly excludes MATLAB and archive directories from Python checks

## Quality Standards

### Required Function Structure

All MATLAB functions must follow this structure (from `.cursorrules.md`):

```matlab
function results = process_swing_data(input_file, options)
    % PROCESS_SWING_DATA Analyze golf swing with reproducibility
    %
    % Inputs:
    %   input_file - Path to data file (must exist)
    %   options    - Struct with fields:
    %                .seed (default: 42) - RNG seed
    %                .tolerance (default: 1e-6) - Numerical tolerance
    %                .max_iter (default: 1000) - Max iterations
    %
    % Outputs:
    %   results    - Struct with fields:
    %                .data - Processed data array
    %                .metadata - Processing metadata struct
    %                .timestamp - ISO 8601 timestamp

    arguments
        input_file (1,1) string {mustBeFile}
        options.seed (1,1) double {mustBePositive} = 42
        options.tolerance (1,1) double {mustBePositive} = 1e-6
        options.max_iter (1,1) double {mustBeInteger, mustBePositive} = 1000
    end

    % Set reproducibility
    rng(options.seed, 'twister');

    % Validate inputs
    assert(exist(input_file, 'file') == 2, ...
           'PROCESS_SWING_DATA:FileNotFound', ...
           'Input file not found: %s', input_file);

    % Store metadata
    results.metadata.matlab_version = version;
    results.metadata.timestamp = datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss');
    results.metadata.git_sha = get_git_sha();
    results.metadata.input_file = char(input_file);
    results.metadata.options = options;

    % Process with comprehensive error handling
    try
        raw_data = readmatrix(input_file);

        % Validate data
        if any(isnan(raw_data(:)))
            warning('PROCESS_SWING_DATA:NaNDetected', ...
                    'NaN values found in input data');
        end

        % Process...
        results.data = raw_data;  % Actual processing here

    catch ME
        fprintf('Error in %s at line %d: %s\n', ...
                ME.stack(1).name, ME.stack(1).line, ME.message);
        results.metadata.error = ME.message;
        rethrow(ME);
    end
end
```

### Banned Patterns

The following patterns are automatically flagged:
- `TODO` - Incomplete implementation markers
- `FIXME` - Known issues that need fixing
- `HACK` - Temporary workarounds
- `XXX` - Critical issues
- `<.*>` - Angle bracket placeholders
- `{{.*}}` - Template placeholders
- Magic numbers (3.14, 1.57, 0.785, etc.)

### Required Elements

1. **Function Docstrings**: Every function must have a comprehensive docstring
2. **Arguments Validation**: Use the `arguments` block for input validation
3. **Error Handling**: Comprehensive try-catch blocks with meaningful error messages
4. **Metadata**: Include MATLAB version, timestamps, and git information
5. **Reproducibility**: Set random number generator seeds for consistent results

## Integration with Development Workflow

### Pre-commit Hooks

Quality checks run automatically before each commit:

```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks manually
pre-commit run --all-files

# Run specific hooks
pre-commit run matlab-quality-check
pre-commit run python-quality-check
```

### Continuous Integration

The quality system integrates with CI/CD pipelines:

```yaml
# Example GitHub Actions step
- name: Run MATLAB Quality Checks
  run: python scripts/matlab_quality_check.py --output-format json
```

### IDE Integration

**VS Code/Cursor with MATLAB Extension**:
- Install the MATLAB extension for VS Code
- Configure the extension to use your MATLAB installation
- Quality checks can be run directly from the command palette

**MATLAB Desktop**:
- Run quality checks directly in MATLAB
- Use the test runner for comprehensive testing
- View detailed quality reports

## Running Quality Checks

### From Command Line

```bash
# Run MATLAB quality checks
python scripts/matlab_quality_check.py

# Run Python quality checks
python scripts/quality_check.py

# Run all quality checks
python scripts/quality_check.py && python scripts/matlab_quality_check.py
```

### From MATLAB

```matlab
% Run quality configuration
results = matlab_quality_config();

% Run test suite
test_results = run_matlab_tests();

% Check specific files
issues = mlint('path/to/file.m');
```

### From Pre-commit

```bash
# Quality checks run automatically on commit
git add .
git commit -m "feat: add new MATLAB function"

# Or run manually
pre-commit run --all-files
```

## Output and Reporting

### Quality Check Results

```json
{
  "timestamp": "2025-08-10T12:00:00",
  "total_files": 2838,
  "issues": [
    "function_name.m (line 15): Missing function docstring",
    "another_file.m (line 42): Missing arguments validation block"
  ],
  "passed": false,
  "summary": "❌ MATLAB quality checks FAILED (2838 files checked, 2 issues found)",
  "checks": {
    "matlab": {
      "success": true,
      "method": "static_analysis",
      "total_files": 2838,
      "issues": [...],
      "passed": false
    }
  }
}
```

### Test Results

```json
{
  "timestamp": "2025-08-10T12:00:00",
  "tests_run": 5,
  "tests_passed": 4,
  "tests_failed": 1,
  "summary": "❌ MATLAB tests FAILED (4 passed, 1 failed)",
  "framework_tests": {
    "run": 3,
    "passed": 2,
    "failed": 1
  }
}
```

## Troubleshooting

### Common Issues

1. **MATLAB Not Found**: Ensure MATLAB is installed and accessible from command line
2. **Quality Checks Failing**: Review the specific issues reported and fix them
3. **Test Failures**: Check test output for specific error messages
4. **Pre-commit Hooks Failing**: Run quality checks manually to identify issues

### Performance Considerations

- **Large Codebases**: With 2,838+ MATLAB files, quality checks may take several minutes
- **Memory Usage**: Large files may require increased memory allocation
- **Timeout Settings**: Adjust timeout values for very large projects

### Customization

The quality system can be customized by modifying:
- `matlab/matlab_quality_config.m` - MATLAB quality rules
- `scripts/matlab_quality_check.py` - Python quality checker
- `.pre-commit-config.yaml` - Pre-commit hook configuration

## Future Enhancements

Planned improvements include:
- **Parallel Processing**: Multi-threaded quality checks for large codebases
- **Incremental Analysis**: Only check changed files for faster feedback
- **IDE Integration**: Real-time quality feedback in MATLAB editor
- **Custom Rules**: User-defined quality rules and patterns
- **Performance Metrics**: Code complexity and performance analysis

## Support

For issues or questions about the MATLAB quality control system:
1. Check the project's `.cursorrules.md` for requirements
2. Review the quality check output for specific issues
3. Consult MATLAB documentation for coding standards
4. Review the test output for debugging information

---

*This quality control system ensures that all MATLAB code in the Golf Model project meets the highest standards of quality, maintainability, and compliance with project requirements.*
