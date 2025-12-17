# MATLAB Compliance Checks

**Last Updated:** 2025-11-29

This document describes the MATLAB-specific compliance checks and quality standards for the 2D_Golf_Model repository and related MATLAB projects.

## Overview

MATLAB compliance ensures:
- Code quality through static analysis
- Reproducible results via fixed random seeds
- Comprehensive testing with MATLAB Unit Test framework
- Documentation standards for all functions
- No magic numbers or approximations

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Code Quality Checks](#code-quality-checks)
3. [Unit Testing](#unit-testing)
4. [Documentation Standards](#documentation-standards)
5. [Reproducibility Requirements](#reproducibility-requirements)
6. [CI Integration](#ci-integration)
7. [Local Development](#local-development)

---

## Quick Start

### Prerequisites

- MATLAB R2023a or later
- Required toolboxes: Simulink (if using .slx files)
- Git for version control

### Running All Checks Locally

```matlab
% In MATLAB command window
cd /path/to/2D_Golf_Model

% Add all paths
addpath(genpath('matlab'));

% Run quality checks
run_matlab_quality_checks();

% Run unit tests
results = runtests('matlab/tests');

% Display results
disp(results);
```

### Quick Validation Script

Create `run_matlab_quality_checks.m` in the root:

```matlab
function run_matlab_quality_checks()
    % Run all MATLAB quality checks
    fprintf('🔍 Running MATLAB Quality Checks...\n\n');
    
    % 1. Check for magic numbers
    check_magic_numbers();
    
    % 2. Run code analyzer
    run_code_analyzer();
    
    % 3. Check reproducibility
    check_reproducibility();
    
    % 4. Validate documentation
    check_documentation();
    
    fprintf('\n✅ All quality checks completed!\n');
end

function check_magic_numbers()
    fprintf('1️⃣ Checking for magic numbers...\n');
    
    % Common magic numbers to avoid
    magicNumbers = {'3.14', '9.8', '6.67', '2.71'};
    
    files = [dir('matlab/**/*.m'); dir('matlab_optimized/**/*.m')];
    found = false;
    
    for i = 1:length(files)
        filePath = fullfile(files(i).folder, files(i).name);
        
        % Skip archived and test files
        if contains(filePath, 'Archive') || contains(filePath, 'Backup') || contains(filePath, 'test')
            continue;
        end
        
        content = fileread(filePath);
        
        for j = 1:length(magicNumbers)
            if contains(content, magicNumbers{j})
                fprintf('   ⚠️ Found %s in %s\n', magicNumbers{j}, filePath);
                found = true;
            end
        end
    end
    
    if ~found
        fprintf('   ✅ No magic numbers found\n');
    end
end

function run_code_analyzer()
    fprintf('2️⃣ Running code analyzer (checkcode)...\n');
    
    files = [dir('matlab/**/*.m'); dir('matlab_optimized/**/*.m')];
    hasIssues = false;
    
    for i = 1:length(files)
        filePath = fullfile(files(i).folder, files(i).name);
        
        % Skip archived and test files
        if contains(filePath, 'Archive') || contains(filePath, 'Backup') || contains(filePath, 'test')
            continue;
        end
        
        issues = checkcode(filePath, '-id');
        
        if ~isempty(issues)
            fprintf('   ⚠️ Issues in %s:\n', filePath);
            for j = 1:length(issues)
                fprintf('      %s\n', issues(j).message);
            end
            hasIssues = true;
        end
    end
    
    if ~hasIssues
        fprintf('   ✅ All files pass code analyzer\n');
    end
end

function check_reproducibility()
    fprintf('3️⃣ Checking reproducibility...\n');
    
    files = [dir('matlab/**/*.m'); dir('matlab_optimized/**/*.m')];
    warnings = 0;
    
    for i = 1:length(files)
        filePath = fullfile(files(i).folder, files(i).name);
        
        % Skip archived files
        if contains(filePath, 'Archive') || contains(filePath, 'Backup')
            continue;
        end
        
        content = fileread(filePath);
        
        % Check for random functions without rng
        if (contains(content, 'rand') || contains(content, 'randn') || contains(content, 'randi'))
            if ~contains(content, 'rng')
                fprintf('   ⚠️ %s uses randomness without rng seed\n', filePath);
                warnings = warnings + 1;
            end
        end
    end
    
    if warnings == 0
        fprintf('   ✅ All files with randomness have seeds\n');
    end
end

function check_documentation()
    fprintf('4️⃣ Checking function documentation...\n');
    
    files = [dir('matlab/**/*.m'); dir('matlab_optimized/**/*.m')];
    missingDocs = 0;
    
    for i = 1:length(files)
        filePath = fullfile(files(i).folder, files(i).name);
        
        % Skip archived and test files
        if contains(filePath, 'Archive') || contains(filePath, 'Backup') || contains(filePath, 'test')
            continue;
        end
        
        content = fileread(filePath);
        
        % Check if it's a function
        if startsWith(strtrim(content), 'function')
            % Check for basic documentation
            if ~contains(content, '%') || length(strfind(content, '%')) < 3
                fprintf('   ⚠️ %s may be missing documentation\n', filePath);
                missingDocs = missingDocs + 1;
            end
        end
    end
    
    if missingDocs == 0
        fprintf('   ✅ All functions have documentation\n');
    else
        fprintf('   ⚠️ %d files may need better documentation\n', missingDocs);
    end
end
```

---

## Code Quality Checks

### Static Analysis with checkcode

MATLAB's built-in `checkcode` function performs static code analysis:

```matlab
% Check a single file
issues = checkcode('myfile.m', '-id');
if ~isempty(issues)
    disp(issues);
end

% Check all files in a directory
files = dir('matlab/**/*.m');
for i = 1:length(files)
    filePath = fullfile(files(i).folder, files(i).name);
    issues = checkcode(filePath, '-id');
    if ~isempty(issues)
        fprintf('Issues in %s:\n', filePath);
        disp(issues);
    end
end
```

### Common Issues Detected

1. **Unused variables** - Variables assigned but never used
2. **Uninitialized variables** - Variables used before assignment
3. **Function signature mismatches** - Input/output count errors
4. **Deprecated functions** - Use of outdated MATLAB functions
5. **Performance warnings** - Inefficient code patterns

### Command Line Usage

```bash
# From command line (if MATLAB is in PATH)
matlab -batch "checkcode('myfile.m')"

# Check all files
matlab -batch "files = dir('matlab/**/*.m'); for i=1:length(files), checkcode(fullfile(files(i).folder, files(i).name)); end"
```

---

## Unit Testing

### Test File Structure

All tests go in `matlab/tests/` directory:

```
matlab/
├── tests/
│   ├── test_example.m           # Example test
│   ├── test_kinematics.m        # Kinematics tests
│   ├── test_dynamics.m          # Dynamics tests
│   └── test_plotting.m          # Plotting tests
├── Scripts/                     # Main scripts
└── run_all.m                    # Main entry point
```

### Writing Unit Tests

**Basic test structure:**

```matlab
% matlab/tests/test_kinematics.m
function tests = test_kinematics
    % Test suite for kinematics calculations
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    % Setup run once before all tests
    addpath(genpath('matlab'));
    
    % Store data in testCase for use in tests
    testCase.TestData.tolerance = 1e-6;
end

function setup(testCase)
    % Setup run before each test
    rng(42);  % Fixed seed for reproducibility
end

function test_position_calculation(testCase)
    % Test basic position calculation
    time = 2.0;  % [s]
    velocity = 5.0;  % [m/s]
    
    expected = 10.0;  % [m]
    actual = calculate_position(time, velocity);
    
    verifyEqual(testCase, actual, expected, ...
        'AbsTol', testCase.TestData.tolerance, ...
        'Position calculation failed');
end

function test_velocity_calculation(testCase)
    % Test velocity calculation
    position1 = 0.0;  % [m]
    position2 = 10.0;  % [m]
    deltaTime = 2.0;  % [s]
    
    expected = 5.0;  % [m/s]
    actual = calculate_velocity(position1, position2, deltaTime);
    
    verifyEqual(testCase, actual, expected, ...
        'AbsTol', testCase.TestData.tolerance, ...
        'Velocity calculation failed');
end

function test_zero_time_error(testCase)
    % Test that zero time throws error
    position1 = 0.0;
    position2 = 10.0;
    deltaTime = 0.0;
    
    verifyError(testCase, ...
        @() calculate_velocity(position1, position2, deltaTime), ...
        'MATLAB:divideByZero', ...
        'Should error on zero time');
end

function teardown(testCase)
    % Cleanup after each test
    close all;
end
```

### Running Tests

**From MATLAB command window:**

```matlab
% Run all tests in matlab/tests/
results = runtests('matlab/tests');

% Run specific test file
results = runtests('matlab/tests/test_kinematics.m');

% Run with code coverage
import matlab.unittest.TestRunner
import matlab.unittest.plugins.CodeCoveragePlugin

runner = TestRunner.withTextOutput;
runner.addPlugin(CodeCoveragePlugin.forFolder('matlab'));

results = runner.run(testsuite('matlab/tests'));
```

**From command line:**

```bash
# Run all tests
matlab -batch "results = runtests('matlab/tests'); disp(results);"

# Run with exit code based on results
matlab -batch "results = runtests('matlab/tests'); exit(any([results.Failed]));"
```

### Test Assertions

Common verification methods:

```matlab
% Equality checks
verifyEqual(testCase, actual, expected)
verifyEqual(testCase, actual, expected, 'AbsTol', 1e-6)
verifyEqual(testCase, actual, expected, 'RelTol', 0.01)

% Comparison checks
verifyGreaterThan(testCase, actual, lowerBound)
verifyLessThan(testCase, actual, upperBound)

% Error/warning checks
verifyError(testCase, @() functionCall(), 'ExpectedErrorID')
verifyWarning(testCase, @() functionCall(), 'ExpectedWarningID')

% Boolean checks
verifyTrue(testCase, condition)
verifyFalse(testCase, condition)

% Size/class checks
verifySize(testCase, actual, [2, 3])
verifyClass(testCase, actual, 'double')
```

---

## Documentation Standards

### Function Documentation Template

All public functions must include comprehensive documentation:

```matlab
function [output1, output2] = example_function(input1, input2, options)
% EXAMPLE_FUNCTION Brief one-line description
%
% Detailed description explaining what the function does, its purpose,
% and any important notes about its behavior.
%
% Syntax:
%   output1 = example_function(input1, input2)
%   [output1, output2] = example_function(input1, input2, options)
%
% Inputs:
%   input1 - First input description [units] (type, size)
%            Additional details if needed
%   input2 - Second input description [units] (type, size)
%   options - (Optional) Structure with fields:
%            .field1 - Description [units] (default: value)
%            .field2 - Description [units] (default: value)
%
% Outputs:
%   output1 - First output description [units] (type, size)
%   output2 - Second output description [units] (type, size)
%
% Examples:
%   % Example 1: Basic usage
%   result = example_function(5.0, 10.0);
%
%   % Example 2: With options
%   opts.field1 = 42;
%   [res1, res2] = example_function(5.0, 10.0, opts);
%
% Notes:
%   - Important note 1
%   - Important note 2
%   - Known limitations or assumptions
%
% References:
%   [1] Author, "Title", Journal, Year
%   [2] Source of equations or algorithms
%
% See also: related_function1, related_function2
%
% Author: Name
% Date: YYYY-MM-DD
% Version: 1.0

    % Validate inputs
    arguments
        input1 (1,1) double {mustBeFinite, mustBePositive}
        input2 (1,1) double {mustBeFinite}
        options.field1 (1,1) double = 0.0
        options.field2 (1,1) logical = true
    end
    
    % Implementation
    output1 = input1 + input2;
    output2 = input1 * input2;
    
end
```

### Physical Constants Documentation

All physical constants must include:
1. **Units** in square brackets `[m/s^2]`
2. **Source** in comment (reference)
3. **Date** of reference if applicable

```matlab
% Physical constants with proper documentation
G_GRAVITY = 9.80665;  % [m/s^2] Standard gravity (NIST SP 330, 2019)
SPEED_OF_LIGHT = 299792458;  % [m/s] Exact by definition (BIPM SI, 2019)
BOLTZMANN = 1.380649e-23;  % [J/K] Boltzmann constant (CODATA 2018)

% Mathematical constants (use built-ins)
circumference = 2 * pi * radius;  % Use built-in pi
euler_number = exp(1);  % Use exp(1) instead of 2.71828...
```

### Variable Naming with Units

Include units in comments for all physical quantities:

```matlab
% Good - Clear units
time = 2.5;  % [s] Time elapsed
velocity = 10.0;  % [m/s] Initial velocity
position = velocity * time;  % [m] Final position

% Better - Units in variable name for clarity
time_s = 2.5;  % [s]
velocity_m_per_s = 10.0;  % [m/s]
position_m = velocity_m_per_s * time_s;  % [m]
```

---

## Reproducibility Requirements

### Random Number Generation

**All randomness must use fixed seeds:**

```matlab
% At the beginning of any script using randomness
rng(42, 'twister');  % Fixed seed for reproducibility

% Document why randomness is needed
% Example: Monte Carlo simulation
rng(42);
samples = randn(1000, 1);

% For different simulations, use different documented seeds
rng(100);  % Seed for sensitivity analysis
sensitivity_samples = randn(500, 1);
```

### Metadata Collection

Record run metadata for reproducibility:

```matlab
% In run_all.m or main scripts
meta.date = datestr(datetime('now'));
meta.matlab_version = version;
meta.platform = computer;
meta.seed = 42;
meta.commit_sha = get_git_commit();  % If available
meta.description = 'Baseline simulation run';

% Save metadata with results
save(fullfile(outdir, 'metadata.mat'), 'meta');
```

### Avoiding Non-Reproducible Patterns

```matlab
% ❌ Bad - Non-reproducible
x = rand(100, 1);  % No seed set

% ❌ Bad - Time-dependent
filename = sprintf('output_%s.mat', datestr(now));

% ❌ Bad - System-dependent paths
load('C:\Users\John\data.mat');

% ✅ Good - Reproducible
rng(42);
x = rand(100, 1);

% ✅ Good - Consistent naming
filename = 'output_baseline.mat';

% ✅ Good - Relative paths
load(fullfile('data', 'input.mat'));
```

---

## CI Integration

### GitHub Actions Workflow

The MATLAB CI workflow is defined in `.github/workflows/ci.yml`:

```yaml
matlab-analysis:
  runs-on: ubuntu-latest
  # Enable when MATLAB license is available
  if: false  # Set to true when ready
  
  steps:
    - uses: actions/checkout@v4
    
    - name: Set up MATLAB
      uses: matlab-actions/setup-matlab@v2
      with:
        release: R2023b
        products: MATLAB Simulink
    
    - name: Run quality checks
      uses: matlab-actions/run-command@v2
      with:
        command: run_matlab_quality_checks()
    
    - name: Run tests
      uses: matlab-actions/run-tests@v2
      with:
        source-folder: matlab
        test-results-junit: test-results/matlab-results.xml
```

### Enabling MATLAB CI

To enable MATLAB CI checks:

1. **Set up MATLAB license**
   - Add license file to repository secrets
   - Or configure network license server

2. **Update workflow**
   - Change `if: false` to `if: true` in `.github/workflows/ci.yml`

3. **Configure required toolboxes**
   - Add toolbox names to `products` list in workflow

4. **Test locally first**
   - Ensure all checks pass locally before enabling CI

### Self-Hosted Runners

For repositories with licensed MATLAB installations:

```yaml
matlab-analysis:
  runs-on: self-hosted  # Use self-hosted runner
  
  steps:
    - uses: actions/checkout@v4
    
    - name: Run MATLAB quality checks
      run: |
        matlab -batch "run_matlab_quality_checks()"
    
    - name: Run MATLAB tests
      run: |
        matlab -batch "results = runtests('matlab/tests'); exit(any([results.Failed]));"
```

---

## Local Development

### Pre-Commit Checks

Create `matlab/.pre-commit` script:

```bash
#!/bin/bash
# Pre-commit hook for MATLAB files

echo "🔍 Running MATLAB quality checks..."

# Check for magic numbers
if find matlab -name "*.m" -type f -exec grep -H '[^0-9a-zA-Z_]3\.14[0-9]*\|9\.8[0-9]*' {} \; 2>/dev/null | grep -q .; then
    echo "❌ Magic numbers found! Use named constants."
    exit 1
fi

# Run MATLAB checks (if MATLAB available)
if command -v matlab &> /dev/null; then
    matlab -batch "run_matlab_quality_checks()" || exit 1
fi

echo "✅ MATLAB quality checks passed!"
```

### IDE Integration

**VS Code with MATLAB extension:**

1. Install MATLAB extension
2. Configure linter settings in `.vscode/settings.json`:

```json
{
    "matlab.linterSeverityLevel": "warning",
    "matlab.linterEncoding": "windows1252"
}
```

**MATLAB Editor settings:**

1. Go to Preferences → Code Analyzer
2. Enable all warnings
3. Enable auto-save
4. Set tab width to 4 spaces

---

## Common Issues and Solutions

### Issue: Magic Numbers

**Problem:**
```matlab
velocity = position / 3.14159;  % ❌
```

**Solution:**
```matlab
velocity = position / pi;  % ✅ Use built-in constant
```

### Issue: Missing Documentation

**Problem:**
```matlab
function y = calc(x)
    y = x^2 + 2*x + 1;
end
```

**Solution:**
```matlab
function y = calc(x)
% CALC Calculate quadratic function
%
% Inputs:
%   x - Input value [-] (scalar)
% Outputs:
%   y - Result [-] (scalar)
%
% Formula: y = x^2 + 2x + 1

    y = x^2 + 2*x + 1;
end
```

### Issue: Non-Reproducible Random Numbers

**Problem:**
```matlab
data = randn(100, 1);  % ❌ No seed
```

**Solution:**
```matlab
rng(42);  % ✅ Fixed seed
data = randn(100, 1);
```

### Issue: Absolute Paths

**Problem:**
```matlab
load('C:\Users\Me\data.mat');  % ❌
```

**Solution:**
```matlab
load(fullfile('data', 'input.mat'));  % ✅
```

---

## Checklist for MATLAB Compliance

Before merging MATLAB code, verify:

- [ ] All functions have comprehensive documentation
- [ ] No magic numbers (use named constants with sources)
- [ ] No approximations (use exact values)
- [ ] Random number generators use fixed seeds
- [ ] Unit tests cover main functionality
- [ ] All tests pass locally
- [ ] Code passes checkcode analysis
- [ ] Physical constants have units and sources
- [ ] Relative paths used (no absolute paths)
- [ ] No system-specific dependencies
- [ ] Metadata saved with results

---

## References

- [MATLAB Unit Testing Framework](https://www.mathworks.com/help/matlab/matlab-unit-test-framework.html)
- [MATLAB Code Analyzer](https://www.mathworks.com/help/matlab/ref/checkcode.html)
- [MATLAB Best Practices](https://www.mathworks.com/matlabcentral/fileexchange/46056-matlab-style-guidelines-2-0)
- [GitHub Actions for MATLAB](https://github.com/matlab-actions)

---

**Document Maintenance:**
- Review when MATLAB version updates
- Update examples as standards evolve
- Sync with UNIFIED_CI_APPROACH.md

**Last Reviewed:** 2025-11-29  
**Next Review:** 2026-02-28
