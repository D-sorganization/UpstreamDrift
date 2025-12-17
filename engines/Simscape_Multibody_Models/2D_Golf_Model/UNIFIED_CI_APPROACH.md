# Unified CI/CD Approach for D-sorganization Repositories

**Last Updated:** 2025-11-29

This document defines the **unified CI/CD standards** for all 15 repositories in the D-sorganization, covering Python, MATLAB, JavaScript/TypeScript, and Arduino projects.

## Table of Contents

1. [Overview](#overview)
2. [Key Principles](#key-principles)
3. [Technology Stack Support](#technology-stack-support)
4. [Python CI Workflow](#python-ci-workflow)
5. [MATLAB CI Workflow](#matlab-ci-workflow)
6. [JavaScript/TypeScript CI Workflow](#javascripttypescript-ci-workflow)
7. [Shell Script Validation](#shell-script-validation)
8. [Common Patterns](#common-patterns)
9. [Security Scanning](#security-scanning)
10. [Documentation Checks](#documentation-checks)

---

## Overview

The unified CI/CD approach ensures:
- **Consistency** across all repositories
- **Reproducibility** via pinned versions
- **Comprehensive checks** (linting, testing, security)
- **Fast feedback** with fail-fast strategies
- **Clear reporting** with proper exit codes

### Repositories Using This Approach

1. Data_Processor (Python)
2. Gasification_Model (Python)
3. MuJoCo_Golf_Swing_Model (Python)
4. MLProjects (Python)
5. Audio_Processor (MATLAB)
6. Golf_Model (MATLAB)
7. 2D_Golf_Model (MATLAB + Python)
8. Robotics (MATLAB)
9. Unit_Converter (JavaScript/TypeScript)
10. Video_Processor (JavaScript/TypeScript)
11. Repository_Management (Shell scripts)
12. Project_Template (Multi-language)
13-15. Additional repositories as needed

---

## Key Principles

### 1. Pinned Versions

**All tool versions must be explicitly specified** for reproducibility:

```yaml
# Python tools
pip install ruff==0.5.0 mypy==1.10.0 black==24.4.2 pytest==8.3.3 pytest-cov==6.0.0

# Security tools
pip install bandit==1.7.7 safety==3.0.1

# Documentation tools
pip install pydocstyle==6.3.0
```

**Action versions:**
```yaml
- uses: actions/checkout@v4
- uses: actions/setup-python@v5
- uses: actions/setup-node@v4
- uses: matlab-actions/setup-matlab@v2
```

### 2. Comprehensive Detection

**Automatically detect source directories:**

```bash
# Python source detection
if [ -d "python/src" ]; then
    SOURCE_DIR="python/src"
elif [ -d "python" ]; then
    SOURCE_DIR="python"
elif [ -d "src" ]; then
    SOURCE_DIR="src"
else
    echo "No Python source directory found"
    exit 1
fi
```

### 3. Proper Exit Codes

**Preserve failure codes for GitHub Actions:**

```bash
# ✅ Good - Preserves exit code
ruff check python/ || exit 1
mypy python/ || exit 1

# ❌ Bad - Masks failures
ruff check python/
mypy python/

# ✅ Alternative - Use continue-on-error
- name: Run linter
  continue-on-error: true
  run: ruff check python/
```

### 4. Conditional Uploads

**Upload coverage only when available:**

```yaml
- name: Check for coverage file
  id: coverage-check
  run: |
    if find . -name 'coverage.xml' -type f | grep -q .; then
      echo "exists=true" >> $GITHUB_OUTPUT
    else
      echo "exists=false" >> $GITHUB_OUTPUT
    fi

- name: Upload coverage to Codecov
  if: steps.coverage-check.outputs.exists == 'true'
  uses: codecov/codecov-action@v4
  with:
    files: "**/coverage.xml"
    fail_ci_if_error: false
```

### 5. Security Checks

**Required security scanning:**
- Dependency scanning (Bandit, Safety)
- Secret detection (TruffleHog)
- Code scanning (CodeQL for public repos)
- Vulnerability alerts (Dependabot)

### 6. Documentation Checks

**Verify documentation quality:**
- Markdown linting
- Link validation
- Docstring coverage (pydocstyle)
- README.md presence

### 7. Replicant Branch Support

**Include `claude/*_Replicants` branches in workflow triggers when they exist:**

```yaml
on:
  push:
    branches: [main, master, claude/2D_Golf_Model_Replicants]
  pull_request:
    branches: [main, master, claude/2D_Golf_Model_Replicants]
```

### 8. Quality Check Scripts

**Support both standard locations:**

```bash
if [ -f scripts/quality-check.py ]; then
  python scripts/quality-check.py
elif [ -f scripts/quality_check.py ]; then
  python scripts/quality_check.py
elif [ -f quality_check_script.py ]; then
  python quality_check_script.py
else
  echo "⚠️ Warning: Quality check script not found"
fi
```

### 9. Fail-Fast Strategy

**Always include `fail-fast: true` in matrix strategies:**

```yaml
strategy:
  fail-fast: true  # Stop all jobs if one fails
  matrix:
    python-version: ['3.10', '3.11', '3.12']
```

### 10. Cache Patterns

**Use comprehensive cache patterns:**

```yaml
- name: Cache pip dependencies
  uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/*requirements*.txt', '**/pyproject.toml', '**/setup.py') }}
    restore-keys: |
      ${{ runner.os }}-pip-
```

---

## Technology Stack Support

### Python Projects

**Required Tools:**
- Testing: pytest==8.3.3 with pytest-cov==6.0.0
- Linting: ruff==0.5.0 (replaces flake8, isort)
- Type checking: mypy==1.10.0 with strict mode
- Formatting: black==24.4.2
- Security: bandit==1.7.7, safety==3.0.1

**Minimum Requirements:**
- Python versions: 3.10, 3.11, 3.12
- Test coverage: 60% minimum (configurable)
- Type hints: Encouraged for public APIs

**Standard Exclusions:**
```toml
# ruff.toml
extend-exclude = [
    "Archive/",
    "legacy/",
    "old_code/",
    "experimental/",
    "*Python Version/",
]
```

### MATLAB Projects

**Required Tools:**
- Testing: MATLAB Unit Test framework
- Code quality: checkcode (built-in code analyzer)
- Style: mlint (code analyzer for best practices)

**Minimum Requirements:**
- MATLAB versions: R2023a or later
- Test coverage: Basic unit tests for core functions
- Code quality: Pass checkcode with standard rules

**Test Structure:**
```matlab
% matlab/tests/test_example.m
function tests = test_example
    tests = functiontests(localfunctions);
end

function test_basic_functionality(testCase)
    verifyEqual(testCase, 1+1, 2);
end
```

**Code Quality Check:**
```bash
# Check all MATLAB files for issues
find matlab -name "*.m" -type f -exec matlab -batch "checkcode -id {}" \;
```

### JavaScript/TypeScript Projects

**Required Tools:**
- Testing: Jest with coverage
- Linting: ESLint with TypeScript support
- Type checking: TypeScript strict mode
- Formatting: Prettier

**Minimum Requirements:**
- Node versions: 18.x, 20.x
- Test coverage: 70% minimum
- TypeScript: strict mode enabled

### Shell Scripts

**Required Tools:**
- Syntax check: `bash -n script.sh`
- Linting: shellcheck (if available)
- Testing: bats-core (optional)

---

## Python CI Workflow

### Complete Workflow Example

```yaml
name: Python CI

on:
  push:
    branches: [main, master]
    # Add replicant branches if they exist:
    # branches: [main, master, claude/RepositoryName_Replicants]
    paths:
      - 'python/**'
      - 'src/**'
      - 'tests/**'
      - '.github/workflows/python-ci.yml'
      - 'requirements*.txt'
      - 'pyproject.toml'
  pull_request:
    branches: [main, master]
    # Add replicant branches if they exist
    paths:
      - 'python/**'
      - 'src/**'
      - 'tests/**'

jobs:
  lint:
    name: Lint and Style Check
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for better diffs

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Cache pip dependencies
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/*requirements*.txt', '**/pyproject.toml') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          # Check both root and python/ subdirectory
          if [ -f python/requirements.txt ]; then pip install -r python/requirements.txt; fi
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
          if [ -f pyproject.toml ]; then pip install -e .; fi
          # Pin tool versions
          pip install ruff==0.5.0 mypy==1.10.0 black==24.4.2

      - name: Run quality check
        run: |
          # Support both standard locations
          if [ -f scripts/quality-check.py ]; then
            python scripts/quality-check.py
          elif [ -f scripts/quality_check.py ]; then
            python scripts/quality_check.py
          elif [ -f quality_check_script.py ]; then
            python quality_check_script.py
          else
            echo "⚠️ Warning: Quality check script not found"
          fi

      - name: Lint with ruff
        run: |
          # Use ruff check . when ruff.toml exists (respects config)
          if [ -f ruff.toml ]; then
            ruff check . || exit 1
          else
            # Fallback with explicit source directory
            if [ -d python ]; then
              ruff check python/ || exit 1
            else
              ruff check . || exit 1
            fi
          fi

      - name: Format check with black
        run: |
          # Check source directories explicitly
          if [ -d python/src ]; then
            black --check --diff python/src python/tests || exit 1
          elif [ -d python ]; then
            black --check --diff python || exit 1
          elif [ -d src ]; then
            black --check --diff src tests/ || exit 1
          else
            black --check --diff . || exit 1
          fi

      - name: Type check with mypy
        run: |
          # Use mypy with ignore-missing-imports
          if [ -d python ]; then
            mypy python/ --ignore-missing-imports || exit 1
          else
            mypy . --ignore-missing-imports || exit 1
          fi

  test:
    name: Test Suite
    runs-on: ubuntu-latest
    needs: lint  # Only run if lint passes
    strategy:
      fail-fast: true  # Always include
      matrix:
        python-version: ['3.10', '3.11', '3.12']

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          if [ -f python/requirements.txt ]; then pip install -r python/requirements.txt; fi
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
          if [ -f pyproject.toml ]; then pip install -e .; fi
          pip install pytest==8.3.3 pytest-cov==6.0.0

      - name: Run tests with coverage
        run: |
          # Detect source directory for accurate coverage
          if [ -d python/src ]; then
            COV_TARGET="python/src"
          elif [ -d python ]; then
            COV_TARGET="python"
          elif [ -d src ]; then
            COV_TARGET="src"
          else
            COV_TARGET="."
          fi

          # Run tests with coverage
          if [ -d python/tests ]; then
            cd python
            pytest tests/ --cov=src --cov-report=xml --cov-report=term-missing || exit 1
            cd ..
          elif [ -d tests ]; then
            pytest tests/ --cov="$COV_TARGET" --cov-report=xml --cov-report=term-missing || exit 1
          else
            echo "⚠️ No tests directory found"
            exit 1
          fi

      - name: Check for coverage file
        id: coverage-check
        run: |
          if find . -name 'coverage.xml' -type f | grep -q .; then
            echo "exists=true" >> $GITHUB_OUTPUT
          else
            echo "exists=false" >> $GITHUB_OUTPUT
          fi

      - name: Upload coverage to Codecov
        if: |
          steps.coverage-check.outputs.exists == 'true' &&
          matrix.python-version == '3.11' &&
          (github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository)
        uses: codecov/codecov-action@v4
        with:
          files: "**/coverage.xml"
          token: ${{ secrets.CODECOV_TOKEN }}
          fail_ci_if_error: false

  security:
    name: Security Scan
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Run Bandit security check
        continue-on-error: true
        run: |
          pip install bandit==1.7.7
          # Exclude test directories, target source code only
          if [ -d python/src ]; then
            bandit -r python/src --exclude 'python/tests,**/tests/**' -f json -o bandit-report.json
          elif [ -d python ]; then
            bandit -r python --exclude 'tests,**/tests/**' -f json -o bandit-report.json
          else
            bandit -r . --exclude 'tests,**/tests/**' -f json -o bandit-report.json
          fi

      - name: Upload Bandit results
        uses: actions/upload-artifact@v4
        with:
          name: bandit-report
          path: bandit-report.json
          if-no-files-found: ignore
```

### Python Best Practices

**1. Source Detection**

Always detect the source directory automatically:

```bash
# Comprehensive detection pattern
if [ -d "python/src" ]; then
    SOURCE_DIR="python/src"
    TEST_DIR="python/tests"
elif [ -d "python" ]; then
    SOURCE_DIR="python"
    TEST_DIR="tests"
elif [ -d "src" ]; then
    SOURCE_DIR="src"
    TEST_DIR="tests"
else
    echo "❌ No Python source directory found"
    exit 1
fi
```

**2. Exit Code Preservation**

```bash
# ✅ Good - Explicit exit on failure
ruff check python/ || exit 1
mypy python/ --ignore-missing-imports || exit 1

# ✅ Alternative - Use continue-on-error for warnings
- name: Check documentation
  continue-on-error: true
  run: pydocstyle src/

# ❌ Bad - Masks failures
ruff check python/
ruff check python/ || true
```

**3. Conditional Steps**

```yaml
# Only upload coverage once per run
- name: Upload coverage
  if: matrix.python-version == '3.11'
  uses: codecov/codecov-action@v4

# Skip uploads for external PRs (security)
if: |
  matrix.python-version == '3.11' &&
  (github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository)
```

---

## MATLAB CI Workflow

### Overview

MATLAB projects require:
1. **Code Quality Checks** - checkcode for static analysis
2. **Unit Tests** - MATLAB Unit Test framework
3. **Documentation** - Function documentation validation
4. **Reproducibility** - Fixed random seeds

### Prerequisites

**GitHub-hosted runners require:**
- `matlab-actions/setup-matlab@v2` action
- MATLAB license (via GitHub Secrets or license file)
- Supported MATLAB version (R2023a or later)

### Complete MATLAB Workflow

```yaml
name: MATLAB CI

on:
  push:
    branches: [main, master]
    # Add replicant branches if they exist
    paths:
      - 'matlab/**'
      - 'matlab_optimized/**'
      - '.github/workflows/matlab-ci.yml'
  pull_request:
    branches: [main, master]
    paths:
      - 'matlab/**'
      - 'matlab_optimized/**'

jobs:
  matlab-quality:
    name: MATLAB Code Quality
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up MATLAB
        uses: matlab-actions/setup-matlab@v2
        with:
          release: R2023b
          products: MATLAB Simulink  # Add required toolboxes

      - name: Check for magic numbers
        run: |
          echo "🔍 Checking for magic numbers in MATLAB files..."
          # Check for common magic numbers that should be constants
          if find matlab matlab_optimized -name "*.m" -type f -exec grep -H '[^0-9a-zA-Z_]3\.14[0-9]*\|9\.8[0-9]*\|6\.67[0-9]*' {} \; 2>/dev/null | grep -v "^Binary"; then
            echo "⚠️ WARNING: Possible magic numbers found (should be named constants with sources)"
            echo "Examples: 3.14→pi, 9.8→g (with source)"
          else
            echo "✅ No obvious magic numbers found"
          fi

      - name: Check for approximations
        run: |
          echo "🔍 Checking for approximations..."
          if find matlab matlab_optimized -name "*.m" -type f -exec grep -Hi "approximately\|approx[^a-z]\|roughly\|~[0-9]" {} \; 2>/dev/null | grep -v "^Binary"; then
            echo "❌ ERROR: Approximations found - use exact values with sources"
            exit 1
          else
            echo "✅ No approximations found"
          fi

      - name: Run MATLAB Code Analyzer
        uses: matlab-actions/run-command@v2
        with:
          command: |
            % Add all MATLAB paths
            addpath(genpath('matlab'));
            if exist('matlab_optimized', 'dir')
                addpath(genpath('matlab_optimized'));
            end
            
            % Find all MATLAB files
            mFiles = [
                dir('matlab/**/*.m');
                dir('matlab_optimized/**/*.m')
            ];
            
            % Run checkcode on each file
            hasIssues = false;
            for i = 1:length(mFiles)
                filePath = fullfile(mFiles(i).folder, mFiles(i).name);
                
                % Skip test files and archived code
                if contains(filePath, 'test') || contains(filePath, 'Archive') || contains(filePath, 'Backup')
                    continue;
                end
                
                % Run code analyzer
                issues = checkcode(filePath, '-id');
                if ~isempty(issues)
                    fprintf('Issues in %s:\n', filePath);
                    disp(issues);
                    hasIssues = true;
                end
            end
            
            % Exit with error if issues found
            if hasIssues
                error('Code quality issues found. Please fix them.');
            else
                fprintf('✅ All MATLAB files pass code analysis\n');
            end

      - name: Verify reproducibility
        uses: matlab-actions/run-command@v2
        with:
          command: |
            % Check for random number generation without seeds
            fprintf('🔍 Checking for unseeded randomness...\n');
            
            mFiles = [dir('matlab/**/*.m'); dir('matlab_optimized/**/*.m')];
            warningCount = 0;
            
            for i = 1:length(mFiles)
                filePath = fullfile(mFiles(i).folder, mFiles(i).name);
                
                % Skip archived code
                if contains(filePath, 'Archive') || contains(filePath, 'Backup')
                    continue;
                end
                
                % Read file content
                content = fileread(filePath);
                
                % Check for random functions without rng
                if contains(content, 'rand') || contains(content, 'randn') || contains(content, 'randi')
                    if ~contains(content, 'rng')
                        fprintf('⚠️ Warning: %s uses randomness without visible rng seed\n', filePath);
                        warningCount = warningCount + 1;
                    end
                end
            end
            
            if warningCount == 0
                fprintf('✅ All files using randomness have seeds or are intentionally random\n');
            end

  matlab-test:
    name: MATLAB Unit Tests
    runs-on: ubuntu-latest
    needs: matlab-quality  # Only run if quality checks pass
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up MATLAB
        uses: matlab-actions/setup-matlab@v2
        with:
          release: R2023b
          products: MATLAB Simulink

      - name: Run MATLAB tests
        uses: matlab-actions/run-tests@v2
        with:
          source-folder: matlab
          test-results-junit: test-results/matlab-results.xml
          code-coverage-cobertura: code-coverage/matlab-coverage.xml

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: matlab-test-results
          path: test-results/

      - name: Upload coverage
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: matlab-coverage
          path: code-coverage/

  matlab-build:
    name: MATLAB Build Check
    runs-on: ubuntu-latest
    if: false  # Enable if MEX files need compilation
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up MATLAB
        uses: matlab-actions/setup-matlab@v2
        with:
          release: R2023b
          products: MATLAB

      - name: Compile MEX files
        uses: matlab-actions/run-command@v2
        with:
          command: |
            % Find and compile all MEX files
            mexFiles = dir('**/*.c');
            for i = 1:length(mexFiles)
                fprintf('Compiling %s\n', mexFiles(i).name);
                mex(fullfile(mexFiles(i).folder, mexFiles(i).name));
            end
```

### MATLAB Best Practices

**1. Code Quality Checks**

Use MATLAB's built-in `checkcode` function:

```matlab
% Check a single file
issues = checkcode('myfile.m', '-id');

% Check all files in a directory
files = dir('matlab/**/*.m');
for i = 1:length(files)
    filePath = fullfile(files(i).folder, files(i).name);
    issues = checkcode(filePath);
    if ~isempty(issues)
        disp(issues);
    end
end
```

**2. Unit Test Structure**

```matlab
% matlab/tests/test_kinematics.m
function tests = test_kinematics
    % Test suite for kinematics functions
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    % Add paths
    addpath(genpath('matlab'));
end

function test_position_calculation(testCase)
    % Test position calculation
    expected = [1.0, 2.0, 3.0];
    actual = calculate_position(1.0, 2.0);
    verifyEqual(testCase, actual, expected, 'AbsTol', 1e-6);
end

function test_velocity_calculation(testCase)
    % Test velocity calculation
    expected = [0.5, 1.0, 1.5];
    actual = calculate_velocity(0.5, 1.0);
    verifyEqual(testCase, actual, expected, 'AbsTol', 1e-6);
end
```

**3. Reproducibility**

Always set random seeds:

```matlab
% At the beginning of scripts
rng(42, 'twister');  % Fixed seed for reproducibility

% Document why randomness is needed
% Example: Monte Carlo simulation requires random samples
rng(42);
samples = randn(1000, 1);
```

**4. Documentation Standards**

```matlab
function result = calculate_position(time, velocity)
% CALCULATE_POSITION Calculate position from time and velocity
%
% Syntax:
%   result = calculate_position(time, velocity)
%
% Inputs:
%   time - Time in seconds [s] (scalar or vector)
%   velocity - Velocity in m/s [m/s] (scalar or vector)
%
% Outputs:
%   result - Position in meters [m] (scalar or vector)
%
% Example:
%   pos = calculate_position(2.0, 5.0);  % Returns 10.0 m
%
% Notes:
%   Uses simple kinematic equation: position = velocity * time
%   Assumes constant velocity (no acceleration)
%
% References:
%   Physics equations from Halliday & Resnick (2013)
%
% See also: calculate_velocity, calculate_acceleration

    % Validate inputs
    validateattributes(time, {'numeric'}, {'real', 'finite'});
    validateattributes(velocity, {'numeric'}, {'real', 'finite'});
    
    % Calculate position
    result = velocity * time;  % [m] = [m/s] * [s]
end
```

**5. Constants Documentation**

```matlab
% Physical constants with sources
G = 9.80665;  % [m/s^2] Standard gravity (NIST 2019)
PI = 3.141592653589793;  % [-] Pi (use built-in pi instead)

% Better: Use MATLAB built-ins
g = 9.80665;  % [m/s^2] Standard gravity (NIST 2019)
circumference = 2 * pi * radius;  % Use built-in pi
```

### MATLAB CI Limitations

**Self-hosted runners:**
- May be needed for licensed MATLAB versions
- Configure with required toolboxes
- Set up license file or network license

**Alternative: MATLAB Docker**
```yaml
- name: Run MATLAB in Docker
  run: |
    docker run --rm -v $(pwd):/work mathworks/matlab:r2023b \
      -batch "cd /work; run_tests"
```

**Offline validation:**
```bash
# Check MATLAB syntax without MATLAB installed
# Use mlint or checkcode from command line
matlab -batch "checkcode('myfile.m')"
```

---

## JavaScript/TypeScript CI Workflow

### Complete Workflow Example

```yaml
name: JavaScript/TypeScript CI

on:
  push:
    branches: [main, master]
    paths:
      - 'src/**'
      - 'test/**'
      - 'package.json'
      - 'tsconfig.json'
      - '.github/workflows/js-ci.yml'
  pull_request:
    branches: [main, master]

jobs:
  lint-and-test:
    name: Lint and Test
    runs-on: ubuntu-latest
    strategy:
      fail-fast: true
      matrix:
        node-version: [18.x, 20.x]

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Lint with ESLint
        run: npm run lint || exit 1

      - name: Type check with TypeScript
        if: hashFiles('tsconfig.json') != ''
        run: npm run type-check || npx tsc --noEmit || exit 1

      - name: Format check with Prettier
        run: npm run format:check || exit 1

      - name: Run tests with coverage
        run: npm test -- --coverage || exit 1

      - name: Upload coverage
        if: matrix.node-version == '20.x'
        uses: codecov/codecov-action@v4
        with:
          files: coverage/coverage-final.json
          fail_ci_if_error: false
```

---

## Shell Script Validation

### Shell Script Checks

```yaml
- name: Validate shell scripts
  run: |
    echo "🔍 Validating shell scripts..."
    
    # Find all shell scripts
    scripts=$(find . -name "*.sh" -type f)
    
    # Syntax check
    for script in $scripts; do
      echo "Checking $script..."
      bash -n "$script" || exit 1
    done
    
    # Shellcheck (if available)
    if command -v shellcheck &> /dev/null; then
      shellcheck $scripts || exit 1
    fi
```

---

## Common Patterns

### Matrix Strategy

**Always use fail-fast:**

```yaml
strategy:
  fail-fast: true  # Stop all jobs if one fails
  matrix:
    python-version: ['3.10', '3.11', '3.12']
    os: [ubuntu-latest, windows-latest, macos-latest]
```

### Path Filters

**Only run CI when relevant files change:**

```yaml
on:
  push:
    paths:
      - 'python/**'
      - 'src/**'
      - 'tests/**'
      - 'requirements*.txt'
      - 'pyproject.toml'
      - '.github/workflows/*.yml'
```

### Reusable Workflows

**Create reusable workflow for common patterns:**

```yaml
# .github/workflows/reusable-python-test.yml
name: Reusable Python Test

on:
  workflow_call:
    inputs:
      python-version:
        required: true
        type: string
      source-dir:
        required: false
        type: string
        default: 'python'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ inputs.python-version }}
      - run: pytest ${{ inputs.source-dir }}/tests/
```

---

## Security Scanning

### Bandit (Python)

```yaml
- name: Run Bandit security check
  continue-on-error: true
  run: |
    pip install bandit==1.7.7
    bandit -r python/src --exclude 'tests' -f json -o bandit-report.json
```

### Safety (Python Dependencies)

```yaml
- name: Check for known vulnerabilities
  continue-on-error: true
  run: |
    pip install safety==3.0.1
    safety check --json || true
```

### TruffleHog (Secret Detection)

```yaml
- name: Scan for secrets
  uses: trufflesecurity/trufflehog@main
  with:
    path: ./
    base: ${{ github.event.repository.default_branch }}
    head: HEAD
```

### Trivy (Container Scanning)

```yaml
- name: Run Trivy scanner
  uses: aquasecurity/trivy-action@master
  with:
    scan-type: 'fs'
    scan-ref: '.'
    severity: 'CRITICAL,HIGH'
```

---

## Documentation Checks

### Markdown Linting

```yaml
- name: Lint markdown files
  run: |
    npm install -g markdownlint-cli
    markdownlint '**/*.md' --ignore node_modules --ignore archive
```

### Link Validation

```yaml
- name: Check for broken links
  run: |
    npm install -g markdown-link-check
    find . -name "*.md" -not -path "./node_modules/*" \
      -exec markdown-link-check {} \;
```

### Docstring Coverage

```yaml
- name: Check docstring coverage
  continue-on-error: true
  run: |
    pip install pydocstyle==6.3.0
    pydocstyle src/ || echo "⚠️ Missing docstrings"
```

---

## Summary

This unified CI/CD approach ensures:

✅ **Consistency** - Same patterns across all repositories  
✅ **Reproducibility** - Pinned versions for all tools  
✅ **Comprehensive** - Linting, testing, security, documentation  
✅ **Fast feedback** - Fail-fast strategies and path filters  
✅ **Clear reporting** - Proper exit codes and artifact uploads  
✅ **Security first** - Multiple scanning tools  
✅ **Multi-language** - Python, MATLAB, JavaScript/TypeScript, Shell  

### References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [MATLAB Actions](https://github.com/matlab-actions)
- [Codecov Documentation](https://docs.codecov.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [pytest Documentation](https://docs.pytest.org/)

---

**Document Maintenance:**
- Review quarterly for tool version updates
- Update when adding new repositories
- Sync with actual CI workflows in repositories
- Validate examples against running workflows

**Last Reviewed:** 2025-11-29  
**Next Review:** 2026-02-28
