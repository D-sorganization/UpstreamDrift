## 2026-01-23 - Security Audit

**Scan Results:**

- Dependencies: 0 vulnerabilities (H/M/L)
- Code Analysis: 4898 issues (0 High, 49 Medium, 4849 Low)
- Pattern Scan: 0 findings

**Issues Created:** None

**Deferred:**

- **B102 (exec_used)**: Found in `tests/test_pinocchio_ecosystem.py`. **Justification**: Verified fixed in codebase (replaced with `importlib`).
- **B103 (set_bad_file_permissions)**: Found in `tests/integration/test_phase1_security_integration.py`. **Justification**: Test setup (setting script permissions).
- **B604 (subprocess_without_shell_equals_true)**: Found in `tests/integration/test_phase1_security_integration.py` and `tests/unit/test_secure_subprocess.py`. **Justification**: Security tests verifying that `shell=True` is rejected.

**Existing Issues:**

- B104 (hardcoded_bind_all_interfaces): Covered by ISSUE_003
- B310 (blacklist - URL): Covered by ISSUE_004
- B314 (blacklist - XML): Covered by ISSUE_005
- B608 (hardcoded_sql_expressions): Covered by ISSUE_006
- B108 (hardcoded_tmp_directory): Covered by ISSUE_007

**Low Severity Findings Summary:**

- B101 (assert_used): 4634 (Primarily in tests)
- B603 (subprocess_without_shell_equals_true): 82
- B404 (blacklist_subprocess): 36
- B110 (try_except_pass): 32
- Other: 65

## 2026-03-08 - [Insecure Deserialization]

**Vulnerability:** Found arbitrary code execution vulnerability via pickle deserialization in `src/learning/imitation/learners.py`. `np.load` was being called with `allow_pickle=True` to load model checkpoints, and `np.savez` was using `dtype=object` which forces the use of pickle.
**Learning:** `allow_pickle=True` in `np.load` is extremely dangerous and can lead to complete system compromise if untrusted files are loaded. The use of nested dictionary objects inside `np.savez` naturally leads developers to rely on pickle.
**Prevention:** We should serialize complex structures (like configurations) using secure formats such as JSON strings, and save neural network weights as individual flat numpy arrays (`policy_0_W`, `policy_0_b`) inside `.npz` files instead of structured Python objects. This allows us to strictly use `allow_pickle=False`.

## 2024-06-13 - Prevent insecure deserialization via np.load

**Vulnerability:** Calling np.load without allow_pickle=False can lead to arbitrary code execution if an attacker supplies a malicious pickle-embedded numpy file.
**Learning:** Always specify allow_pickle=False when loading numpy files to ensure insecure deserialization does not happen.
**Prevention:** Ensured allow_pickle=False is the default behavior across all data loading utilities and modules that use np.load.

## 2026-03-09 - Command Injection in pandas query()

**Vulnerability:** Found arbitrary code execution vulnerability in `DataProcessorEngine.query()` because user input was passed directly to `pandas.DataFrame.query()` without validation. `DataFrame.query()` effectively uses `pd.eval()` under the hood, which is prone to executing arbitrary code.
**Learning:** `DataFrame.query()` carries the exact same security risks as `DataFrame.eval()`. While previous fixes addressed `add_calculated_column` which uses `eval()`, the `query()` method was missed, showing that developers must audit _all_ pandas methods that parse string expressions (`eval`, `query`).
**Prevention:** Always validate expressions passed to `pd.DataFrame.query()` or `pd.DataFrame.eval()` using an AST-based validator to ensure they only contain safe operations.

## 2026-04-05 - [Insecure Deserialization via pandas read_pickle]

**Vulnerability:** Found arbitrary code execution vulnerability via pickle deserialization in `src/shared/python/upstream_drift_tools/data_processing/io.py` where `pd.read_pickle` was being called for files with the `.pkl` or `.pickle` extension.
**Learning:** Even though the function caller is assumed to provide trusted data, `pd.read_pickle` allows for extremely dangerous code execution. We cannot rely on caller context for safety.
**Prevention:** Explicitly raise a `ValueError` when attempting to read or write pickle formats, forcing the use of safer serialization formats like JSON, Parquet, or CSV.

## 2026-04-06 - Command Injection in pandas query() via filter_data

**Vulnerability:** Found arbitrary code execution vulnerability in `DataProcessorEngine.filter_data()` where user-provided column and operator strings were concatenated and passed directly to `pandas.DataFrame.query()` without validation.
**Learning:** `DataFrame.query()` evaluates string expressions and is vulnerable to injection if the concatenated string isn't validated, even if parts of it are formatted dynamically.
**Prevention:** Always validate constructed query expressions passed to `pd.DataFrame.query()` using an AST-based validator to ensure they only contain safe operations.

## 2026-04-10 - Enable Docker Image Scanning

**Vulnerability:** No container security scanning was configured in CI workflows, leaving Docker images unaudited for vulnerabilities. Base image vulnerabilities could be inherited silently.
**Learning:** Container vulnerabilities are undetected until production without proper CI scanning tools. Relying solely on source code scanning leaves a gap in the security posture regarding the environment the code runs in.
**Prevention:** Ensured the `.github/workflows/docker-security-scan.yml` is enabled by removing its `.disabled` extension so that Trivy runs on every PR and push.

## 2024-04-24 - Fix SQL Injection Vulnerability in Recording Library

**Vulnerability:** String-based query construction with `f-strings` in `get_unique_values` (Bandit B608).
**Learning:** Even when input is validated against a whitelist, security scanners will flag string-based SQL query construction. Using a hardcoded mapping of query strings eliminates both the actual risk and the static analysis warnings.
**Prevention:** Use hardcoded SQL query maps for column names (which cannot be parameterized in standard SQL bindings) instead of dynamic string construction.

## 2026-01-20 - Fix SQL Injection in recording_library.py

**Vulnerability:** String-based query construction allows potential SQL injection (Bandit B608).
**Learning:** For dynamic column selection where parameterization is not possible, hardcoded query mapping prevents SQL injection and satisfies static analysis without needing nosec annotations.
**Prevention:** Use dictionary mapping with static SQL strings for queries that depend on variable column names.

## 2026-04-27 - Annotate B404/B604 in test cases

**Vulnerability:** Subprocess `shell=True` (Bandit B604) flagged in testing files intentionally checking security blocks.
**Learning:** Static analysis tools flag intentional security failures in tests unless explicitly suppressed.
**Prevention:** Added `# nosec` annotations to intentional `shell=True` tests.

## 2024-04-27 - Command Injection Risk in Process Worker

**Vulnerability:** Direct use of `subprocess.Popen` without command validation in `ProcessWorker`.
**Learning:** Raw subprocess calls can allow arbitrary command injection if unsanitized inputs are provided as command arguments.
**Prevention:** Always use the custom `secure_popen` wrapper from `src.shared.python.security.secure_subprocess` which provides validation against allowed commands and prevents dangerous arguments like `shell=True`.

## 2024-05-09 - [Insecure Deserialization in Imitation Learning Models]

**Vulnerability:** Arbitrary Code Execution via `np.load(..., allow_pickle=True)`
**Learning:** Legacy ML saving routines (`_bc.py`, `_gail.py`) directly dumped nested dictionaries (neural net layers, training config) into `.npz` files, which mandated `allow_pickle=True` to reload. This creates a critical insecure deserialization vulnerability.
**Prevention:** Always flatten complex objects before saving. Serialize dictionaries or configurations into JSON strings, and extract individual layer weights (`W`, `b`) into separate numpy arrays with a primitive naming schema (`layer_0_W`, `layer_0_b`). This pattern completely eliminates the need for Python pickling during deserialization.

## 2024-05-15 - Insecure Deserialization via np.load

**Vulnerability:** Found uses of `np.load` without explicitly passing `allow_pickle=False` when loading `.npy` and `.npz` files (e.g., in `src/shared/python/physics/_topography_io.py`, `src/shared/python/pose_interchange/pose_io.py`, `src/engines/physics_engines/putting_green/python/_green_loader.py`, `src/engines/physics_engines/putting_green/python/_surface_io.py`).
**Learning:** `np.load` in older numpy versions and by default can allow loading pickled python objects which is prone to arbitrary code execution attacks. Always explicitly pass `allow_pickle=False` unless absolutely necessary and loading trusted data.
**Prevention:** Explicitly pass `allow_pickle=False` when using `np.load` or `np.savez`.
## 2026-05-20 - Insecure XML Parsing

**Vulnerability:** Found uses of `xml.etree.ElementTree` to parse potentially untrusted XML/URDF/MJCF/SDF files, which is vulnerable to XML External Entity (XXE) and billion laughs attacks.
**Learning:** Standard library `xml.etree.ElementTree` is not secure against maliciously constructed data.
**Prevention:** Always use the `defusedxml` package, which acts as a drop-in replacement but protects against these vulnerabilities.
