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
