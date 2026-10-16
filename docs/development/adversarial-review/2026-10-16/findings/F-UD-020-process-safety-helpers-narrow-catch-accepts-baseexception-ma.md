# F-UD-020

**`process_safety` helpers: `narrow_catch` accepts `BaseException`; `managed_popen` swallows timeouts**

| Severity  | Category    | Confidence | Effort | Status   |
| --------- | ----------- | ---------- | ------ | -------- |
| 🟨 Medium | correctness | high       | S      | proposed |

**Locations:** `src/shared/python/core/process_safety.py:272`, `src/shared/python/core/process_safety.py:116`, `src/shared/python/core/process_safety.py:145`

## Finding

- Severity: Medium - Category: correctness
- Confidence: high
- Location: src/shared/python/core/process_safety.py:272-288; src/shared/python/core/process_safety.py:116-135;
  src/shared/python/core/process_safety.py:145-153
- Evidence:
  ```python
  if exc_type is Exception:
      raise ValueError("catching bare Exception defeats the purpose ...")
  # BaseException passes the issubclass(..., BaseException) check above
  ```
  and on timeout `_wait_or_terminate` logs, terminates and returns; the context manager then exits normally with
  `proc.returncode == -15` only logged at WARNING.
- Failure scenario: `with narrow_catch(BaseException, log_message="x")` swallows `KeyboardInterrupt`/`SystemExit` — strictly
  worse than the `except Exception` it replaces, and the rejection message implies the helper protects you. A caller of
  `managed_popen(cmd, timeout=30)` whose child hangs gets no exception; the success path continues with partial output
  (none of the five current call sites pass `timeout`, so today the helper waits forever instead).
- Why a newer engineer misses this: the identity check `is Exception` reads as "the broad one"; `BaseException` is broader.
  Context managers that clean up are assumed to also report.
- Proposed change: reject `BaseException` and `GeneratorExit`; raise `subprocess.TimeoutExpired` after cleanup unless
  `on_timeout="terminate"` is passed explicitly.
- Verification: `pytest.raises(ValueError)` for `narrow_catch(BaseException, ...)`; `pytest.raises(TimeoutExpired)` for a
  `sleep 10` with `timeout=0.1`.
- Effort: S

## Draft Pull Request

**Title:** `fix(core): narrow_catch rejects BaseException; popen raises on timeout`

### Why This Matters (The Pitch)

The process_safety helpers promise safety they do not deliver. narrow_catch refuses a bare Exception on the grounds that it defeats the purpose, but BaseException passes the check, so narrow_catch(BaseException, ...) swallows KeyboardInterrupt and SystemExit, which is strictly worse than the except Exception it replaces. managed_popen on timeout logs, terminates and then exits the context normally, leaving a returncode of -15 logged at WARNING; the caller continues with partial output and no exception. None of the five current call sites pass a timeout, so today the helper simply waits forever. Both are classic cases where a cleaning helper is assumed to also report. Rejecting BaseException and GeneratorExit, and raising subprocess.TimeoutExpired after cleanup unless on_timeout='terminate' is explicit, closes both before anyone starts passing timeouts. Size S, and callers get the failure they asked for.

### Proposed PR Body

## Problem

`src/shared/python/core/process_safety.py:272-288` rejects only `exc_type is Exception`, so `BaseException` is accepted; `:116-135` and `:145-153` make `_wait_or_terminate` log, terminate and return on timeout so the context manager exits normally with `proc.returncode == -15`.

## Change

- [ ] Reject `BaseException` and `GeneratorExit` in `narrow_catch`
- [ ] Raise `subprocess.TimeoutExpired` after cleanup unless `on_timeout="terminate"` is passed explicitly

## Verification

`pytest.raises(ValueError)` for `narrow_catch(BaseException, ...)`; `pytest.raises(TimeoutExpired)` for a `sleep 10` with `timeout=0.1`.

## Risk and Rollback

Low: no current call site passes a timeout or BaseException. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-UD-020

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
