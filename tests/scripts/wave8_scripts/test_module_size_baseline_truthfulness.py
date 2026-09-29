"""Truthfulness guard for ``scripts/config/module_size_budget_baseline.json``.

Issue #5922 documented that the module-size baseline had become *fraudulent*:
exceptions claimed line counts 3-5x larger than the files actually contained
(e.g. ``drake_gui_app.py`` claimed 2177 lines, actual 487). The whole budget
governance is undermined when reviewers can't trust the numbers in the file.

These tests pin down two invariants that, once they pass, prevent regression:

A. **Every active exception names a file that genuinely exceeds the budget.**
   It is not legitimate to keep an exception for a file that has since been
   decomposed down to a healthy size — the exception becomes load-bearing fiction.

B. **Every active exception's ``reason`` is within tolerance of the file's
   actual line count.** When a reason quotes "N lines, pending decomposition",
   that N must be within +/-10% of the real count today, so reviewers can
   trust the baseline at a glance.

Both invariants are also enforced inside ``check_module_size_budget.py`` so
that CI catches a regression at the same time as this test.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts import check_module_size_budget as mod

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_PATH = REPO_ROOT / "scripts" / "config" / "module_size_budget_baseline.json"

# Tolerance for the "N lines" claim embedded in an exception's reason.
# Files breathe a little as imports and helpers shift; +/- 10% gives wiggle
# room without permitting the 3-5x overstatement that motivated #5922.
TRUTHFULNESS_TOLERANCE = 0.10

# Regex picks up "1984 lines", "(2007 lines, ...)", "approx 1500 lines" etc.
_LINES_CLAIM_RE = re.compile(r"(\d{3,5})\s*lines", re.IGNORECASE)


def _load_baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _actual_line_count(rel_path: str) -> int:
    return mod.count_lines(REPO_ROOT / rel_path)


@pytest.fixture(scope="module")
def baseline() -> dict:
    return _load_baseline()


@pytest.fixture(scope="module")
def budget(baseline: dict) -> int:
    return int(baseline.get("max_lines", mod.DEFAULT_MAX_LINES))


def test_baseline_file_exists() -> None:
    assert BASELINE_PATH.is_file(), f"baseline missing: {BASELINE_PATH}"


def test_every_exception_path_resolves(baseline: dict) -> None:
    """Pre: every exception names a file that exists in the tree."""
    missing = [
        exc["path"]
        for exc in baseline.get("exceptions", [])
        if not (REPO_ROOT / exc["path"]).is_file()
    ]
    assert not missing, (
        "Exceptions reference files that do not exist:\n  " + "\n  ".join(missing)
    )


def test_no_exception_for_file_under_budget(baseline: dict, budget: int) -> None:
    """Post: an exception is only legitimate when the file truly exceeds budget.

    This is invariant (A) from the module docstring. A file that has been
    decomposed back under the budget must have its exception removed; leaving
    it in place is the precise failure mode #5922 called out.
    """
    stale = []
    for exc in baseline.get("exceptions", []):
        rel = exc["path"]
        actual = _actual_line_count(rel)
        if actual <= budget:
            stale.append(f"{rel}: actual={actual}, budget={budget}")
    assert not stale, (
        "These exceptions are stale — the files are now under budget and "
        "the exception should be deleted:\n  " + "\n  ".join(stale)
    )


def test_exception_reason_states_truthful_line_count(
    baseline: dict,
) -> None:
    """Post: any "N lines" claim in ``reason`` matches actual +/- 10%.

    Invariant (B). Prevents the "claim 2177, actual 487" rot.
    """
    liars = []
    for exc in baseline.get("exceptions", []):
        rel = exc["path"]
        reason = exc.get("reason", "")
        match = _LINES_CLAIM_RE.search(reason)
        if match is None:
            # No numeric claim, nothing to verify. The reason still has to
            # mention an issue/decomposition/legacy to pass the existing
            # _collect_active_exceptions check.
            continue
        claimed = int(match.group(1))
        actual = _actual_line_count(rel)
        tolerance = max(1, int(actual * TRUTHFULNESS_TOLERANCE))
        if abs(claimed - actual) > tolerance:
            liars.append(
                f"{rel}: claimed={claimed}, actual={actual}, tolerance=+/-{tolerance}"
            )
    assert not liars, (
        "Exception reasons quote line counts that no longer match the file. "
        "Update the reason (or remove the exception):\n  " + "\n  ".join(liars)
    )


# --------------------------------------------------------------------------
# Programmatic guard — the same invariants exposed as a callable so that
# check_module_size_budget.py can ratchet against future fraud at CI time.
# --------------------------------------------------------------------------


def test_validate_baseline_truthfulness_helper_passes(baseline: dict) -> None:
    """The script-level helper must agree with the test-level invariants.

    DbC: helper returns ``[]`` exactly when (A) and (B) both hold for every
    active exception. Keeping a shared implementation prevents drift between
    "what the tests check" and "what CI checks".
    """
    problems = mod.validate_baseline_truthfulness(baseline, REPO_ROOT)
    assert problems == []


def test_validate_baseline_truthfulness_flags_stale_exception(
    tmp_path: Path,
) -> None:
    """Synthetic case: a 5-line file with a 999-line claim must be flagged."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "tiny.py").write_text("x\n" * 5)
    config = {
        "max_lines": 100,
        "exceptions": [
            {
                "path": "src/tiny.py",
                "owner": "@m",
                "reason": "Legacy thing (999 lines, pending decomposition)",
            }
        ],
    }
    problems = mod.validate_baseline_truthfulness(config, tmp_path)
    # One file: under-budget AND lie about size. Either failure mode alone
    # would be enough; we expect at least one problem reported.
    assert problems, "expected truthfulness helper to flag the synthetic fraud"
    joined = "\n".join(problems)
    assert "src/tiny.py" in joined


def test_validate_baseline_truthfulness_accepts_honest_oversize(
    tmp_path: Path,
) -> None:
    """Synthetic case: a real oversize file with a matching claim must pass."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "big.py").write_text("x\n" * 200)
    config = {
        "max_lines": 100,
        "exceptions": [
            {
                "path": "src/big.py",
                "owner": "@m",
                "reason": "Legacy module (200 lines, pending decomposition)",
            }
        ],
    }
    assert mod.validate_baseline_truthfulness(config, tmp_path) == []
