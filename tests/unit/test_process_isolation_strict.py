"""Process isolation tests for strict unit-level physics engine adapters.

This module executes mock-driven Drake and Pinocchio adapter tests in separate
Python processes to avoid 'numpy' corruption caused by incompatible
C-extension mocking/reloading within a single pytest session (Issue #496).

Both tests spawn a full pytest session as a child process, so they are slower
than the suite-wide ``--timeout=60`` allows and they own an explicit
``@pytest.mark.timeout``. The child gets a bounded ``subprocess`` timeout that
sits *below* that budget, which is the point of the arrangement: a child that
hangs must fail this one test, not reach the session timeout. When pytest-timeout
fires it raises inside whichever frame is running -- here, ``communicate()`` --
and takes the whole session down with it, so the run ends at the first hang with
no ``short test summary info`` and no failure list at all. That is exactly how
the ``tests (3.12)`` lane on ``main`` died at 78% (UpstreamDrift#9474).
"""

import os
import subprocess  # nosec B404 - fixed interpreter + test path, no shell
import sys
from pathlib import Path

import pytest
from _pytest.outcomes import Failed

# Paths to the isolated unit test files
ISOLATED_TESTS_DIR = Path(__file__).parent / "isolated"
TEST_DRAKE_STRICT = ISOLATED_TESTS_DIR / "test_drake_strict.py"
TEST_PINOCCHIO_STRICT = ISOLATED_TESTS_DIR / "test_pinocchio_strict.py"

# The child does its actual work in under 5s -- both isolated modules skip when
# Drake/Pinocchio are absent, which is the case in CI -- and then *hangs at
# interpreter shutdown*: its captured stdout contains a complete
# ``short test summary info`` and the process still never exits, so the parent
# blocks in ``communicate()``. See UpstreamDrift#9511.
#
# These two values are a pair and must keep their ordering: the child is killed
# first so this test reports a clean failure, and only a genuinely stuck kill
# path should ever reach the outer mark. They are sized for the shutdown hang,
# not for the test run -- a larger bound only buys more wasted lane time.
_CHILD_TIMEOUT_SECONDS = 90
_TEST_TIMEOUT_SECONDS = 120


class TestProcessIsolationStrict:
    """Run specific strict unit tests in isolated subprocesses."""

    def run_isolated_test(self, test_file: Path) -> None:
        """Run pytest on a single file in a subprocess, bounded by a timeout.

        Args:
            test_file: Isolated test module to execute in its own interpreter.

        Raises:
            Failed: If the child exits non-zero or exceeds
                :data:`_CHILD_TIMEOUT_SECONDS`.
        """
        cmd = [sys.executable, "-m", "pytest", str(test_file), "-v", "--no-cov"]

        env = os.environ.copy()
        env.pop("MUJOCO_GL", None)

        try:
            result = subprocess.run(  # nosec B603 - fixed argv, shell=False
                cmd,
                capture_output=True,
                text=True,
                check=False,  # returncode is checked below for better reporting
                env=env,
                timeout=_CHILD_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as expired:
            # Without this branch the parent blocks in communicate() until the
            # suite-wide pytest timeout fires, which kills the whole session.
            stdout = expired.stdout or b""
            stderr = expired.stderr or b""
            pytest.fail(
                f"Isolated test {test_file.name} did not exit within "
                f"{_CHILD_TIMEOUT_SECONDS}s and was killed. If the captured stdout "
                f"below ends in a complete pytest summary, the tests themselves "
                f"finished and the child hung at interpreter shutdown "
                f"(UpstreamDrift#9511), not during the run.\n"
                f"--- STDOUT ---\n{_as_text(stdout)}\n"
                f"--- STDERR ---\n{_as_text(stderr)}"
            )

        if result.returncode != 0:
            pytest.fail(
                f"Isolated test {test_file.name} failed with exit code {result.returncode}.\n"
                f"--- STDOUT ---\n{result.stdout}\n"
                f"--- STDERR ---\n{result.stderr}"
            )

    @pytest.mark.timeout(_TEST_TIMEOUT_SECONDS)
    def test_drake_strict_isolated(self) -> None:
        """Run Drake strict tests in an isolated process to prevent numpy corruption."""
        if not TEST_DRAKE_STRICT.exists():
            pytest.fail(f"Test file not found: {TEST_DRAKE_STRICT}")
        self.run_isolated_test(TEST_DRAKE_STRICT)

    @pytest.mark.timeout(_TEST_TIMEOUT_SECONDS)
    def test_pinocchio_strict_isolated(self) -> None:
        """Run Pinocchio strict tests in an isolated process to prevent numpy corruption."""
        if not TEST_PINOCCHIO_STRICT.exists():
            pytest.fail(f"Test file not found: {TEST_PINOCCHIO_STRICT}")
        self.run_isolated_test(TEST_PINOCCHIO_STRICT)


@pytest.mark.unit
@pytest.mark.headless_safe
class TestIsolatedChildTimeoutIsContained:
    """A hung child must fail one test, never the pytest session.

    Marked ``unit`` because, unlike the two tests above, these spawn nothing:
    the subprocess call is monkeypatched. The isolation tests themselves stay
    unmarked so this fix does not quietly pull a multi-minute subprocess run
    into the fast unit lane (see issue #7158 for the marker backlog).
    """

    def test_timeout_becomes_a_test_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``TimeoutExpired`` is converted into a normal test failure.

        The regression: with no ``timeout=`` the parent blocked in
        ``communicate()`` until the suite-wide pytest timeout fired, which ends
        the session and produces no failure list for anything after it.
        """

        def fake_run(*_args: object, **kwargs: object) -> subprocess.CompletedProcess:
            assert kwargs.get("timeout") == _CHILD_TIMEOUT_SECONDS, (
                "the child must be spawned with a bounded timeout"
            )
            raise subprocess.TimeoutExpired(
                cmd=["pytest"],
                timeout=_CHILD_TIMEOUT_SECONDS,
                output=b"partial stdout",
                stderr=b"partial stderr",
            )

        monkeypatch.setattr(subprocess, "run", fake_run)

        with pytest.raises(Failed) as excinfo:
            TestProcessIsolationStrict().run_isolated_test(TEST_DRAKE_STRICT)

        message = str(excinfo.value)
        assert "did not exit within" in message
        assert "partial stdout" in message
        assert "partial stderr" in message

    def test_child_timeout_is_below_the_test_timeout(self) -> None:
        """The child must be killed before pytest-timeout can end the session."""
        assert _CHILD_TIMEOUT_SECONDS < _TEST_TIMEOUT_SECONDS


def _as_text(stream: bytes | str) -> str:
    """Return captured child output as text.

    ``TimeoutExpired`` carries bytes even when ``text=True`` was requested,
    unlike ``CompletedProcess``.

    Args:
        stream: Captured stdout or stderr from the child process.

    Returns:
        The stream decoded as UTF-8, replacing undecodable bytes.
    """
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream
