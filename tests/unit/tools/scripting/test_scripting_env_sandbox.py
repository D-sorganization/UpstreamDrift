"""Security-boundary and behavior tests for the scripting sandbox.

Covers issue #7004 (security):
- _make_restricted_import blocks os/subprocess/socket/... and allows whitelist
- _make_restricted_builtins excludes eval/exec/open/compile/breakpoint
- ConsoleEnvironment.execute returns (stdout, stderr), captures exceptions
- introspection-dunder sandbox escape is blocked (regression for the real
  ``().__class__.__bases__[0].__subclasses__()`` -> real-builtins escape)
- timeout enforcement does not hang
- save/append/get_user_code round-trip
- reset clears the namespace

The escape tests are the security crux: a sandbox that merely removes
dangerous builtins is NOT sufficient — see _screen_source_for_escapes.
"""

from __future__ import annotations

import time

import pytest
from scripting.scripting_env import (
    _BLOCKED_BUILTINS,
    ConsoleEnvironment,
    _make_restricted_builtins,
    _make_restricted_import,
)


@pytest.fixture
def env(tmp_path) -> ConsoleEnvironment:
    """A console env with a short timeout and a temp user-lib path."""
    return ConsoleEnvironment(
        user_lib_path=str(tmp_path / "user_funcs.py"),
        max_execution_time=5,
    )


def run(env: ConsoleEnvironment, src: str) -> tuple[str, str]:
    out, err = env.execute(src)
    return out.strip(), err.strip()


class TestRestrictedImport:
    """_make_restricted_import denies host modules, allows the rest."""

    restricted = staticmethod(_make_restricted_import())

    @pytest.mark.parametrize(
        "module",
        ["os", "subprocess", "socket", "sys", "shutil", "ctypes", "pickle"],
    )
    def test_blocks_dangerous_modules(self, module: str) -> None:
        with pytest.raises(ImportError, match="blocked in the scripting sandbox"):
            self.restricted(module)

    def test_blocks_submodule_of_blocked_top_level(self) -> None:
        with pytest.raises(ImportError):
            self.restricted("os.path")

    @pytest.mark.parametrize("module", ["math", "json", "itertools"])
    def test_allows_safe_modules(self, module: str) -> None:
        mod = self.restricted(module)
        assert mod.__name__ == module


class TestRestrictedBuiltins:
    """_make_restricted_builtins removes code-injection primitives."""

    def test_dangerous_builtins_absent(self) -> None:
        safe = _make_restricted_builtins()
        for name in ("open", "eval", "exec", "compile", "breakpoint"):
            assert name not in safe
            assert name in _BLOCKED_BUILTINS

    def test_safe_builtins_present(self) -> None:
        safe = _make_restricted_builtins()
        for name in ("len", "range", "sum", "print", "list", "dict"):
            assert name in safe

    def test_import_is_replaced_with_restricted_wrapper(self) -> None:
        safe = _make_restricted_builtins()
        assert "__import__" in safe
        with pytest.raises(ImportError):
            safe["__import__"]("os")


class TestExecuteBlocksDangerousOps:
    """User code cannot import host modules or use removed builtins."""

    @pytest.mark.parametrize(
        "src",
        ["import os", "import subprocess", "import socket", "import sys"],
    )
    def test_import_blocked(self, env: ConsoleEnvironment, src: str) -> None:
        out, err = run(env, src)
        assert out == ""
        assert "blocked in the scripting sandbox" in err

    @pytest.mark.parametrize(
        "src",
        ['open("x", "w")', 'eval("1+1")', 'exec("y=1")', 'compile("1", "", "eval")'],
    )
    def test_removed_builtins_unavailable(
        self, env: ConsoleEnvironment, src: str
    ) -> None:
        out, err = run(env, src)
        assert "NameError" in err or "not defined" in err


class TestSandboxEscapeBlocked:
    """Regression: the classic CPython introspection escape must be blocked.

    Without source screening, user code could walk
    ``().__class__.__bases__[0].__subclasses__()`` to a class whose
    ``__init__.__globals__['__builtins__']`` is the real, unrestricted
    builtins dict — leaking open/eval/exec/__import__ and fully escaping.
    """

    @pytest.mark.parametrize(
        "src",
        [
            "().__class__.__bases__[0].__subclasses__()",
            "[c.__init__.__globals__ for c in int.__subclasses__()]",
            "(1).__class__.__mro__",
            'getattr((), "__class__")',
            "object.__subclasses__()",
        ],
    )
    def test_introspection_escape_blocked(
        self, env: ConsoleEnvironment, src: str
    ) -> None:
        out, err = run(env, src)
        assert out == ""
        assert "blocked in the scripting sandbox" in err

    def test_full_escape_to_os_is_blocked(self, env: ConsoleEnvironment) -> None:
        """The end-to-end escape (reaching os.getcwd) must fail."""
        escape = (
            "real=None\n"
            "for c in ().__class__.__bases__[0].__subclasses__():\n"
            "    g = c.__init__.__globals__\n"
            "    b = g.get('__builtins__')\n"
            "    if isinstance(b, dict) and 'open' in b:\n"
            "        real = b['__import__']('os'); break\n"
            "print(real.getcwd())\n"
        )
        out, err = run(env, escape)
        # Screen rejects the source before any introspection runs.
        assert "blocked in the scripting sandbox" in err
        # And crucially: no working directory leaked to stdout.
        assert out == ""


class TestExecuteAllowsPermittedOps:
    """Permitted scientific computing operations still work."""

    def test_arithmetic_expression(self, env: ConsoleEnvironment) -> None:
        out, err = run(env, "2 + 3 * 4")
        assert out == "14"
        assert err == ""

    def test_numpy_available(self, env: ConsoleEnvironment) -> None:
        out, err = run(env, "int(np.array([1, 2, 3]).sum())")
        assert out == "6"
        assert err == ""

    def test_math_available(self, env: ConsoleEnvironment) -> None:
        out, err = run(env, "math.sqrt(16)")
        assert out == "4.0"

    def test_statement_then_use(self, env: ConsoleEnvironment) -> None:
        env.execute("vals = [i * i for i in range(4)]")
        out, _ = run(env, "sum(vals)")
        assert out == "14"

    def test_print_captured_to_stdout(self, env: ConsoleEnvironment) -> None:
        out, _ = run(env, "print('hello')")
        assert out == "hello"


class TestExecuteContract:
    """execute() return-value and error-capture contract."""

    def test_returns_stdout_stderr_tuple(self, env: ConsoleEnvironment) -> None:
        result = env.execute("1 + 1")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_empty_source_returns_empty(self, env: ConsoleEnvironment) -> None:
        assert env.execute("   ") == ("", "")

    def test_none_source_raises(self, env: ConsoleEnvironment) -> None:
        with pytest.raises(ValueError, match="source must be provided"):
            env.execute(None)

    def test_user_exception_captured_not_raised(self, env: ConsoleEnvironment) -> None:
        out, err = run(env, "1 / 0")
        assert "ZeroDivisionError" in err

    def test_name_error_captured(self, env: ConsoleEnvironment) -> None:
        _, err = run(env, "undefined_name")
        assert "NameError" in err

    def test_negative_timeout_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="max_execution_time must be >= 0"):
            ConsoleEnvironment(
                user_lib_path=str(tmp_path / "u.py"),
                max_execution_time=-1,
            )


class TestTimeoutEnforcement:
    """The per-call timeout interrupts long-running code without hanging."""

    def test_infinite_loop_times_out(self, tmp_path) -> None:
        env = ConsoleEnvironment(
            user_lib_path=str(tmp_path / "u.py"),
            max_execution_time=1,
        )
        # A busy loop that never imports anything must be interrupted within
        # roughly the timeout window — the key guarantee is that execute()
        # RETURNS (does not hang).  On Unix the timeout surfaces as a captured
        # TimeoutError in stderr; on Windows the daemon-thread mechanism raises
        # KeyboardInterrupt in the calling thread, which execute() re-raises.
        start = time.monotonic()
        try:
            out, err = env.execute("n = 0\nwhile True:\n    n += 1\n")
        except KeyboardInterrupt:
            # Windows path: interrupt fired -> the loop was bounded, no hang.
            elapsed = time.monotonic() - start
            assert elapsed < 10, "timeout failed to interrupt the busy loop"
            return
        # Unix path: a TimeoutError was captured to stderr.
        elapsed = time.monotonic() - start
        assert elapsed < 10, "timeout failed to interrupt the busy loop"
        assert "Timeout" in (err or "")


class TestUserCodePersistence:
    """save/append/get_user_code round-trip and reset clears namespace."""

    def test_save_and_get_roundtrip(self, env: ConsoleEnvironment) -> None:
        code = "def my_func():\n    return 42\n"
        env.save_user_code(code)
        assert env.get_user_code() == code

    def test_append_user_code(self, env: ConsoleEnvironment) -> None:
        env.save_user_code("a = 1\n")
        env.append_user_code("b = 2")
        stored = env.get_user_code()
        assert "a = 1" in stored
        assert "b = 2" in stored

    def test_get_user_code_missing_file(self, env: ConsoleEnvironment) -> None:
        assert env.get_user_code() == ""

    def test_set_empty_library_path_rejected(self, env: ConsoleEnvironment) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            env.set_user_library_path("")

    def test_reset_clears_user_variables(self, env: ConsoleEnvironment) -> None:
        env.execute("custom_var = 123")
        assert "custom_var" in env.namespace
        env.reset()
        assert "custom_var" not in env.namespace
        # But baseline names are restored.
        assert "np" in env.namespace
        assert "__builtins__" in env.namespace

    def test_user_library_loaded_on_reset(self, env: ConsoleEnvironment) -> None:
        env.save_user_code("def helper():\n    return 7\n")
        env.reset()
        out, _ = run(env, "helper()")
        assert out == "7"
