# ruff: noqa: E501
"""Python scripting environment (MATLAB-like backend).

Provides an interactive console environment that can be embedded into applications
or run standalone, featuring namespace management, persistent user scripts, and
stdout capturing.

Security notice — controlled use of ``exec`` / ``eval``
--------------------------------------------------------
This module intentionally uses Python's ``exec()`` and ``eval()`` builtins.
The usage is *controlled* for the following reasons:

1. **Explicit consent**: The scripting environment is an opt-in interactive
   console.  Users deliberately type code into it; they are not supplying
   untrusted third-party input.
2. **Restricted namespace**: The execution namespace is seeded with a
   restricted ``__builtins__`` dict.  Dangerous primitives (``open``,
   ``exec``, ``eval``, ``compile``, ``breakpoint``) are removed.
   ``__import__`` is *replaced* with a sandbox-aware wrapper that denies
   host-process modules (``os``, ``subprocess``, ``sys``, ``socket``, …)
   while still allowing internal C-extension sub-imports (numpy, scipy, …).
3. **No web-facing exposure**: This class is never called from a network
   handler.  It is only instantiated by GUI widgets running in-process.
4. **Execution timeout**: Each ``execute()`` call is bounded by
   ``max_execution_time`` seconds (default 30 s).  On Linux/macOS the
   timeout is enforced via ``signal.alarm``; on Windows a daemon thread
   is used to interrupt the main thread.
5. **Resource limits (Linux only)**: CPU time and virtual-memory ceilings
   are applied via ``resource.setrlimit`` when the ``resource`` module is
   available.  On Windows these limits cannot be applied in-process;
   full process-level sandboxing requires running ``ConsoleEnvironment``
   inside a subprocess (e.g. via ``multiprocessing`` with a ``Pool``).
6. **AST escape screen**: Before any user source is compiled, it is passed
   through :func:`_screen_source_for_escapes`, which rejects the classic
   attribute-introspection sandbox escapes (``__class__`` /
   ``__bases__`` / ``__subclasses__`` / ``__globals__`` traversal) and
   runtime-constructed dunder names built via ``getattr``/``setattr``/
   ``vars``/``type``/``delattr`` with a non-literal or dunder name
   argument.  Violations raise :class:`SecurityError`.
7. **Bandit suppression**: The ``# nosec B102`` / ``# nosec B307`` markers
   are intentional and reviewed.  Do not remove them without re-evaluating
   the threat model above.

Real security boundary
-----------------------
The **authoritative** trust boundary for genuinely untrusted code is
**out-of-process** isolation (a separate OS process with OS-level memory,
CPU, filesystem, and network limits).  CPython's object model makes a
perfect in-process sandbox impossible: a sufficiently determined attacker
can find new introspection gadgets faster than a blocklist can grow.

The restricted ``__builtins__`` dict and the AST escape screen are
*defense-in-depth* layers for the intended use case (an opt-in interactive
console where the operator types their own code), **not** a guarantee
against a hostile adversary.  Do not expose ``ConsoleEnvironment`` to
network-supplied input; run it out-of-process if you must.

If you need to evaluate *user-supplied expressions from an untrusted source*
(e.g. a REST API), use :mod:`shared.python.safe_eval` instead, which
performs strict AST-level allowlisting before execution.
"""

import ast
import builtins
import contextlib
import ctypes
import io
import math
import os
import sys
import threading
import traceback
from collections.abc import Iterator
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Resource limits (Linux / macOS only)
# ---------------------------------------------------------------------------
try:
    import resource as _resource

    _HAS_RESOURCE = True
except ImportError:  # Windows
    _resource = None  # type: ignore[assignment]
    _HAS_RESOURCE = False

# CPU soft/hard limits in seconds. These are *process-wide* OS ceilings.
# They are NOT applied at import time: importing this module must be free of
# side effects so any consumer (e.g. a numpy/scipy GUI) is not silently
# sentenced to a 10s cumulative-CPU SIGKILL or a 512 MiB allocation ceiling.
# Call ``apply_process_resource_limits()`` explicitly, or set the
# ``SCRIPTING_ENV_APPLY_RLIMITS`` env var to opt in at import time.
_CPU_SOFT_LIMIT_S = 5
_CPU_HARD_LIMIT_S = 10
# Virtual memory ceiling: 512 MiB
_MEM_LIMIT_BYTES = 512 * 1024 * 1024


def apply_process_resource_limits() -> bool:
    """Apply process-wide CPU/memory rlimits (Linux/macOS only).

    WARNING: ``setrlimit`` mutates the *entire* OS process, not just this
    sandbox. Only call this from a process you own (e.g. a dedicated
    out-of-process worker), never from a shared library import path.

    Returns ``True`` if limits were applied, ``False`` on unsupported
    platforms (Windows) or if ``resource`` is unavailable.
    """
    if sys.platform == "win32" or not _HAS_RESOURCE or _resource is None:
        return False
    applied = False
    try:
        _resource.setrlimit(
            _resource.RLIMIT_CPU, (_CPU_SOFT_LIMIT_S, _CPU_HARD_LIMIT_S)
        )
        applied = True
    except (ValueError, getattr(_resource, "error", Exception)):
        pass
    try:
        _resource.setrlimit(
            getattr(_resource, "RLIMIT_AS", 0), (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES)
        )
        applied = True
    except (ValueError, getattr(_resource, "error", Exception)):
        pass
    return applied


# Opt-in import-time application for embedders that explicitly want the
# legacy process-global behaviour.
if os.environ.get("SCRIPTING_ENV_APPLY_RLIMITS"):
    apply_process_resource_limits()

# ---------------------------------------------------------------------------
# Restricted builtins
# ---------------------------------------------------------------------------
# The following names are *removed* from the sandbox __builtins__ dict.
# NOTE: ``__import__`` is handled separately — it must exist so that C
# extensions (numpy, scipy) can perform internal sub-imports, but it is
# *replaced* with a restricted wrapper that blocks dangerous top-level
# module names (see _make_restricted_import / _BLOCKED_IMPORT_MODULES).
_BLOCKED_BUILTINS: frozenset[str] = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "breakpoint",
    }
)

# Modules that user code must NOT be able to import from the sandbox.
_BLOCKED_IMPORT_MODULES: frozenset[str] = frozenset(
    {
        "os",
        "subprocess",
        "sys",
        "socket",
        "shutil",
        "pathlib",
        "io",
        "builtins",
        "importlib",
        "ctypes",
        "signal",
        "resource",
        "threading",
        "multiprocessing",
        "pty",
        "atexit",
        "gc",
        "code",
        "codeop",
        "runpy",
        "ast",
        "dis",
        "marshal",
        "pickle",
        "shelve",
        "dbm",
        "sqlite3",
        "struct",
    }
)


# ---------------------------------------------------------------------------
# AST escape screen (defense-in-depth, issue #3180)
# ---------------------------------------------------------------------------
# Removing dangerous *builtins* does not stop user code reaching dangerous
# *types* through object introspection, e.g.::
#
#     ().__class__.__bases__[0].__subclasses__()  # -> os / subprocess gadgets
#
# The screen below is a static AST pre-pass run before compile/exec.  It is a
# blocklist (defense-in-depth), NOT the real boundary — see the module
# docstring for the authoritative out-of-process boundary.


class SecurityError(Exception):
    """Raised when user source fails the scripting sandbox AST screen.

    Signals that the submitted code contains an attribute-introspection
    escape gadget (dunder traversal) or a runtime-constructed dunder name.
    This is *not* a normal user-code error and therefore is reported
    separately from :data:`USER_CODE_ERROR_TYPES`.
    """


# Builtins that can fabricate or follow an arbitrary attribute name at
# runtime, defeating a purely syntactic dunder-attribute screen.
_INTROSPECTION_GADGETS: frozenset[str] = frozenset(
    {
        "getattr",
        "setattr",
        "delattr",
        "vars",
        "type",
        "globals",
        "locals",
    }
)


def _is_dunder(name: str) -> bool:
    """Return True if *name* is a ``__dunder__`` identifier."""
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _screen_source_for_escapes(source: str) -> None:
    """Reject attribute-introspection sandbox escapes in *source*.

    Performs a static AST pre-pass before the source is compiled/executed.
    Rejected constructs (each raising :class:`SecurityError`):

    1. **Dunder attribute access** — any ``x.__dunder__`` access
       (``__class__``, ``__bases__``, ``__subclasses__``, ``__globals__``,
       ``__dict__``, …) used to walk the object graph toward host types.
    2. **Runtime-constructed dunder names** — calls to introspection
       gadgets (``getattr``/``setattr``/``delattr``/``vars``/``type``/
       ``globals``/``locals``) whose attribute-name argument is **not** a
       safe screened string literal.  A non-literal name argument (e.g.
       ``chr(95) * 2 + 'class' + ...``) could smuggle a dunder past a purely
       syntactic check, so such calls are rejected outright.  A literal
       dunder name (``getattr(x, '__class__')``) is also rejected.

    Bare ``type(x)`` / ``globals()`` / ``vars()`` calls (no fabricated
    attribute name) are permitted; it is only the name-fabrication and
    dunder-traversal patterns that are blocked.

    Args:
        source: Raw user source about to be compiled.

    Raises:
        SecurityError: If a screened escape pattern is present.
        SyntaxError: If *source* is not parseable (callers handle this as a
            normal syntax error during compile).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Defer to the normal compile() path for syntax-error reporting.
        return

    for node in ast.walk(tree):
        # (1) Static dunder attribute access: x.__class__, x.__bases__, ...
        if isinstance(node, ast.Attribute) and _is_dunder(node.attr):
            raise SecurityError(
                f"access to dunder attribute '{node.attr}' is blocked in "
                "the scripting sandbox"
            )

        # (2) Dunder string literals anywhere — these have no legitimate use
        # in console code and are the payload for indirect traversal such as
        # ``vars(x)['__class__']`` or ``getattr(x, '__class__')``.
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _is_dunder(node.value):
                raise SecurityError(
                    f"use of dunder name literal '{node.value}' is blocked "
                    "in the scripting sandbox"
                )

        # (3) Introspection-gadget calls with a fabricated/dunder name arg.
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Name)
                and func.id in _INTROSPECTION_GADGETS
                and len(node.args) >= 2
            ):
                name_arg = node.args[1]
                if not isinstance(name_arg, ast.Constant) or not isinstance(
                    name_arg.value, str
                ):
                    raise SecurityError(
                        f"'{func.id}' with a non-literal attribute name is "
                        "blocked in the scripting sandbox"
                    )
                if _is_dunder(name_arg.value):
                    raise SecurityError(
                        f"'{func.id}' with dunder attribute name "
                        f"'{name_arg.value}' is blocked in the scripting "
                        "sandbox"
                    )


def _make_restricted_import(
    blocked: frozenset[str] = _BLOCKED_IMPORT_MODULES,
) -> Any:
    """Return a ``__import__`` wrapper that blocks host-process modules.

    The wrapper delegates to the real ``builtins.__import__`` for safe
    modules so that C extensions (numpy, scipy) can perform their internal
    sub-imports normally.  Any attempt to import a blocked top-level module
    raises ``ImportError``.

    Args:
        blocked: Set of top-level module names to deny.

    Returns:
        A callable with the same signature as ``builtins.__import__``.
    """
    _real_import = builtins.__import__

    def _restricted_import(
        name: str,
        globals: dict[str, Any] | None = None,  # noqa: A002
        locals: dict[str, Any] | None = None,  # noqa: A002
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        top_level = name.split(".")[0]
        if top_level in blocked:
            raise ImportError(f"import of '{name}' is blocked in the scripting sandbox")
        return _real_import(name, globals, locals, fromlist, level)

    return _restricted_import


def _make_restricted_builtins() -> dict[str, Any]:
    """Return a copy of ``builtins.__dict__`` with dangerous names removed/replaced.

    Returns:
        A dict suitable for use as the ``__builtins__`` value in an exec/eval
        namespace.  Dangerous primitives are absent; ``__import__`` is replaced
        with a module-blocking wrapper.
    """
    safe: dict[str, Any] = {
        k: v for k, v in vars(builtins).items() if k not in _BLOCKED_BUILTINS
    }
    # Replace __import__ with a restricted wrapper so internal C-extension
    # imports still work while user-initiated imports of dangerous modules
    # are blocked.
    safe["__import__"] = _make_restricted_import()
    return safe


# Default per-call execution timeout (seconds).
_DEFAULT_MAX_EXECUTION_TIME = 30

USER_CODE_ERROR_TYPES = (
    ArithmeticError,
    AssertionError,
    AttributeError,
    ImportError,
    LookupError,
    NameError,
    OSError,
    RuntimeError,
    SyntaxError,
    TypeError,
    ValueError,
)

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

try:
    import scipy

    HAS_SCIPY = True
except ImportError:
    scipy = None
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Timeout helpers
# ---------------------------------------------------------------------------

try:
    import signal as _signal

    _HAS_SIGNAL_ALARM = hasattr(_signal, "alarm")
except ImportError:
    _signal = None  # type: ignore[assignment]
    _HAS_SIGNAL_ALARM = False


def _raise_keyboard_interrupt_in_thread(thread_id: int) -> None:
    """Raise ``KeyboardInterrupt`` in *thread_id* using the CPython C API.

    This is the Windows-compatible fallback for enforcing the execution
    timeout.  Only works with CPython; on other runtimes the call is a
    no-op.

    ``PyThreadState_SetAsyncExc`` returns the number of thread states that
    were modified: ``1`` on success, ``0`` if the target thread id was not
    found, and ``>1`` if it (erroneously) matched multiple states — in which
    case the CPython C-API contract requires reverting the pending exception
    by calling it again with ``NULL`` so a stray ``KeyboardInterrupt`` is not
    left queued in an unrelated thread.
    """
    modified = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread_id),
        ctypes.py_object(KeyboardInterrupt),
    )
    if modified > 1:
        # Revert: clear the pending async exception we just set.
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread_id),
            ctypes.c_void_p(0),
        )


class ConsoleEnvironment:
    """Manages the interactive Python namespace and execution for the CLI."""

    def __init__(
        self,
        default_namespace: dict[str, Any] | None = None,
        user_lib_path: str = "~/.shared_console_user_funcs.py",
        max_execution_time: int = _DEFAULT_MAX_EXECUTION_TIME,
    ) -> None:
        """Initialize the console environment.

        Args:
            default_namespace: Variables/modules to inject into the namespace.
            user_lib_path: Path to the user's persistent functions file.
            max_execution_time: Wall-clock timeout in seconds for each
                ``execute()`` call (default 30 s).  On Linux/macOS the
                timeout is enforced via ``signal.alarm``; on Windows a
                daemon thread raises ``KeyboardInterrupt`` in the calling
                thread after this many seconds.  Set to 0 to disable.

        Raises:
            ValueError: If ``max_execution_time`` is negative.
        """
        if max_execution_time < 0:
            raise ValueError("max_execution_time must be >= 0")
        self._initial_namespace = default_namespace or {}
        self.namespace: dict[str, Any] = {}
        self._user_lib_path = os.path.expanduser(user_lib_path)
        self._max_execution_time = max_execution_time
        self.reset()

    def reset(self) -> None:
        """Reset the namespace to its initial state.

        Follows DbC by ensuring the namespace is clear before population.
        The namespace is seeded with a restricted ``__builtins__`` dict that
        removes dangerous host-process access primitives (``open``, ``exec``,
        ``eval``, ``compile``, ``breakpoint``) and replaces ``__import__``
        with a module-blocking wrapper.
        """
        self.namespace.clear()

        # Install restricted builtins — this is the primary blast-radius guard.
        # User code running inside exec/eval will only see this dict for name
        # lookups on builtins, preventing access to file I/O and code injection.
        self.namespace["__builtins__"] = _make_restricted_builtins()

        # Add basic dependencies
        self.namespace["np"] = np
        self.namespace["math"] = math
        if HAS_PANDAS:
            self.namespace["pd"] = pd
        if HAS_SCIPY:
            self.namespace["scipy"] = scipy

        # Inject caller-supplied defaults (vetted by the embedding application)
        self.namespace.update(self._initial_namespace)

        # Provide specialized help function that works within the console
        self.namespace["help"] = self._custom_help

        # Load user functions if they exist
        self.refresh_user_functions()

    def _custom_help(self, obj: Any) -> None:
        """Custom help function that writes to stdout, redirectable by the CLI."""
        import pydoc

        sys.stdout.write(pydoc.render_doc(obj) + "\n")

    def set_user_library_path(self, path: str) -> None:
        """Set the path for saving/loading user custom functions.

        Args:
            path: The file path for the user library. If relative, it will be
                relative to the current working directory. If absolute, it will
                be used as-is.

        Raises:
            ValueError: If path is empty.
        """
        # Precondition check
        if not path:
            raise ValueError("Library path cannot be empty.")
        expanded_path = os.path.expanduser(path)
        self._user_lib_path = expanded_path

    def save_user_code(self, code: str) -> None:
        """Save raw python code to the user library path."""
        with open(self._user_lib_path, "w", encoding="utf-8") as f:
            f.write(code)

    def append_user_code(self, code: str) -> None:
        """Append raw python code to the user library path."""
        with open(self._user_lib_path, "a", encoding="utf-8") as f:
            f.write("\n" + code + "\n")

    def get_user_code(self) -> str:
        """Get the current user code."""
        if not os.path.exists(self._user_lib_path):
            return ""
        with open(self._user_lib_path, encoding="utf-8") as f:
            return f.read()

    def refresh_user_functions(self) -> None:
        """Reload user functions into the namespace.

        Expected user-code errors (those listed in ``USER_CODE_ERROR_TYPES``)
        are caught, reported to stderr, and swallowed so the host application
        keeps running.  System-level failures (``MemoryError``, etc.) are
        **re-raised** so they are not silently hidden.

        Raises:
            KeyboardInterrupt: If the user interrupts execution.
            SystemExit: If the code explicitly calls sys.exit().
            BaseException: Any non-user-code exception (e.g. ``MemoryError``)
                is re-raised after reporting.
        """
        if not os.path.exists(self._user_lib_path):
            return

        with open(self._user_lib_path, encoding="utf-8") as f:
            code = f.read()

        try:
            # Defense-in-depth: screen the persisted library for escape
            # gadgets before executing it into the live namespace.
            _screen_source_for_escapes(code)
            # Execute within current namespace so imports/functions are persistent
            exec(code, self.namespace)  # nosec B102
        except (
            SecurityError,
            *USER_CODE_ERROR_TYPES,
        ) as e:  # noqa: BLE001 — user library code may raise anything; report and continue
            sys.stderr.write(f"Error loading user library: {e}\n")
            sys.stderr.flush()

    # ------------------------------------------------------------------
    # Internal timeout context managers
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _timeout_context(self) -> Iterator[None]:
        """Context manager that enforces ``_max_execution_time``.

        On Linux/macOS uses ``signal.alarm`` (only valid on the main thread).
        On Windows, or when called from a non-main thread, falls back to a
        daemon thread that raises ``KeyboardInterrupt`` via the CPython C API.
        """
        timeout = self._max_execution_time
        if not timeout:
            yield
            return

        on_main_thread = threading.current_thread() is threading.main_thread()

        if _HAS_SIGNAL_ALARM and on_main_thread:
            # Unix path — signal.alarm is precise and zero-overhead.
            def _handler(signum: int, frame: object) -> None:  # noqa: ARG001
                raise TimeoutError(f"Execution exceeded {timeout} s CPU time limit")

            old_handler = _signal.signal(_signal.SIGALRM, _handler)  # type: ignore[attr-defined]
            _signal.alarm(timeout)  # type: ignore[attr-defined]
            try:
                yield
            finally:
                _signal.alarm(0)  # type: ignore[attr-defined]
                _signal.signal(_signal.SIGALRM, old_handler)  # type: ignore[attr-defined]
        else:
            # Windows / non-main-thread path — daemon thread fires KI.
            #
            # ``timer.cancel()`` only stops a Timer whose target has not yet
            # *begun* running; once ``_fire`` has been entered, cancel() is a
            # no-op and the async ``KeyboardInterrupt`` would still be queued —
            # possibly landing AFTER ``yield`` returns and leaking into
            # unrelated host code.  Guard delivery with a lock + done flag:
            # the ``finally`` marks completion under the lock before cancelling,
            # and ``_fire`` re-checks the flag under the same lock and skips the
            # injection if the protected block already finished.  Because the
            # async exception is raised while the lock is held, it cannot be
            # delivered until ``_fire`` returns and releases the lock, so a
            # late-but-still-pre-completion fire is fully serialized against the
            # ``finally``.
            caller_id = threading.get_ident()
            delivery_lock = threading.Lock()
            state = {"completed": False, "fired": False}

            def _fire() -> None:
                with delivery_lock:
                    if state["completed"]:
                        # The guarded block already exited; do not inject a
                        # stray interrupt into whatever the caller does next.
                        return
                    state["fired"] = True
                    _raise_keyboard_interrupt_in_thread(caller_id)

            timer = threading.Timer(timeout, _fire)
            timer.daemon = True
            timer.start()
            try:
                yield
            finally:
                timer.cancel()
                # Close the delivery race deterministically. Acquiring the lock
                # serializes against ``_fire``: once we hold it, ``_fire`` has
                # either already injected the KeyboardInterrupt (state["fired"]
                # is True) or is now permanently disarmed by state["completed"].
                fired = False
                while True:
                    try:
                        with delivery_lock:
                            state["completed"] = True
                            fired = state["fired"]
                        break
                    except KeyboardInterrupt:
                        # A timeout KI surfaced while we were finalizing. Absorb
                        # it here so it cannot leak past the context boundary,
                        # then retry the bookkeeping that the interrupt aborted.
                        fired = True
                if fired:
                    # Drain any KI that ``_fire`` injected but that has not yet
                    # surfaced, then re-raise the timeout deterministically as a
                    # TimeoutError so it never escapes as a bare interrupt.
                    try:
                        # A no-op statement gives a pending async exception a
                        # bytecode boundary to surface at, under our control.
                        for _ in range(1):
                            pass
                    except KeyboardInterrupt:
                        pass
                    raise TimeoutError(f"Execution exceeded {timeout} s time limit")

    def execute(self, source: str | None) -> tuple[str, str]:
        """Execute a block of source code, capturing stdout and stderr.

        Args:
            source: Raw python command string.

        Returns:
            (stdout_output, stderr_output)
        """
        if source is None:
            raise ValueError("source must be provided")
        if not source.strip():
            return "", ""

        out_buf = io.StringIO()
        err_buf = io.StringIO()

        # Defense-in-depth: reject attribute-introspection escape gadgets
        # before the source ever reaches compile/exec.  Reported like a
        # user-visible error so the console does not crash the host app.
        try:
            _screen_source_for_escapes(source)
        except SecurityError as e:
            err_buf.write(f"SecurityError: {e}\n")
            return out_buf.getvalue(), err_buf.getvalue()

        try:
            with (
                contextlib.redirect_stdout(out_buf),
                contextlib.redirect_stderr(err_buf),
                self._timeout_context(),
            ):
                # Try to evaluate as an expression first for REPL output behavior
                try:
                    code_obj = builtins.compile(source, "<console>", "eval")
                    res = builtins.eval(code_obj, self.namespace)  # nosec B307
                    if res is not None:
                        sys.stdout.write(repr(res) + "\n")
                except SyntaxError:
                    # Not an expression, try exec
                    code_obj = builtins.compile(source, "<console>", "exec")
                    builtins.exec(code_obj, self.namespace)  # nosec B102

        except (KeyboardInterrupt, SystemExit):
            raise
        except TimeoutError as e:
            err_buf.write(f"TimeoutError: {e}\n")
        except USER_CODE_ERROR_TYPES:
            # Expected user-code failures are formatted REPL-style so the
            # console displays them without crashing the host application.
            exc_type, exc_value, exc_traceback = sys.exc_info()
            if exc_traceback:
                # Skip the context wrapper internal frames
                tb_lines = traceback.format_exception(
                    exc_type, exc_value, exc_traceback
                )
                # Keep the last portion regarding user code
                err_buf.write("".join(tb_lines[-3:]))
            elif exc_type:
                err_buf.write(f"{exc_type.__name__}: {exc_value}\n")

        return out_buf.getvalue(), err_buf.getvalue()
