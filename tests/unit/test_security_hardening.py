"""Security regression tests for issue #5913.

These tests use static AST analysis to verify that security-sensitive code
patterns are present (or absent) in the relevant source files, ensuring that
future refactors do not accidentally reintroduce the vulnerabilities.

Checks:
    - WebSocket error responses never contain ``str(e)`` in the ``detail`` key.
    - ``speed_factor`` validation (``<= 0`` guard) is present in simulation_ws.py.
    - ``_MAX_PROMPT_BYTES`` cap is defined and used in bitnet_adapter.py.
"""

from __future__ import annotations

import ast
import pathlib
import re
import unittest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).parents[2]


def _source(rel: str) -> str:
    """Return the source text of a repository file."""
    return (_REPO_ROOT / rel).read_text(encoding="utf-8")


def _ast(rel: str) -> ast.Module:
    """Return the parsed AST of a repository file."""
    src = _source(rel)
    return ast.parse(src, filename=rel)


# ---------------------------------------------------------------------------
# Task 1 – WebSocket error responses must not leak str(e) details
# ---------------------------------------------------------------------------


class TestWebSocketNoExceptionLeak(unittest.TestCase):
    """Verify that WS JSON error payloads never include raw str(e) values."""

    # Pattern that would send `str(e)` as the ``detail`` field
    _LEAK_PATTERN: re.Pattern[str] = re.compile(
        r'send_json\s*\(\s*\{[^}]*"detail"\s*:\s*str\s*\(', re.DOTALL
    )

    def _assert_no_leak(self, rel_path: str) -> None:
        source = _source(rel_path)
        match = self._LEAK_PATTERN.search(source)
        self.assertIsNone(
            match,
            f"{rel_path}: found str(e) detail leak at pos {match.start() if match else '?'}",
        )

    def test_simulation_ws_no_exception_leak(self) -> None:
        self._assert_no_leak("src/api/routes/simulation_ws.py")

    def test_chat_ws_no_exception_leak(self) -> None:
        self._assert_no_leak("src/api/routes/chat_ws.py")


# ---------------------------------------------------------------------------
# Task 2 – speed_factor validation in simulation_ws.py
# ---------------------------------------------------------------------------


class TestSpeedFactorValidation(unittest.TestCase):
    """Verify that set_speed action validates speed_factor > 0."""

    def test_speed_factor_guard_present(self) -> None:
        source = _source("src/api/routes/simulation_ws.py")
        # The guard must check that speed_factor > 0 (i.e. 'speed_factor <= 0')
        self.assertIn(
            "speed_factor <= 0",
            source,
            "simulation_ws.py must contain a 'speed_factor <= 0' guard to prevent "
            "division-by-zero in _compute_real_time_sleep_delay.",
        )

    def test_speed_factor_isinstance_check_present(self) -> None:
        source = _source("src/api/routes/simulation_ws.py")
        # isinstance guard must co-exist with the <= 0 check
        self.assertIn(
            "isinstance(speed_factor, (int, float))",
            source,
            "simulation_ws.py must validate speed_factor type before use.",
        )


# ---------------------------------------------------------------------------
# Task 3 – Prompt size cap in bitnet_adapter.py
# ---------------------------------------------------------------------------


class TestBitnetPromptSizeCap(unittest.TestCase):
    """Verify that _MAX_PROMPT_BYTES is defined and used in bitnet_adapter.py."""

    _REL = "src/shared/python/ai/adapters/bitnet_adapter.py"

    def test_constant_defined(self) -> None:
        source = _source(self._REL)
        self.assertIn(
            "_MAX_PROMPT_BYTES",
            source,
            "bitnet_adapter.py must define _MAX_PROMPT_BYTES constant.",
        )

    def test_constant_is_int_assignment(self) -> None:
        """_MAX_PROMPT_BYTES must be an integer literal (not 0, not negative)."""
        tree = _ast(self._REL)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "_MAX_PROMPT_BYTES"
            ):
                # The value must be a positive integer literal
                self.assertIsNotNone(node.value, "_MAX_PROMPT_BYTES must have a value")
                assert node.value is not None  # mypy narrowing
                self.assertIsInstance(
                    node.value, ast.Constant, "_MAX_PROMPT_BYTES must be a literal"
                )
                self.assertIsInstance(
                    node.value.value, int, "_MAX_PROMPT_BYTES must be an int"
                )
                self.assertGreater(
                    node.value.value, 0, "_MAX_PROMPT_BYTES must be positive"
                )
                return
        self.fail(
            "_MAX_PROMPT_BYTES annotated assignment not found in bitnet_adapter.py"
        )

    def test_prompt_size_check_present_in_send_message(self) -> None:
        source = _source(self._REL)
        # The UTF-8 size check must appear in the file
        self.assertIn(
            'prompt.encode("utf-8")',
            source,
            "bitnet_adapter.py must check prompt byte length via encode('utf-8').",
        )

    def test_prompt_size_check_raises_value_error(self) -> None:
        source = _source(self._REL)
        # The check must raise ValueError (not just log)
        self.assertIn(
            "raise ValueError",
            source,
            "bitnet_adapter.py must raise ValueError when prompt exceeds size cap.",
        )


if __name__ == "__main__":
    unittest.main()
