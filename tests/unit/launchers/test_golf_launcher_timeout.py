"""Tests for golf_launcher startup timeout and create_model_card — issue #5488.

RED → GREEN cycle:
  - GolfLauncher must have _handle_startup_timeout method
  - create_model_card must either be implemented or raise NotImplementedError
  - Timeout handler must not raise, must log an error
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestStartupTimeoutHandler:
    """#5488 — GolfLauncher must guard against infinite startup wait."""

    def test_startup_timeout_handler_exists(self) -> None:
        """GolfLauncher must have a _handle_startup_timeout method."""
        from src.launchers.golf_launcher import GolfLauncher

        assert hasattr(GolfLauncher, "_handle_startup_timeout"), (
            "GolfLauncher is missing _handle_startup_timeout method"
        )

    def test_startup_timeout_handler_is_callable(self) -> None:
        """_handle_startup_timeout must be callable."""
        from src.launchers.golf_launcher import GolfLauncher

        assert callable(getattr(GolfLauncher, "_handle_startup_timeout", None))

    def test_startup_timeout_does_not_raise(self) -> None:
        """Calling _handle_startup_timeout on a mock launcher must not raise."""
        from src.launchers.golf_launcher import GolfLauncher

        # Build a minimal mock that satisfies the method's requirements
        mock_self = MagicMock(spec=GolfLauncher)
        mock_self.loading = True

        # Should not raise
        GolfLauncher._handle_startup_timeout(mock_self)

    def test_startup_timeout_clears_loading_flag(self) -> None:
        """_handle_startup_timeout must clear the loading flag."""
        from src.launchers.golf_launcher import GolfLauncher

        mock_self = MagicMock(spec=GolfLauncher)
        mock_self.loading = True

        GolfLauncher._handle_startup_timeout(mock_self)

        assert mock_self.loading is False, (
            "_handle_startup_timeout did not clear the loading flag"
        )

    def test_startup_timeout_shows_error_message(self) -> None:
        """_handle_startup_timeout must surface a user-visible error."""
        from src.launchers.golf_launcher import GolfLauncher

        mock_self = MagicMock(spec=GolfLauncher)
        mock_self.loading = True

        with patch("src.launchers.golf_launcher.logger") as mock_logger:
            GolfLauncher._handle_startup_timeout(mock_self)
            # Must log an error-level message
            assert mock_logger.error.called or mock_logger.warning.called, (
                "_handle_startup_timeout did not log any error/warning"
            )


class TestCreateModelCard:
    """#5488 — create_model_card must not be a silent no-op placeholder."""

    def test_create_model_card_not_empty_placeholder(self) -> None:
        """create_model_card must either be implemented or raise NotImplementedError.

        A silent no-op (body is just 'pass') is not acceptable as it silently
        discards work; the method must signal that it needs a model argument or
        raise if it cannot be called without further context.
        """
        import ast
        import inspect
        import textwrap

        from src.launchers.golf_launcher import GolfLauncher

        raw_src = inspect.getsource(GolfLauncher.create_model_card)
        # dedent so ast.parse can handle the indented source
        src = textwrap.dedent(raw_src)
        tree = ast.parse(src)

        # Find the function body
        func_def = tree.body[0]
        # Acceptable bodies: raises NotImplementedError, or has real statements
        # Unacceptable: only a docstring + pass (or just pass)
        body_stmts = func_def.body  # type: ignore[attr-defined]

        # Filter out docstrings (Expr nodes with string constants)
        non_doc_stmts = [
            s for s in body_stmts
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
        ]

        # If the non-docstring body is empty (just `pass`) or only `pass`, fail
        is_only_pass = all(isinstance(s, ast.Pass) for s in non_doc_stmts) or (
            len(non_doc_stmts) == 0
        )
        assert not is_only_pass, (
            "create_model_card is still an empty placeholder (body is `pass`). "
            "Either implement it or raise NotImplementedError."
        )
