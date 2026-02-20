import ast
import math
from unittest.mock import patch

from double_pendulum_model.ui.pendulum_pyqt_app import PendulumController


def test_safe_eval_valid_math() -> None:
    """Test that safe mathematical expressions are evaluated correctly."""
    assert PendulumController._safe_eval(None, "1 + 2") == 3.0
    assert PendulumController._safe_eval(None, "sin(0)") == 0.0
    assert PendulumController._safe_eval(None, "cos(0)") == 1.0
    assert PendulumController._safe_eval(None, "pi") == math.pi
    assert PendulumController._safe_eval(None, "3 * 4 + 5") == 17.0


def test_safe_eval_handles_errors() -> None:
    """Test that invalid expressions return 0.0 gracefully."""
    assert PendulumController._safe_eval(None, "invalid_syntax(") == 0.0
    assert PendulumController._safe_eval(None, "unknown_var") == 0.0


def test_safe_eval_security_blocks() -> None:
    """Test that potential security exploits are blocked."""
    # Attribute access - returns 0.0
    assert PendulumController._safe_eval(None, "pi.__class__") == 0.0

    # Builtin access
    assert PendulumController._safe_eval(None, "__import__('os')") == 0.0


def test_safe_eval_enforces_ast_validation() -> None:
    """Test that AST validation is actually performed."""
    # In the vulnerable version, ast.parse is NOT called.
    # In the fixed version, it MUST be called.
    with patch("ast.parse", side_effect=ast.parse) as mock_parse:
        PendulumController._safe_eval(None, "1+1")
        assert mock_parse.called, "ast.parse should be called to validate input"
