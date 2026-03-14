"""Safe mathematical expression evaluator.

Replaces all uses of ``eval()`` with a hardened AST-based evaluator that:

1. Parses the expression into an AST and validates every node type.
2. Compiles the validated AST into a code object (no raw string eval).
3. Executes the compiled code in a namespace that contains **only** the
   caller-supplied names -- ``__builtins__`` is always empty.

This eliminates the class of attacks where ``eval()`` can be abused even
when ``__builtins__`` is set to ``{}``.

Design-by-Contract
-------------------
* **Precondition**: ``expression`` is a non-empty string; ``namespace`` keys
  are all plain identifiers.
* **Postcondition**: the return value is whatever the compiled expression
  produces; no side-effects outside ``namespace``.
* **Invariant**: only the node types listed in ``_ALLOWED_NODE_TYPES`` will
  ever be executed.
"""

from __future__ import annotations

import ast
import math
from typing import Any

import numpy as np

# Conditional import: contracts module may or may not be importable depending on PYTHONPATH.
# The try/except pattern is intentional — noqa suppresses isort on this block.
try:
    from src.shared.python.contracts import require  # noqa: I001
except ImportError:
    try:
        from contracts import require  # type: ignore[no-redef]  # noqa: I001
    except ImportError:

        def require(condition: bool, *args: object) -> None:  # type: ignore[misc]
            """Fallback require() when contracts module is unavailable."""
            if not condition:
                msg = args[0] if args else "Precondition violated"
                raise AssertionError(msg)


__all__ = [
    "safe_eval",
    "safe_eval_math",
    "validate_expression",
    "NUMPY_MATH_NAMESPACE",
    "SCALAR_MATH_NAMESPACE",
]

# ── Allowed AST node types ──────────────────────────────────────────────
# These are the *only* node kinds we permit.  Anything else (Import,
# FunctionDef, Attribute access, etc.) is rejected.

_ALLOWED_NODE_TYPES: tuple[type, ...] = (
    ast.Expression,
    ast.Load,
    # Arithmetic / logic
    ast.BinOp,
    ast.UnaryOp,
    ast.operator,
    ast.unaryop,
    ast.cmpop,
    ast.Compare,
    ast.BoolOp,
    ast.boolop,
    # Literals & names
    ast.Constant,
    ast.Name,
    # Function calls (only bare-name calls, no attribute calls)
    ast.Call,
    ast.keyword,
    # Subscript / slice (for array indexing)
    ast.Subscript,
    ast.Index,  # kept for Python 3.8 compat
    ast.Slice,
    # Starred args (e.g. f(*x))
    ast.Starred,
    # IfExp (ternary)
    ast.IfExp,
)


# ── Pre-built namespaces ────────────────────────────────────────────────

NUMPY_MATH_NAMESPACE: dict[str, Any] = {
    # Standard functions (numpy versions for array support)
    "abs": np.abs,
    "min": np.minimum,
    "max": np.maximum,
    "sum": np.sum,
    "len": len,
    "round": np.round,
    # Trigonometric
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    # Exponential / logarithmic
    "sqrt": np.sqrt,
    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "pow": np.power,
    # Statistical
    "mean": np.mean,
    "std": np.std,
    "median": np.median,
    # Constants
    "pi": np.pi,
    "e": np.e,
    # np-prefixed aliases
    "np_sqrt": np.sqrt,
    "np_log": np.log,
    "np_log10": np.log10,
    "np_exp": np.exp,
    "np_sin": np.sin,
    "np_cos": np.cos,
    "np_tan": np.tan,
    "np_abs": np.abs,
    "np_mean": np.mean,
    "np_std": np.std,
    "np_min": np.min,
    "np_max": np.max,
}

SCALAR_MATH_NAMESPACE: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pow": pow,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
    "math": math,
}


# ── Core functions ──────────────────────────────────────────────────────


def validate_expression(
    expression: str,
    allowed_names: set[str] | None = None,
) -> ast.Expression:
    """Parse *expression* and validate every AST node.

    Parameters
    ----------
    expression:
        The math expression to validate.
    allowed_names:
        If provided, every ``ast.Name`` node must reference a name in this
        set.  Pass ``None`` to skip name checking (the caller is
        responsible for controlling the execution namespace).

    Returns
    -------
    ast.Expression
        The validated AST, ready to be compiled.

    Raises
    ------
    ValueError
        If the expression contains disallowed constructs.
    """
    if not expression or not expression.strip():
        raise ValueError("Expression must not be empty")

    require(
        isinstance(expression, str),
        "expression must be a string",
        type(expression).__name__,
    )

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid syntax: {exc}") from exc

    for node in ast.walk(tree):
        # Check node type is allowed
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise ValueError(f"Unsafe operation detected: {type(node).__name__}")

        # Validate names
        if isinstance(node, ast.Name):
            if allowed_names is not None and node.id not in allowed_names:
                raise ValueError(f"Unknown variable or function: {node.id}")

        # Only bare-name function calls allowed (no attribute calls like
        # os.system)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if allowed_names is not None and node.func.id not in allowed_names:
                    raise ValueError(f"Unknown function: {node.func.id}")
            else:
                raise ValueError("Attribute-based function calls not allowed")

    return tree


def safe_eval(
    expression: str,
    namespace: dict[str, Any],
    *,
    allowed_names: set[str] | None = None,
) -> Any:
    """Evaluate *expression* safely in *namespace*.

    Parameters
    ----------
    expression:
        Mathematical expression to evaluate.
    namespace:
        Dict of names the expression may reference (variables, functions,
        constants).  ``__builtins__`` is always forced to ``{}``.
    allowed_names:
        Optional explicit allowlist.  Defaults to ``namespace.keys()``.

    Returns
    -------
    Any
        Result of the expression evaluation.
    """
    assert expression is not None, "expression must be provided"
    if allowed_names is None:
        allowed_names = set(namespace.keys())

    tree = validate_expression(expression, allowed_names)
    code = compile(tree, "<safe_eval>", "eval")
    return eval(code, {"__builtins__": {}}, namespace)  # noqa: S307


def safe_eval_math(
    expression: str,
    variables: dict[str, Any] | None = None,
    *,
    use_numpy: bool = True,
) -> Any:
    """Convenience wrapper that merges caller variables with math functions.

    Parameters
    ----------
    expression:
        Mathematical expression.
    variables:
        Caller-supplied variables (signal data, parameters, etc.).
    use_numpy:
        If True, use numpy math functions (array-safe).  Otherwise use
        scalar ``math`` module functions.
    """
    assert expression is not None, "expression must be provided"
    base = dict(NUMPY_MATH_NAMESPACE if use_numpy else SCALAR_MATH_NAMESPACE)
    if variables:
        base.update(variables)
    return safe_eval(expression, base)
