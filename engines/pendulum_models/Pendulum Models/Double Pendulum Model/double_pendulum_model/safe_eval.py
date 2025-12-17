"""
Safe evaluation of user-provided mathematical expressions.
"""

import ast
import math
import typing
from types import CodeType


class SafeEvaluator:
    """Safe evaluation of user-provided expressions using AST whitelisting."""

    _ALLOWED_NODES: typing.ClassVar[set[type[ast.AST]]] = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Name,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Call,
        ast.Constant,
        # ast.BitXor is excluded to prevent confusion with exponentiation (^)
        # ast.Attribute is excluded to prevent access to object internals
    }
    """
    Allowlist of AST node types permitted in user expressions.

    - ast.Expression: Root node for 'eval' mode.
    - ast.BinOp, ast.UnaryOp: Basic arithmetic structure.
    - ast.Name, ast.Load: Safe variable and function access 
      (validated against allowed lists).
    - ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod: Arithmetic operators.
    - ast.USub, ast.UAdd: Unary operators (negation).
    - ast.Call: Function calls (strictly validated against allowed functions).
    - ast.Constant: Literal values (numbers).

    Explicitly Excluded:
    - ast.Attribute: Prevents object attribute access (e.g. `obj.__class__`).
    - ast.BitXor: Prevents `^` operator, often mistaken for exponentiation in math.
    - ast.Import, ast.ImportFrom: Prevents module loading.
    - ast.Lambda, ast.FunctionDef: Prevents function definition.
    - ast.Subscript: Prevents indexing/slicing (e.g. `"".__class__.__mro__[1]`).
    """

    _ALLOWED_MATH_NAMES: typing.ClassVar[dict[str, typing.Any]] = {
        name: getattr(math, name)
        for name in (
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "atan2",
            "sqrt",
            "log",
            "log10",
            "exp",
            "pi",
            "tau",
            "fabs",
        )
    }

    def __init__(self, allowed_variables: set[str] | None = None) -> None:
        """Initialize the SafeEvaluator with a set of allowed variable names."""
        self.allowed_variables = allowed_variables or set()
        self.allowed_names = {**self._ALLOWED_MATH_NAMES}

    def validate(self, expression: str) -> ast.AST:  # noqa: C901, PLR0912
        """Parses and validates the expression against the allowlist."""
        try:
            parsed = ast.parse(expression.strip(), mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid syntax: {e}") from e

        for node in ast.walk(parsed):
            if isinstance(node, ast.BitXor):
                raise ValueError("Use '**' for exponentiation instead of '^'")

            if type(node) not in self._ALLOWED_NODES:
                raise ValueError(
                    f"Disallowed syntax in expression: {type(node).__name__}"
                )

            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Load):
                    if (
                        node.id not in self.allowed_variables
                        and node.id not in self.allowed_names
                    ):
                        raise ValueError(f"Unknown variable '{node.id}'")
                elif not isinstance(node.ctx, ast.Load):
                    # Should be caught by node type check
                    # (only Load is in ALLOWED_NODES usually)
                    # But explicitly:
                    raise ValueError("Variable assignment/store is not allowed")

            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise ValueError("Only direct function calls are permitted")
                if node.func.id not in self.allowed_names:
                    raise ValueError(f"Function '{node.func.id}' is not permitted")

        return parsed

    def compile(self, expression: str) -> CodeType:
        """Validates and compiles the expression."""
        parsed = self.validate(expression)
        return typing.cast(
            "CodeType",
            compile(parsed, filename="<SafeEvaluator>", mode="eval"),  # type: ignore[call-overload]
        )

    def evaluate_code(
        self, code: CodeType, context: dict[str, float] | None = None
    ) -> float:
        """Evaluates compiled code with the given context."""
        # Start with context, but override with allowed math names to prevent shadowing
        # This addresses the risk of context={'sin': malicious_func}
        eval_context: dict[str, typing.Any] = context.copy() if context else {}
        eval_context.update(self.allowed_names)

        # Execute with empty locals, everything in globals
        # Defense in depth: pass eval_context as globals.
        # Explicitly remove __builtins__ and set it to empty dict.
        eval_context.pop("__builtins__", None)
        eval_context["__builtins__"] = {}

        return float(eval(code, eval_context))

    def evaluate(
        self, expression: str, context: dict[str, float] | None = None
    ) -> float:
        """Evaluates the expression with the given context."""
        code = self.compile(expression)
        return self.evaluate_code(code, context)
