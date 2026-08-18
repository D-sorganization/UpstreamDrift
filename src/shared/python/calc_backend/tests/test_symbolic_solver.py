"""Tests for the symbolic solver router with optional SymPy support."""

from __future__ import annotations

import pytest
from calc_backend.app import app
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the symbolic solver endpoints."""
    return TestClient(app)


class TestSymbolicSolverHelp:
    """Tests for the symbolic solver help endpoint."""

    def test_help_endpoint_returns_metadata(self, client: TestClient) -> None:
        """Test that the help endpoint returns supported operations."""
        response = client.get("/api/calc/symbolic/help")
        assert response.status_code == 200
        data = response.json()
        assert "supported_operations" in data
        assert "examples" in data
        assert "available" in data


class TestSymbolicSolve:
    """Tests for the symbolic equation solving endpoint."""

    def test_solve_returns_unavailable_when_sympy_missing(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that solve returns unavailable response when SymPy is missing."""
        import calc_backend.routers.symbolic_solver as mod

        monkeypatch.setattr(mod, "SYMPY_AVAILABLE", False)
        try:
            response = client.post(
                "/api/calc/symbolic/solve",
                json={"equation": "x**2 - 4 = 0", "variable": "x"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["available"] is False
            assert "SymPy is not available" in data["error"]
        finally:
            monkeypatch.setattr(mod, "SYMPY_AVAILABLE", True)

    def test_solve_quadratic_equation(self, client: TestClient) -> None:
        """Test solving a quadratic equation."""
        response = client.post(
            "/api/calc/symbolic/solve",
            json={"equation": "x**2 - 4 = 0", "variable": "x"},
        )
        data = response.json()
        if data.get("available"):
            assert response.status_code == 200
            assert "-2" in data["solutions"] or "2" in data["solutions"]
        else:
            assert response.status_code == 200
            assert data["available"] is False

    def test_solve_requires_variable(self, client: TestClient) -> None:
        """Test that solve endpoint validates required fields."""
        response = client.post(
            "/api/calc/symbolic/solve",
            json={"equation": "x**2 - 4 = 0"},
        )
        assert response.status_code == 422


class TestSymbolicDerivative:
    """Tests for the symbolic derivative endpoint."""

    def test_derivative_returns_unavailable_when_sympy_missing(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that derivative returns unavailable response when SymPy is missing."""
        import calc_backend.routers.symbolic_solver as mod

        monkeypatch.setattr(mod, "SYMPY_AVAILABLE", False)
        try:
            response = client.post(
                "/api/calc/symbolic/derivative",
                json={"expression": "x**3", "variable": "x"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["available"] is False
            assert "SymPy is not available" in data["error"]
        finally:
            monkeypatch.setattr(mod, "SYMPY_AVAILABLE", True)

    def test_derivative_polynomial(self, client: TestClient) -> None:
        """Test computing derivative of a polynomial."""
        response = client.post(
            "/api/calc/symbolic/derivative",
            json={"expression": "x**3 + 2*x", "variable": "x"},
        )
        data = response.json()
        if data.get("available"):
            assert response.status_code == 200
            assert "3*x**2" in data["derivative"] or "3*x^2" in data["derivative"]
        else:
            assert response.status_code == 200
            assert data["available"] is False


class TestSymbolicSimplify:
    """Tests for the symbolic simplify endpoint."""

    def test_simplify_returns_unavailable_when_sympy_missing(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that simplify returns unavailable response when SymPy is missing."""
        import calc_backend.routers.symbolic_solver as mod

        monkeypatch.setattr(mod, "SYMPY_AVAILABLE", False)
        try:
            response = client.post(
                "/api/calc/symbolic/simplify",
                json={"expression": "(x**2 - 1)/(x - 1)"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["available"] is False
            assert "SymPy is not available" in data["error"]
        finally:
            monkeypatch.setattr(mod, "SYMPY_AVAILABLE", True)

    def test_simplify_rational(self, client: TestClient) -> None:
        """Test simplifying a rational expression."""
        response = client.post(
            "/api/calc/symbolic/simplify",
            json={"expression": "(x**2 - 1)/(x - 1)"},
        )
        data = response.json()
        if data.get("available"):
            assert response.status_code == 200
            # (x^2 - 1)/(x - 1) = x + 1
            assert "x" in str(data.get("simplified", ""))
        else:
            assert response.status_code == 200
            assert data["available"] is False


# ── Input validation (issue #8675) ──────────────────────────────────────
# ``sympy.parse_expr`` is an *evaluating* parser: without a guard, an
# attribute-based call such as ``__import__("os").getcwd()`` is executed by
# the parser itself. These endpoints are remote by construction, so every
# ``parse_expr`` call site must be preceded by ``validate_expression``.

#: Payload that ``parse_expr`` would otherwise execute (arbitrary import +
#: attribute call). ``validate_expression`` rejects it as an
#: "Attribute-based function call".
ATTRIBUTE_CALL_PAYLOAD = '__import__("os").getcwd()'

#: Payload rejected by the node-type allowlist (``ast.Attribute``).
ATTRIBUTE_ACCESS_PAYLOAD = "x.__class__"

#: Payload rejected by the exponentiation-bomb bound (MAX_POW_EXPONENT).
POW_BOMB_PAYLOAD = "9**9**9**9"


class TestSymbolicSolverRejectsUnsafeInput:
    """Every endpoint must refuse expressions the validator blocks."""

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (ATTRIBUTE_CALL_PAYLOAD, "Attribute-based function calls not allowed"),
            (ATTRIBUTE_ACCESS_PAYLOAD, "Unsafe operation detected: Attribute"),
            (POW_BOMB_PAYLOAD, "Exponent too large"),
        ],
    )
    def test_solve_rejects_unsafe_bare_equation(
        self, client: TestClient, payload: str, expected: str
    ) -> None:
        """Solve rejects unsafe input on the no-``=`` (bare expression) branch."""
        response = client.post(
            "/api/calc/symbolic/solve",
            json={"equation": payload, "variable": "x"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["error"] is not None, f"{payload!r} was not rejected"
        assert expected in data["error"]
        assert not data["solutions"]

    @pytest.mark.parametrize(
        "equation",
        [
            f"{ATTRIBUTE_CALL_PAYLOAD} = 0",  # unsafe on the left-hand side
            f"0 = {ATTRIBUTE_CALL_PAYLOAD}",  # unsafe on the right-hand side
        ],
    )
    def test_solve_rejects_unsafe_either_side_of_equals(
        self, client: TestClient, equation: str
    ) -> None:
        """Solve validates *both* sides of an ``lhs = rhs`` equation."""
        response = client.post(
            "/api/calc/symbolic/solve",
            json={"equation": equation, "variable": "x"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["error"] is not None, f"{equation!r} was not rejected"
        assert "Attribute-based function calls not allowed" in data["error"]
        assert not data["solutions"]

    def test_derivative_rejects_unsafe_expression(self, client: TestClient) -> None:
        """Derivative rejects an attribute-based call payload."""
        response = client.post(
            "/api/calc/symbolic/derivative",
            json={"expression": ATTRIBUTE_CALL_PAYLOAD, "variable": "x"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["error"] is not None
        assert "Attribute-based function calls not allowed" in data["error"]
        assert data["derivative"] is None

    def test_simplify_rejects_unsafe_expression(self, client: TestClient) -> None:
        """Simplify rejects an attribute-based call payload."""
        response = client.post(
            "/api/calc/symbolic/simplify",
            json={"expression": ATTRIBUTE_CALL_PAYLOAD},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["error"] is not None
        assert "Attribute-based function calls not allowed" in data["error"]
        assert data["simplified"] is None

    def test_unsafe_payload_is_never_evaluated(self, client: TestClient) -> None:
        """The rejected payload must not reach the evaluating parser.

        ``__import__("os").getcwd()`` returns the server's working directory
        when it is actually executed, so leaking that string into the response
        is direct evidence the guard was bypassed.
        """
        import os

        # Compare against the *parsed* body: on Windows the raw response text
        # JSON-escapes backslashes, which would hide a leaked path. The leaf
        # directory name contains no separators and survives either way.
        cwd = os.getcwd()
        leaf = os.path.basename(cwd)
        for endpoint, body in (
            (
                "/api/calc/symbolic/solve",
                {"equation": ATTRIBUTE_CALL_PAYLOAD, "variable": "x"},
            ),
            (
                "/api/calc/symbolic/derivative",
                {"expression": ATTRIBUTE_CALL_PAYLOAD, "variable": "x"},
            ),
            (
                "/api/calc/symbolic/simplify",
                {"expression": ATTRIBUTE_CALL_PAYLOAD},
            ),
        ):
            response = client.post(endpoint, json=body)
            rendered = " ".join(
                str(value) for value in response.json().values() if value is not None
            )
            for marker in (cwd, leaf):
                assert marker not in rendered, (
                    f"{endpoint} evaluated the payload and leaked "
                    f"the working directory ({marker!r})"
                )


class TestSymbolicSolverAcceptsLegitimateInput:
    """The guard must not break ordinary symbolic work."""

    def test_solve_accepts_equation_with_equals(self, client: TestClient) -> None:
        """A normal ``lhs = rhs`` equation still solves."""
        response = client.post(
            "/api/calc/symbolic/solve",
            json={"equation": "x**2 - 4 = 0", "variable": "x"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["error"] is None
        assert sorted(data["solutions"]) == ["-2", "2"]

    def test_solve_accepts_bare_expression(self, client: TestClient) -> None:
        """A bare expression (implicitly ``== 0``) still solves."""
        response = client.post(
            "/api/calc/symbolic/solve",
            json={"equation": "x**2 - 9", "variable": "x"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["error"] is None
        assert sorted(data["solutions"]) == ["-3", "3"]

    def test_solve_accepts_caret_exponent(self, client: TestClient) -> None:
        """``convert_xor`` input (``x^2``) survives validation.

        ``^`` is a legal ``ast.BinOp``/``BitXor`` to Python's parser only if
        the validator allows it; this pins the interaction between the guard
        and the ``convert_xor`` transformation.
        """
        response = client.post(
            "/api/calc/symbolic/solve",
            json={"equation": "x^2 - 4 = 0", "variable": "x"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["error"] is None
        assert sorted(data["solutions"]) == ["-2", "2"]

    def test_derivative_accepts_polynomial(self, client: TestClient) -> None:
        """A normal derivative still computes."""
        response = client.post(
            "/api/calc/symbolic/derivative",
            json={"expression": "x**3 + 2*x", "variable": "x"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["error"] is None
        assert data["derivative"] == "3*x**2 + 2"

    def test_simplify_accepts_rational(self, client: TestClient) -> None:
        """A normal simplification still runs."""
        response = client.post(
            "/api/calc/symbolic/simplify",
            json={"expression": "(x**2 - 1)/(x - 1)"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["error"] is None
        assert data["simplified"] == "x + 1"

    def test_simplify_accepts_function_call(self, client: TestClient) -> None:
        """Bare-name function calls (``sin(x)``) are still permitted."""
        response = client.post(
            "/api/calc/symbolic/simplify",
            json={"expression": "sin(x)**2 + cos(x)**2"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["error"] is None
        assert data["simplified"] == "1"


class TestParseExprIsUnreachableWithoutValidation:
    """``parse_expr`` must never run on input the validator has not cleared."""

    @staticmethod
    def _instrument(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
        """Record the interleaving of validate/parse calls on the router."""
        import calc_backend.routers.symbolic_solver as mod

        events: list[tuple[str, str]] = []
        real_validate = mod.validate_expression
        real_parse = mod.parse_expr

        def recording_validate(expression: str, *args: object, **kwargs: object):
            events.append(("validate", expression))
            return real_validate(expression, *args, **kwargs)

        def recording_parse(expression: str, *args: object, **kwargs: object):
            events.append(("parse", expression))
            return real_parse(expression, *args, **kwargs)

        monkeypatch.setattr(mod, "validate_expression", recording_validate)
        monkeypatch.setattr(mod, "parse_expr", recording_parse)
        return events

    @staticmethod
    def _assert_never_parses_unvalidated(events: list[tuple[str, str]]) -> None:
        """Assert no ``parse`` ever outruns the validations preceding it."""
        assert events, "endpoint performed no parsing at all"
        assert events[0][0] == "validate", (
            f"first operation was {events[0][0]!r}, expected 'validate'"
        )
        validated = 0
        parsed = 0
        for kind, _ in events:
            if kind == "validate":
                validated += 1
            else:
                parsed += 1
            assert parsed <= validated, (
                f"parse_expr ran ahead of validate_expression: {events}"
            )
        assert parsed == validated, (
            f"expected one validation per parse, got {validated} validations "
            f"for {parsed} parses: {events}"
        )

    def test_solve_with_equals_validates_every_parse(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both sides of an equation are validated before either is parsed."""
        events = self._instrument(monkeypatch)
        client.post(
            "/api/calc/symbolic/solve",
            json={"equation": "x**2 - 4 = 0", "variable": "x"},
        )
        self._assert_never_parses_unvalidated(events)
        assert [e[0] for e in events] == ["validate", "validate", "parse", "parse"]

    def test_solve_bare_expression_validates_every_parse(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The implicit ``== 0`` branch validates before parsing."""
        events = self._instrument(monkeypatch)
        client.post(
            "/api/calc/symbolic/solve",
            json={"equation": "x**2 - 4", "variable": "x"},
        )
        self._assert_never_parses_unvalidated(events)

    def test_derivative_validates_every_parse(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The derivative endpoint validates before parsing."""
        events = self._instrument(monkeypatch)
        client.post(
            "/api/calc/symbolic/derivative",
            json={"expression": "x**3 + 2*x", "variable": "x"},
        )
        self._assert_never_parses_unvalidated(events)

    def test_simplify_validates_every_parse(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The simplify endpoint validates before parsing."""
        events = self._instrument(monkeypatch)
        client.post(
            "/api/calc/symbolic/simplify",
            json={"expression": "(x**2 - 1)/(x - 1)"},
        )
        self._assert_never_parses_unvalidated(events)

    def test_router_imports_the_validator(self) -> None:
        """The router module must expose the shared validator (not a local stub)."""
        import calc_backend.routers.symbolic_solver as mod
        from src.shared.python.safe_eval import validate_expression

        assert mod.validate_expression is validate_expression
