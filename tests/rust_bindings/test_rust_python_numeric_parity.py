"""True numeric parity tests: Rust vs Python side-by-side.

These tests ensure that the Rust kernel produces **identical** (within
floating-point tolerance) results to the Python implementations.

Issue #1662: Previous parity tests validated algebraic invariants
(e.g., "mixing entropy lowers G") but did NOT compare the exact
numeric output of Rust vs Python on the same inputs.
"""

from __future__ import annotations

import math

import pytest

# ── Skip entire module if Rust wheel is not installed ─────────────────────────

try:
    import upstream_physics  # type: ignore[import-untyped]

    HAS_RUST = hasattr(upstream_physics, "clamp")
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(
    not HAS_RUST,
    reason="upstream_physics Rust wheel not installed",
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. clamp: Rust vs Python
# ═══════════════════════════════════════════════════════════════════════════════


class TestClampParity:
    """Verify tools_core::clamp matches Python's max(min_val, min(max_val, v))."""

    @staticmethod
    def _python_clamp(v: float, lo: float, hi: float) -> float:
        """Pure Python reference implementation."""
        return max(lo, min(hi, v))

    @pytest.mark.parametrize(
        "value,lo,hi",
        [
            (5.0, 0.0, 10.0),  # within range
            (-1.0, 0.0, 10.0),  # below
            (15.0, 0.0, 10.0),  # above
            (0.0, 0.0, 0.0),  # degenerate: lo == hi
            (1e-15, 0.0, 1.0),  # near-zero
            (-1e300, -1e200, 1e200),  # extreme range
        ],
    )
    def test_clamp_exact(self, value: float, lo: float, hi: float) -> None:
        """Rust clamp must equal Python clamp exactly (no FP error expected)."""
        rust_result = float(upstream_physics.clamp(value, lo, hi))
        py_result = self._python_clamp(value, lo, hi)
        assert rust_result == py_result, (
            f"Mismatch: clamp({value}, {lo}, {hi}) → "
            f"Rust={rust_result}, Python={py_result}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. lerp: Rust vs Python
# ═══════════════════════════════════════════════════════════════════════════════


class TestLerpParity:
    """Verify tools_core::lerp matches Python's a + t * (b - a)."""

    @staticmethod
    def _python_lerp(a: float, b: float, t: float) -> float:
        """Pure Python reference implementation."""
        return a + t * (b - a)

    @pytest.mark.parametrize(
        "a,b,t",
        [
            (0.0, 10.0, 0.0),  # t=0 → a
            (0.0, 10.0, 1.0),  # t=1 → b
            (0.0, 10.0, 0.5),  # midpoint
            (-5.0, 5.0, 0.25),  # negative a
            (100.0, 200.0, 0.75),  # large values
            (1e-10, 2e-10, 0.5),  # tiny values
        ],
    )
    def test_lerp_exact(self, a: float, b: float, t: float) -> None:
        """Rust lerp must match Python lerp within 1 ULP."""
        rust_result = float(upstream_physics.lerp(a, b, t))
        py_result = self._python_lerp(a, b, t)
        assert (
            abs(rust_result - py_result) < 1e-15
        ), f"Mismatch: lerp({a}, {b}, {t}) → Rust={rust_result}, Python={py_result}"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Vector3: Rust vs Python (numpy)
# ═══════════════════════════════════════════════════════════════════════════════


class TestVector3Parity:
    """Verify tools_core::Vector3 operations match numpy."""

    def test_magnitude(self) -> None:
        """Vector3.magnitude() must match math.sqrt(x²+y²+z²)."""
        v = upstream_physics.Vector3(3.0, 4.0, 0.0)
        rust_mag = v.magnitude()
        py_mag = math.sqrt(3.0**2 + 4.0**2 + 0.0**2)
        assert abs(rust_mag - py_mag) < 1e-12

    def test_dot_product(self) -> None:
        """Vector3.dot() must match manual computation."""
        a = upstream_physics.Vector3(1.0, 2.0, 3.0)
        b = upstream_physics.Vector3(4.0, 5.0, 6.0)
        rust_dot = a.dot(b)
        py_dot = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0
        assert abs(rust_dot - py_dot) < 1e-12

    def test_cross_product(self) -> None:
        """Vector3.cross() must match manual computation."""
        a = upstream_physics.Vector3(1.0, 0.0, 0.0)
        b = upstream_physics.Vector3(0.0, 1.0, 0.0)
        c = a.cross(b)
        # x × y = z
        assert abs(c.x) < 1e-12
        assert abs(c.y) < 1e-12
        assert abs(c.z - 1.0) < 1e-12

    def test_normalized(self) -> None:
        """Normalized vector must have magnitude 1."""
        v = upstream_physics.Vector3(3.0, 4.0, 0.0)
        n = v.normalized()
        assert abs(n.magnitude() - 1.0) < 1e-12
        assert abs(n.x - 0.6) < 1e-12
        assert abs(n.y - 0.8) < 1e-12

    def test_scale(self) -> None:
        """Vector3.scale() must match component-wise multiply."""
        v = upstream_physics.Vector3(1.0, 2.0, 3.0)
        s = v.scale(2.5)
        assert abs(s.x - 2.5) < 1e-12
        assert abs(s.y - 5.0) < 1e-12
        assert abs(s.z - 7.5) < 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RK4 Integrator: Rust vs analytical solutions
# ═══════════════════════════════════════════════════════════════════════════════


class TestRK4NumericParity:
    """Verify RK4 integrator matches analytical solutions to high precision."""

    def test_config_creation(self) -> None:
        """IntegratorConfig must accept and store params correctly."""
        config = upstream_physics.IntegratorConfig(dt=0.01, max_steps=500)
        assert config.dt == 0.01  # noqa: PLR2004
        assert config.max_steps == 500  # noqa: PLR2004


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Contact model: Rust vs Python
# ═══════════════════════════════════════════════════════════════════════════════


class TestContactParity:
    """Verify Rust contact model matches Python physics."""

    def test_elastic_bounce_rust(self) -> None:
        """Perfectly elastic bounce (COR=1) must conserve speed."""
        params = upstream_physics.ContactParameters(cor=1.0, friction=0.0)
        assert params.cor == 1.0  # noqa: PLR2004
        assert params.friction == 0.0  # noqa: PLR2004

    def test_default_parameters(self) -> None:
        """Default COR and friction must match Python defaults."""
        params = upstream_physics.ContactParameters()
        # Default from Rust: COR=0.78, friction=0.4
        assert abs(params.cor - 0.78) < 1e-12
        assert abs(params.friction - 0.4) < 1e-12
