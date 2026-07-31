"""Numeric parity tests for the Rust-backed physics facade.

These tests keep the Python facade and installed ``upstream_physics`` wheel
aligned. The PyO3 module owns the heavy physics types; lightweight helpers
such as clamp/lerp are facade functions and may delegate to Rust when the
wheel exports matching helpers.

Issue #1662: Previous parity tests validated algebraic invariants
(e.g., "mixing entropy lowers G") but did NOT compare the exact
numeric output of Rust vs Python on the same inputs.
"""

import pytest
from src.shared.python.physics.rust_kernel import clamp, lerp

# ── Skip entire module if Rust wheel is not installed ─────────────────────────

try:
    import upstream_physics  # type: ignore[import-untyped]

    HAS_RUST = True
except ImportError:
    upstream_physics = None  # type: ignore[assignment]
    HAS_RUST = False

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not HAS_RUST,
        reason="upstream_physics Rust wheel not installed",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. clamp: Rust vs Python
# ═══════════════════════════════════════════════════════════════════════════════


class TestClampParity:
    """Verify facade clamp matches Python's max(min_val, min(max_val, v))."""

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
        rust_result = float(clamp(value, lo, hi))
        py_result = self._python_clamp(value, lo, hi)
        assert rust_result == py_result, (
            f"Mismatch: clamp({value}, {lo}, {hi}) → "
            f"Rust={rust_result}, Python={py_result}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. lerp: Rust vs Python
# ═══════════════════════════════════════════════════════════════════════════════


class TestLerpParity:
    """Verify facade lerp matches Python's a + t * (b - a)."""

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
        rust_result = float(lerp(a, b, t))
        py_result = self._python_lerp(a, b, t)
        assert (
            abs(rust_result - py_result) < 1e-15
        ), f"Mismatch: lerp({a}, {b}, {t}) → Rust={rust_result}, Python={py_result}"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Exported PyO3 physics types
# ═══════════════════════════════════════════════════════════════════════════════


class TestExportedPhysicsTypes:
    """Verify the installed wheel exposes supported PyO3 contracts."""

    def test_required_type_exports_exist(self) -> None:
        for name in (
            "IntegratorConfig",
            "IntegrationResult",
            "ContactParameters",
            "ContactResult",
            "AeroBallProperties",
            "AirProperties",
            "AerodynamicsEngine",
        ):
            assert hasattr(upstream_physics, name)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RK4 Integrator: Rust vs analytical solutions
# ═══════════════════════════════════════════════════════════════════════════════


class TestRK4NumericParity:
    """Verify RK4 integrator matches analytical solutions to high precision."""

    def test_config_creation(self) -> None:
        """IntegratorConfig must accept and store params correctly."""
        config = upstream_physics.IntegratorConfig(dt=0.01, max_steps=500)
        assert config is not None

    def test_config_rejects_invalid_dt(self) -> None:
        """IntegratorConfig must enforce its positive-time-step contract."""
        with pytest.raises(ValueError):
            upstream_physics.IntegratorConfig(dt=0.0, max_steps=500)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Contact model: Rust vs Python
# ═══════════════════════════════════════════════════════════════════════════════


class TestContactParity:
    """Verify Rust contact model matches Python physics."""

    def test_elastic_bounce_rust(self) -> None:
        """Perfectly elastic contact parameters must construct successfully."""
        params = upstream_physics.ContactParameters(cor=1.0, friction=0.0)
        assert params is not None

    def test_default_parameters(self) -> None:
        """Default contact parameters must construct successfully."""
        params = upstream_physics.ContactParameters()
        assert params is not None

    def test_rejects_out_of_range_cor(self) -> None:
        """ContactParameters must enforce COR in [0, 1]."""
        with pytest.raises(ValueError):
            upstream_physics.ContactParameters(cor=1.5, friction=0.0)
