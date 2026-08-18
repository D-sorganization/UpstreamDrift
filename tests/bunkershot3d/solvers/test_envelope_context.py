"""Invariants of :class:`EnvelopeContext` (#8607 gate decomposition).

``EnvelopeContext`` was introduced to bring ``evaluate_envelope`` inside the
8-parameter architecture budget, but it did more than shorten a signature: it
put two invariants on a constructor that previously rode all the way onto a
public value object unchecked.

Both failure modes were silent before:

- a ``clamped_area_fraction`` outside ``[0, 1]`` was carried onto the verdict
  and reported as if it were a share of area;
- a non-:class:`Caveat` in ``extra_caveats`` surfaced only later, as a
  ``KeyError`` from :meth:`ValidityVerdict.summary` — far from the call that
  introduced it.

These tests exist because the refactor that added the guards was verified for
*behaviour preservation* against a bit-exact fingerprint, which by construction
could not cover newly-rejected inputs.
"""

from __future__ import annotations

import math

import pytest

from bunkershot3d.solvers.envelope import Caveat, EnvelopeContext
from bunkershot3d.solvers.exceptions import SolverInputError

pytestmark = pytest.mark.unit


class TestClampedAreaFraction:
    """It is a share of area, so it must lie in [0, 1]."""

    @pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
    def test_accepts_the_closed_unit_interval(self, fraction: float) -> None:
        assert EnvelopeContext(
            clamped_area_fraction=fraction
        ).clamped_area_fraction == (fraction)

    @pytest.mark.parametrize("fraction", [-0.1, 1.0000001, 2.0, -1.0])
    def test_rejects_values_outside_it(self, fraction: float) -> None:
        with pytest.raises(SolverInputError, match=r"clamped_area_fraction"):
            EnvelopeContext(clamped_area_fraction=fraction)

    @pytest.mark.parametrize("fraction", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite(self, fraction: float) -> None:
        """NaN is the dangerous one: every ordering comparison is False.

        A bare ``0.0 <= x <= 1.0`` bound would admit NaN, which then
        propagates into the verdict as a silently unusable number.
        """
        with pytest.raises(SolverInputError, match=r"clamped_area_fraction"):
            EnvelopeContext(clamped_area_fraction=fraction)


class TestExtraCaveats:
    """A caveat list holding a non-Caveat used to fail far from its cause."""

    def test_rejects_a_non_caveat_member(self) -> None:
        with pytest.raises(SolverInputError):
            EnvelopeContext(extra_caveats=("not a caveat",))  # type: ignore[arg-type]

    def test_accepts_real_caveats(self) -> None:
        caveat = next(iter(Caveat))
        context = EnvelopeContext(extra_caveats=(caveat,))
        assert context.extra_caveats == (caveat,)

    def test_defaults_to_empty(self) -> None:
        assert EnvelopeContext().extra_caveats == ()


class TestGravity:
    """The field the dimensionless groups are formed in."""

    @pytest.mark.parametrize("gravity", [0.0, -9.81, math.nan, math.inf])
    def test_rejects_non_positive_or_non_finite(self, gravity: float) -> None:
        with pytest.raises(SolverInputError, match=r"gravity"):
            EnvelopeContext(gravity_m_s2=gravity)

    def test_defaults_to_the_package_gravity_constant(self) -> None:
        from bunkershot3d.solvers.envelope import GRAVITY_M_S2

        assert EnvelopeContext().gravity_m_s2 == GRAVITY_M_S2


def test_guards_survive_python_dash_o() -> None:
    """The guards are ``raise``, not ``assert``.

    ``python -O`` strips ``assert``. A validity guard that evaporates under
    optimisation is worse than no guard, because callers believe it ran.
    """
    import os
    import subprocess
    import sys

    source = (
        "from bunkershot3d.solvers.envelope import EnvelopeContext\n"
        "from bunkershot3d.solvers.exceptions import SolverInputError\n"
        "try:\n"
        "    EnvelopeContext(clamped_area_fraction=2.0)\n"
        "except SolverInputError:\n"
        "    print('RAISED')\n"
    )
    # The child is a bare interpreter and inherits none of pytest's pythonpath
    # entries. Importing bunkershot3d pulls in the backends subpackage, which
    # needs the repo's shared roots too, so hand over this process's whole
    # sys.path rather than guessing which entries matter.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)

    result = subprocess.run(
        [sys.executable, "-O", "-c", source],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert "RAISED" in result.stdout, result.stderr
