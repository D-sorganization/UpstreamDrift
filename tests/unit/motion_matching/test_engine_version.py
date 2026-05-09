"""Unit tests for the ``engine_version()`` provider hook (issue #4705).

The :class:`FitSwingProvider` Protocol gained an optional
``engine_version()`` method so that two cross-engine leaderboard rows
produced against different engine wheels stay distinguishable. The
default-implementation contract is:

* Subclassing the Protocol without overriding the method MUST yield
  ``"unknown"`` (back-compat for providers predating #4705).
* Each shipped provider MUST override and return the underlying
  engine's version string when the engine wheel is installed.
* When the wheel is *not* installed, the provider MUST still return
  ``"unknown"`` (rather than raising) so callers can construct the
  provider in engine-less environments.
"""

from __future__ import annotations

import importlib.util
from unittest.mock import MagicMock

import pytest

from src.shared.python.motion_matching.provider import FitSwingProvider


def _engine_really_installed(name: str) -> bool:
    """Return ``True`` only when ``name`` resolves to a real, non-mocked install.

    The unit-test conftest pre-populates ``sys.modules`` with
    :class:`MagicMock` shims for ``pydrake`` and ``pinocchio`` so that
    Drake / Pinocchio test files can be collected on a CI runner that
    does not actually have those wheels. ``pytest.importorskip`` would
    consider those shims importable and run the test against the mock,
    which is not what we want here -- the engine-version assertion is
    *about* the real wheel being installed.
    """
    import sys

    mod = sys.modules.get(name)
    if isinstance(mod, MagicMock):
        return False
    if mod is not None:
        # Real module already imported (not a mock) -- definitely installed.
        return True
    # Not yet imported. Probe the real loader.
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return False
    return spec is not None


class _StubProviderNoOverride(FitSwingProvider):
    """Stub provider that does NOT override ``engine_version``.

    Inherits the Protocol's default behaviour, which #4705 requires to
    be ``"unknown"``.
    """

    engine_name = "stub"

    def fit_swing(self, target, opts):  # pragma: no cover -- never called
        raise NotImplementedError

    def supports_body_target(self) -> bool:  # pragma: no cover -- never called
        return False

    def supports_ball_target(self) -> bool:  # pragma: no cover -- never called
        return False


class TestProtocolDefault:
    """The Protocol's own default implementation returns ``"unknown"``."""

    def test_protocol_default_returns_unknown(self) -> None:
        # ``FitSwingProvider.engine_version`` is a regular function in the
        # Protocol body, so it can be invoked as an unbound method on any
        # object. We pass a sentinel because the default does not touch
        # ``self`` -- this isolates "what does the default body return?"
        # from any subclass behaviour.
        sentinel = object()
        assert FitSwingProvider.engine_version(sentinel) == "unknown"  # type: ignore[arg-type]

    def test_subclass_without_override_returns_unknown(self) -> None:
        """A provider that does not override the method inherits ``"unknown"``."""
        provider = _StubProviderNoOverride()
        assert provider.engine_version() == "unknown"


class TestMujocoProvider:
    """The MuJoCo provider returns a real version when ``mujoco`` is installed."""

    def test_mujoco_provider_advertises_version(self) -> None:
        if not _engine_really_installed("mujoco"):
            pytest.skip("mujoco not installed")
        import mujoco

        from src.engines.physics_engines.mujoco.python.motion_matching import (
            MujocoFitSwingProvider,
        )

        version = MujocoFitSwingProvider().engine_version()
        assert isinstance(version, str)
        assert version != "unknown"
        # When mujoco is importable, the provider should surface the
        # exact ``__version__`` string to keep leaderboard rows
        # bit-reproducible across runs against the same wheel.
        assert version == mujoco.__version__


class TestDrakeProvider:
    """The Drake provider returns a real version when ``pydrake`` is installed."""

    def test_drake_provider_advertises_version(self) -> None:
        if not _engine_really_installed("pydrake"):
            pytest.skip(
                "pydrake not installed (the unit-test conftest mock is ignored)"
            )
        import pydrake

        from src.engines.physics_engines.drake.python.motion_matching.provider import (
            DrakeFitSwingProvider,
        )

        version = DrakeFitSwingProvider().engine_version()
        assert isinstance(version, str)
        assert version != "unknown"
        if isinstance(getattr(pydrake, "__version__", None), str):
            assert version == pydrake.__version__

    def test_drake_provider_returns_unknown_without_engine(self) -> None:
        """Provider must stay queryable without a real Drake wheel."""
        if _engine_really_installed("pydrake"):
            pytest.skip("pydrake is installed; cannot test the missing-wheel path")
        from src.engines.physics_engines.drake.python.motion_matching.provider import (
            DrakeFitSwingProvider,
        )

        # The conftest installs a MagicMock under ``pydrake``; the provider
        # must not be fooled by that and must still return ``"unknown"``
        # because MagicMock's auto-attributes are not real version strings.
        assert DrakeFitSwingProvider().engine_version() == "unknown"


class TestPinocchioProvider:
    """The Pinocchio provider returns a real version when installed."""

    def test_pinocchio_provider_advertises_version(self) -> None:
        if not _engine_really_installed("pinocchio"):
            pytest.skip(
                "pinocchio not installed (the unit-test conftest mock is ignored)"
            )
        import pinocchio

        from src.engines.physics_engines.pinocchio.python.motion_matching.provider import (
            PinocchioFitSwingProvider,
        )

        version = PinocchioFitSwingProvider().engine_version()
        assert isinstance(version, str)
        assert version != "unknown"
        if isinstance(getattr(pinocchio, "__version__", None), str):
            assert version == pinocchio.__version__

    def test_pinocchio_provider_returns_unknown_without_engine(self) -> None:
        """Provider must stay queryable without a real Pinocchio wheel."""
        if _engine_really_installed("pinocchio"):
            pytest.skip("pinocchio is installed; cannot test the missing-wheel path")
        from src.engines.physics_engines.pinocchio.python.motion_matching.provider import (
            PinocchioFitSwingProvider,
        )

        assert PinocchioFitSwingProvider().engine_version() == "unknown"
